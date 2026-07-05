"""
Two-handed gesture orchestra -- SAMPLE version (plays real audio files).

Instead of synthesizing the timbre, each note plays a looped audio recording
from the samples/ folder (samples/G3.wav ... samples/C5.wav; .mp3/.flac/.ogg
also work). Any missing note falls back to a soft synth tone so it always runs.
Controlled by both hands at once:
    RIGHT index finger x -> pitch   (snapped to the white-key scale G3..C5)
    LEFT hand openness   -> volume  (fist = silent, spread = full)

Run:
    ./venv/bin/python gesture_orchestra_samples.py
Press ESC in the video window to quit. Press 'c' to (re)calibrate the left
hand's fist/open range to your own hand.

Which hand is which
-------------------
We mirror the camera (selfie view), so your left hand appears on the left of
the screen and MediaPipe's handedness labels match your real hands. If they
come out swapped on your setup, flip SWAP_HANDS below.

How the controls are measured
-----------------------------
Pitch: the horizontal position (x) of the RIGHT index fingertip, snapped to
the white-key scale G3..C5 -- far left = G3, far right = C5.
Volume: a scale-invariant openness ratio of the LEFT hand -- for the 4 long
fingers, each fingertip's distance to its base knuckle, divided by palm size.
Both are independent of how far the hand is from the camera.
"""

import math
import os
import threading

import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import soundfile as sf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---- Audio config -------------------------------------------------------
SAMPLE_RATE = 44100
TONE_HZ = 220.0          # starting pitch of the continuous tone (A3)
MAX_AMPLITUDE = 0.42     # overall loudness ceiling at 100% volume (soft)
PITCH_MIN_HZ = 55.0      # keyboard pitch control clamps to this range
PITCH_MAX_HZ = 2000.0
SEMITONE = 2.0 ** (1.0 / 12.0)   # one-semitone frequency multiplier

# ---- Timbre: soft, graceful string ensemble -----------------------------
# A warm, gentle sound ("柔和优美") rather than a big tutti. We drop the harsh
# extremes -- no contrabass rumble, no brass blare, no piccolo shrillness --
# and keep warm strings around the played note with just a little octave
# colour. Each section is a pool of independent players (own detuning, vibrato
# rate/phase and harmonic phases), so they bloom into a soft, breathing
# ensemble. Warm (low) rolloff values keep the highs mellow. Voices are panned
# across stereo for width; it all renders live so pitch/volume stay instant.
N_HARMONICS = 16
HARMONIC_K = np.arange(1, N_HARMONICS + 1)
ANTIALIAS_HZ = 0.45 * SAMPLE_RATE           # drop harmonics above this (no aliasing)

# Each section: (octave multiplier, brightness rolloff, n_voices, gain/voice).
# Bigger rolloff = brighter; lower octave = deeper. Low rolloffs here keep the
# tone soft and warm rather than edgy.
SECTIONS = [
    (0.5,  4.0, 3, 0.45),    # soft lower octave (cello-ish)  -> gentle warmth
    (1.0,  5.5, 10, 0.70),   # warm strings (unison)          -> mellow body
    (2.0,  5.0, 3, 0.22),    # soft octave up                 -> airy sheen
]

# Expand sections into flat per-voice arrays.
_oct, _gain, _roll = [], [], []
for _octave, _rolloff, _n, _g in SECTIONS:
    _oct += [_octave] * _n
    _gain += [_g] * _n
    _roll += [_rolloff] * _n
VOICE_OCTAVES = np.array(_oct)
VOICE_GAINS = np.array(_gain)
N_VOICES = len(VOICE_OCTAVES)
# Per-voice harmonic amplitude profile (1/k overtones with the section rolloff).
HARM_A = np.array([(1.0 / HARMONIC_K) * np.exp(-(HARMONIC_K - 1) / r) for r in _roll])

DETUNE_SPREAD_CENTS = 6.0                   # std-dev of random per-voice detune
VIB_RATE_RANGE = (4.5, 6.0)                 # gentle vibrato speed (Hz)
VIB_DEPTH_RANGE = (0.003, 0.006)            # gentle vibrato depth

# Make-up gain after normalization; tuned so the worst-case peak across all
# notes stays clear of clipping (verified by a sweep).
_TIMBRE_GAIN = 4.5
_TIMBRE_NORM = _TIMBRE_GAIN / (HARM_A.sum(axis=1) * VOICE_GAINS).sum()

_NOTE_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_freq(note):
    """'A4' / 'A#4' / 'Db5' -> frequency in Hz (A4 = 440)."""
    semis = _NOTE_SEMITONE[note[0].upper()]
    i = 1
    if note[i] in "#b":
        semis += 1 if note[i] == "#" else -1
        i += 1
    midi = (int(note[i:]) + 1) * 12 + semis
    return 440.0 * (SEMITONE ** (midi - 69))


# ---- Right index-finger x -> pitch (C major scale, G3..C5) -------------
# The RIGHT hand's index-fingertip horizontal position selects a white-key note
# (no sharps): far left = G3, far right = C5. Each scale step gets an equal
# slice of the horizontal range.
SWAP_HANDS = False               # set True if your left/right come out swapped
INDEX_TIP = 8                    # MediaPipe landmark: index fingertip
_WHITE = ["C", "D", "E", "F", "G", "A", "B"]


def white_range(low, high):
    """White-key note names from low to high inclusive, e.g. 'G3'..'C5'."""
    notes = []
    letter, octv, i = low[0], int(low[1:]), _WHITE.index(low[0])
    while True:
        notes.append(f"{_WHITE[i]}{octv}")
        if notes[-1] == high:
            return notes
        i = (i + 1) % 7
        if i == 0:
            octv += 1


SCALE_NOTES = white_range("G3", "C5")                  # G3 A3 B3 C4 ... B4 C5
SCALE_FREQS = [note_to_freq(n) for n in SCALE_NOTES]
PITCH_X_LEFT = 0.07              # index x at/left of this  -> lowest note (G3)
PITCH_X_RIGHT = 0.93            # index x at/right of this -> highest note (C5)
PITCH_RULER_Y = 0.40             # vertical position of the on-screen note ruler
# Robustness knobs (units are "notes"): while the finger moves faster than
# PITCH_MOVE_SPEED per frame it is treated as in-transit and the pitch is held,
# so a fast reposition snaps straight to the destination note instead of
# glissando-ing through the notes in between. PITCH_HYSTERESIS is an extra
# deadband beyond the half-note boundary, so a small tremble never flips the
# note -- the finger must clearly cross into the next note to change it.
PITCH_MOVE_SPEED = 0.28          # notes/frame above which the finger is moving
PITCH_HYSTERESIS = 0.20          # extra margin (notes) beyond the 0.5 boundary


def index_x_to_pos(x):
    """Continuous scale position (0 .. len-1) from the index fingertip's x."""
    t = (x - PITCH_X_LEFT) / (PITCH_X_RIGHT - PITCH_X_LEFT)   # 0 left, 1 right
    t = float(np.clip(t, 0.0, 1.0))
    return t * (len(SCALE_FREQS) - 1)


# ---- Openness -> volume calibration ------------------------------------
# Raw ratio values that map to 0% and 100%. Defaults work for most hands;
# press 'c' in the app to recalibrate to yours.
OPEN_RATIO_FIST = 0.55   # ratio when making a fist  -> 0%
OPEN_RATIO_OPEN = 1.30   # ratio when fully spread   -> 100%

SMOOTHING = 0.25         # 0..1, lower = smoother/slower volume changes

# Finger (tip_idx, mcp_idx) pairs for the 4 long fingers.
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]


# ---- Sample bank: one audio file per scale note -------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
CROSSFADE_SEC = 0.03     # crossfade when switching notes (click-free)
MASTER_GAIN = 0.9        # overall output level


def _load_one(path):
    """Read an audio file -> (N, 2) float32 stereo at SAMPLE_RATE."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:                        # mono -> stereo
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if sr != SAMPLE_RATE:                          # linear resample to our SR
        n = int(round(data.shape[0] * SAMPLE_RATE / sr))
        xp = np.linspace(0.0, 1.0, data.shape[0])
        x = np.linspace(0.0, 1.0, n)
        data = np.stack([np.interp(x, xp, data[:, 0]),
                         np.interp(x, xp, data[:, 1])], axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def _fallback_tone(freq):
    """A soft, seamlessly-loopable tone for any note that has no sample file."""
    periods = max(1, round(freq * 1.0))            # ~1s of whole periods -> loops clean
    length = int(round(periods * SAMPLE_RATE / freq))
    t = np.arange(length) / SAMPLE_RATE
    wave = (0.6 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(4 * np.pi * freq * t) +
            0.12 * np.sin(6 * np.pi * freq * t)) * 0.35
    return np.repeat(wave[:, None].astype(np.float32), 2, axis=1)


def load_samples():
    """One sample per scale note. Uses samples/<NOTE>.(wav|mp3|flac|ogg|aiff)
    when present, otherwise a synth fallback tone so the app runs with no files."""
    exts = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
    samples, missing = [], []
    for name, freq in zip(SCALE_NOTES, SCALE_FREQS):
        path = next((os.path.join(SAMPLE_DIR, name + e)
                     for e in exts if os.path.exists(os.path.join(SAMPLE_DIR, name + e))), None)
        if path:
            try:
                samples.append(_load_one(path))
                continue
            except Exception as ex:
                print(f"  ! could not load {path}: {ex}")
        missing.append(name)
        samples.append(_fallback_tone(freq))
    if missing:
        print(f"[samples] no file for: {', '.join(missing)} -> using synth fallback.")
        print(f"[samples] drop recordings named e.g. G3.wav or C4.mp3 into: {SAMPLE_DIR}")
    else:
        print(f"[samples] loaded all {len(samples)} note samples from {SAMPLE_DIR}")
    return samples


class SamplePlayer:
    """Loops the current note's sample; volume set live; crossfades on note
    change so switches are click-free."""

    def __init__(self, samples, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.samples = samples
        self._lock = threading.Lock()
        self._target_idx = len(samples) // 2
        self._target_vol = 0.0
        self._cur_vol = 0.0
        self._xfade = max(1, int(CROSSFADE_SEC * sample_rate))
        # Active voices, each: {idx, pos, gain, tgt}. Only the audio thread
        # mutates this list, so it needs no lock.
        self._voices = [{"idx": self._target_idx, "pos": 0, "gain": 1.0, "tgt": 1.0}]
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=2,
            callback=self._callback, blocksize=1024)

    def set_note(self, idx):
        with self._lock:
            self._target_idx = int(idx)

    def set_volume(self, vol):
        with self._lock:
            self._target_vol = float(np.clip(vol, 0.0, 1.0))

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            target_idx = self._target_idx
            target_vol = self._target_vol

        # Note changed -> fade out the old voice(s), fade in the new one.
        if not self._voices or self._voices[-1]["idx"] != target_idx:
            for vc in self._voices:
                vc["tgt"] = 0.0
            self._voices.append({"idx": target_idx, "pos": 0, "gain": 0.0, "tgt": 1.0})

        out = np.zeros((frames, 2), dtype=np.float32)
        step = frames / self._xfade                 # max gain change this block
        for vc in list(self._voices):
            buf = self.samples[vc["idx"]]
            L = len(buf)
            seg = buf[(vc["pos"] + np.arange(frames)) % L]   # looped read
            vc["pos"] = int((vc["pos"] + frames) % L)
            g0 = vc["gain"]
            g1 = g0 + float(np.clip(vc["tgt"] - g0, -step, step))
            out += seg * np.linspace(g0, g1, frames)[:, None]
            vc["gain"] = g1
            if vc["tgt"] == 0.0 and g1 <= 1e-4:
                self._voices.remove(vc)

        vol = np.linspace(self._cur_vol, target_vol, frames)[:, None]
        self._cur_vol = target_vol
        out *= vol * MASTER_GAIN
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:] = out

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


def hand_openness(landmarks):
    """Return a scale-invariant openness ratio for one hand."""
    def pt(i):
        lm = landmarks[i]
        return np.array([lm.x, lm.y, lm.z])

    palm = np.linalg.norm(pt(0) - pt(9))      # wrist -> middle knuckle
    if palm < 1e-6:
        return 0.0
    total = sum(np.linalg.norm(pt(tip) - pt(mcp)) for tip, mcp in FINGERS)
    return (total / len(FINGERS)) / palm


def ratio_to_volume(ratio, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((ratio - lo) / (hi - lo), 0.0, 1.0))


_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note_name(freq):
    """Nearest note name like 'A4' for a frequency (for the HUD)."""
    midi = round(69 + 12 * math.log2(freq / 440.0))
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def split_hands(results):
    """Return (pitch_hand, volume_hand) landmark objects, or None for each.

    Right hand -> pitch, left hand -> volume. We mirror the camera, so
    MediaPipe's handedness matches the user's real hands (flip SWAP_HANDS if
    not). If only one hand is seen it is assigned by its label.
    """
    pitch_hand = volume_hand = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label          # 'Left' or 'Right'
            if SWAP_HANDS:
                label = "Right" if label == "Left" else "Left"
            if label == "Right":
                pitch_hand = lms
            else:
                volume_hand = lms
    return pitch_hand, volume_hand


# Camera index. None = auto-detect the first camera that delivers live
# frames (skips a black/frozen Continuity Camera). Set an int to force one.
CAMERA_INDEX = None


def open_camera():
    """Open a webcam that delivers LIVE (non-black, non-frozen) frames.

    On macOS the first index is often a Continuity Camera (iPhone) that can
    hand back a black OR a frozen/static image -- the usual cause of a black or
    "strange static" window. So we warm up each index and only accept one whose
    frames are both bright enough AND actually changing over time.
    """
    indices = (CAMERA_INDEX,) if CAMERA_INDEX is not None else (0, 1, 2)
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        nonblack = False
        motion = 0.0
        prev = None
        for _ in range(40):                       # ~1.3s warm-up
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if float(frame.std()) > 5.0:
                nonblack = True
            if prev is not None:
                motion = max(motion, float(np.mean(
                    np.abs(frame.astype(np.int16) - prev.astype(np.int16)))))
            prev = frame
        if nonblack and motion > 0.4:             # live sensor noise/movement
            print(f"Using camera index {index} (live).")
            return cap
        reason = "only black frames" if not nonblack else "a frozen/static image (no motion)"
        print(f"Camera index {index}: {reason}; skipping.")
        cap.release()
    raise RuntimeError(
        "No camera delivered a LIVE image. Common macOS causes:\n"
        " - A Continuity Camera (iPhone) grabbed the camera and is showing a\n"
        "   black or frozen frame. Disable it: System Settings > General >\n"
        "   AirPlay & Handoff > turn OFF 'Continuity Camera'. Then retry.\n"
        " - A stuck camera daemon: run  sudo killall VDCAssistant  then retry.\n"
        " - Camera permission: System Settings > Privacy & Security > Camera\n"
        "   -> enable your terminal app, then fully quit (Cmd+Q) and reopen it.\n"
        " - Another app is using the camera (Zoom / FaceTime / Photo Booth)."
    )


def draw_pitch_ruler(frame, committed_idx, finger_x=None):
    """Draw a labeled note ruler you can aim the index finger at.

    A horizontal line spans the pitch zone; each note gets a tick + label, the
    current note is highlighted, and a red pointer marks the finger's x.
    """
    h, w, _ = frame.shape
    y = int(h * PITCH_RULER_Y)
    x_lo, x_hi = int(PITCH_X_LEFT * w), int(PITCH_X_RIGHT * w)
    cv2.line(frame, (x_lo, y), (x_hi, y), (235, 235, 235), 3)

    n = len(SCALE_NOTES)
    for i, name in enumerate(SCALE_NOTES):
        x = int(x_lo + (i / (n - 1)) * (x_hi - x_lo))
        active = (i == committed_idx)
        color = (0, 255, 255) if active else (235, 235, 235)
        tick = 24 if active else 18
        cv2.line(frame, (x, y - tick), (x, y + tick), color, 4 if active else 3)
        fs = 0.85 if active else 0.7
        (tw, _th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(frame, name, (x - tw // 2, y - tick - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color, 2, cv2.LINE_AA)
        if active:
            cv2.circle(frame, (x, y), 9, (0, 255, 255), cv2.FILLED)

    if finger_x is not None:
        fx = int(float(np.clip(finger_x, PITCH_X_LEFT, PITCH_X_RIGHT)) * w)
        cv2.drawMarker(frame, (fx, y + 30), (0, 0, 255),
                       cv2.MARKER_TRIANGLE_UP, 28, 3)


def main():
    lo, hi = OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    cap = open_camera()

    with SamplePlayer(load_samples()) as player, mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        smoothed_vol = 0.0
        committed_idx = len(SCALE_FREQS) // 2   # the currently sounding scale note
        prev_pos = None                          # last frame's finger position
        player.set_note(committed_idx)
        current_note = SCALE_NOTES[committed_idx]

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            pitch_hand, volume_hand = split_hands(results)

            # RIGHT hand index-finger x -> pitch (left=low, right=high). Robust:
            # hold the note while the finger is moving (no glissando on a fast
            # reposition) and only commit a new note once it settles AND clearly
            # crosses a note boundary (hysteresis), so trembles never change it.
            if pitch_hand is not None:
                pos = index_x_to_pos(pitch_hand.landmark[INDEX_TIP].x)
                vel = 0.0 if prev_pos is None else abs(pos - prev_pos)
                prev_pos = pos
                if vel <= PITCH_MOVE_SPEED and abs(pos - committed_idx) > 0.5 + PITCH_HYSTERESIS:
                    committed_idx = int(round(pos))
                    player.set_note(committed_idx)
                    current_note = SCALE_NOTES[committed_idx]
                mp_drawing.draw_landmarks(
                    frame, pitch_hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                # Mark the index fingertip -- the point that sets the pitch.
                h0, w0, _ = frame.shape
                tip = pitch_hand.landmark[INDEX_TIP]
                cv2.circle(frame, (int(tip.x * w0), int(tip.y * h0)), 10, (0, 255, 255), 2)
            else:
                prev_pos = None       # reset so re-entry doesn't glide from a stale pos

            # LEFT hand openness -> volume (silent when not visible).
            target_vol = 0.0
            if volume_hand is not None:
                ratio = hand_openness(volume_hand.landmark)
                target_vol = ratio_to_volume(ratio, lo, hi)
                mp_drawing.draw_landmarks(
                    frame, volume_hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            smoothed_vol += SMOOTHING * (target_vol - smoothed_vol)
            player.set_volume(smoothed_vol)

            # ---- HUD ----
            h, w, _ = frame.shape
            # Note ruler: labeled line to aim the index finger at.
            finger_x = pitch_hand.landmark[INDEX_TIP].x if pitch_hand is not None else None
            draw_pitch_ruler(frame, committed_idx, finger_x)
            pct = int(round(smoothed_vol * 100))
            bar_w = int(smoothed_vol * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
            pitch_state = "tracking" if pitch_hand is not None else "--"
            vol_state = "tracking" if volume_hand is not None else "--"
            cv2.putText(frame, f"Note: {current_note}   Volume: {pct}%",
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"R index=pitch [{pitch_state}]   L hand=volume [{vol_state}]   ESC quit   c calibrate",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Orchestra - ESC quit, c calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                    # ESC -> quit
                break
            elif key == ord("c"):
                lo, hi = calibrate(cap, hands)

    cap.release()
    cv2.destroyAllWindows()


def calibrate(cap, hands):
    """Capture a fist then an open-hand sample (LEFT hand); return (lo, hi)."""
    samples = {}
    for stage, prompt in (("FIST", "LEFT hand: make a FIST, hold still, press SPACE"),
                          ("OPEN", "LEFT hand: SPREAD it, hold still, press SPACE")):
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            _, volume_hand = split_hands(res)
            ratio = hand_openness(volume_hand.landmark) if volume_hand is not None else None
            cv2.putText(frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if ratio is not None:
                cv2.putText(frame, f"ratio={ratio:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Orchestra - ESC quit, c calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and ratio is not None:
                samples[stage] = ratio
                break
            if key == 27:                      # ESC cancels calibration
                return OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    lo, hi = samples["FIST"], samples["OPEN"]
    if hi <= lo:                       # guard against a bad capture
        lo, hi = OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    print(f"Calibrated: fist={lo:.2f} (0%), open={hi:.2f} (100%)")
    return lo, hi


if __name__ == "__main__":
    main()
