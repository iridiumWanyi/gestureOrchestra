"""
Two-handed gesture orchestra (theremin-style).

A continuous orchestral tutti plays through your speakers, controlled by both
hands at once:
    RIGHT index finger x -> pitch   (move left = lower, move right = higher)
    LEFT hand openness   -> volume  (fist = silent, spread = full)

Run:
    ./venv/bin/python gesture_orchestra.py
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
import threading

import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd

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
MAX_CHORD = 4            # max simultaneous chord tones
CHORD_FADE_SEC = 0.03    # fade a chord tone in/out when the chord changes (no click)

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
# (Lighter than the single-note version: chords render this ensemble PER note,
# so we keep the voice count modest to stay well within the audio-time budget.)
SECTIONS = [
    (0.5,  4.0, 1, 0.45),    # soft lower octave (cello-ish)  -> gentle warmth
    (1.0,  5.5, 5, 0.70),    # warm strings (unison)          -> mellow body
    (2.0,  5.0, 1, 0.22),    # soft octave up                 -> airy sheen
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


# ---- Two-row pitch: white keys (lower line) + black keys (upper line) ----
# The RIGHT index finger's HORIZONTAL position picks the note; its VERTICAL
# position picks the ROW -- raise it to the upper line for black keys (sharps),
# lower it for the white keys. Range C3..C5.
SWAP_HANDS = False               # set True if your left/right come out swapped
INDEX_TIP = 8                    # MediaPipe landmark: index fingertip
_WHITE = ["C", "D", "E", "F", "G", "A", "B"]

PITCH_X_LEFT = 0.05              # note zone: far left  = C3
PITCH_X_RIGHT = 0.95            #            far right = C5
RULER_Y_BLACK = 0.30             # upper line (black keys / sharps)
RULER_Y_WHITE = 0.45             # lower line (white keys / naturals)
ROW_SPLIT_Y = 0.375              # finger y above this -> black row, below -> white
SETTLE_SPEED = 0.015             # hold the note while the finger moves faster than
                                 # this (no glissando); commit once it settles


def white_range(low, high):
    """White-key note names from low to high inclusive, e.g. 'C3'..'C5'."""
    notes = []
    letter, octv, i = low[0], int(low[1:]), _WHITE.index(low[0])
    while True:
        notes.append(f"{_WHITE[i]}{octv}")
        if notes[-1] == high:
            return notes
        i = (i + 1) % 7
        if i == 0:
            octv += 1


def build_keys():
    """White row C3..C5 (evenly spaced) + black row (sharps at white midpoints)."""
    names = white_range("C3", "C5")
    nW = len(names)
    white = [{"name": n, "freq": note_to_freq(n),
              "x": PITCH_X_LEFT + k / (nW - 1) * (PITCH_X_RIGHT - PITCH_X_LEFT)}
             for k, n in enumerate(names)]
    black = []
    for k in range(nW - 1):
        if round(12 * math.log2(white[k + 1]["freq"] / white[k]["freq"])) == 2:
            nm = white[k]["name"][0] + "#" + white[k]["name"][1:]
            black.append({"name": nm, "freq": note_to_freq(nm),
                          "x": (white[k]["x"] + white[k + 1]["x"]) / 2})
    return white, black


WHITE_KEYS, BLACK_KEYS = build_keys()


def pick_key(fx, fy):
    """Nearest key to finger x, in the row chosen by finger y (up = black)."""
    row = BLACK_KEYS if fy < ROW_SPLIT_Y else WHITE_KEYS
    best = min(row, key=lambda k: abs(k["x"] - fx))
    return best, (row is BLACK_KEYS)


# ---- Chords: right-hand finger count -> chord built on the pointed note ---
# Value = (semitone offsets from the pointed root note, label). Index only = 1
# finger = single note; each extra finger adds a richer chord.
CHORDS = {
    1: ([0], "single"),
    2: ([0, 4, 7], "major"),
    3: ([0, 3, 7], "minor"),
    4: ([0, 4, 7, 10], "dom7"),
}
# (tip, pip) for the four long fingers -- extended if the tip is farther from
# the wrist than its middle joint (works regardless of hand orientation).
_EXT_FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def count_extended(landmarks):
    """How many of index/middle/ring/pinky are extended (0..4)."""
    w = landmarks[0]
    def far(i):
        return math.hypot(landmarks[i].x - w.x, landmarks[i].y - w.y)
    return sum(1 for tip, pip in _EXT_FINGERS if far(tip) > far(pip))


# ---- Openness -> volume calibration ------------------------------------
# Raw ratio values that map to 0% and 100%. Defaults work for most hands;
# press 'c' in the app to recalibrate to yours.
OPEN_RATIO_FIST = 0.55   # ratio when making a fist  -> 0%
OPEN_RATIO_OPEN = 1.30   # ratio when fully spread   -> 100%

SMOOTHING = 0.25         # 0..1, lower = smoother/slower volume changes

# Finger (tip_idx, mcp_idx) pairs for the 4 long fingers.
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]


class TonePlayer:
    """Polyphonic soft-ensemble synth. Plays up to MAX_CHORD notes at once, each
    rendered with the full ensemble; chord tones fade in/out click-free."""

    def __init__(self, sample_rate=SAMPLE_RATE, freq=TONE_HZ):
        self.sample_rate = sample_rate
        self._target_vol = 0.0
        self._cur_vol = 0.0
        # Chord state: one frequency + fade-gain target per slot.
        self._chord_freqs = np.full(MAX_CHORD, freq, dtype=float)
        self._slot_target = np.zeros(MAX_CHORD)   # 1 = slot in use, 0 = fading out
        self._slot_target[0] = 1.0
        self._slot_gain = np.zeros(MAX_CHORD)     # smoothed, advanced in callback
        self._lock = threading.Lock()

        # Fixed random character per ensemble voice (seeded for a stable timbre).
        rng = np.random.default_rng(7)
        self._detune = 2.0 ** (rng.normal(0.0, DETUNE_SPREAD_CENTS, N_VOICES) / 1200.0)
        self._vib_rate = rng.uniform(*VIB_RATE_RANGE, size=N_VOICES)
        self._vib_depth = rng.uniform(*VIB_DEPTH_RANGE, size=N_VOICES)
        self._harm_off = rng.uniform(0.0, 2.0 * np.pi, size=(N_VOICES, N_HARMONICS))
        pos = np.linspace(0.0, 1.0, N_VOICES)
        rng.shuffle(pos)
        theta = pos * (np.pi / 2.0)
        self._pan_l = np.cos(theta)
        self._pan_r = np.sin(theta)

        # Phase state per (chord slot, voice), so each note stays click-free.
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=(MAX_CHORD, N_VOICES))
        self._vib_phases = rng.uniform(0.0, 2.0 * np.pi, size=(MAX_CHORD, N_VOICES))

        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=2,
            callback=self._callback, blocksize=1024)

    def set_volume(self, vol):
        with self._lock:
            self._target_vol = float(np.clip(vol, 0.0, 1.0))

    def set_chord(self, freqs):
        """Set the sounding notes (list of frequencies, 1..MAX_CHORD)."""
        with self._lock:
            n = max(1, min(len(freqs), MAX_CHORD))
            for i in range(n):
                self._chord_freqs[i] = float(np.clip(freqs[i], PITCH_MIN_HZ, PITCH_MAX_HZ))
                self._slot_target[i] = 1.0
            for i in range(n, MAX_CHORD):
                self._slot_target[i] = 0.0        # fade out (keep last freq -> no click)

    def set_freq(self, freq):
        self.set_chord([freq])

    @property
    def freq(self):
        with self._lock:
            return float(self._chord_freqs[0])

    def _render_note(self, slot, freq, frames, idx, two_pi, sr):
        left = np.zeros(frames)
        right = np.zeros(frames)
        for v in range(N_VOICES):
            vib = 1.0 + self._vib_depth[v] * np.sin(
                self._vib_phases[slot, v] + two_pi * self._vib_rate[v] * idx / sr)
            self._vib_phases[slot, v] = float(
                (self._vib_phases[slot, v] + two_pi * self._vib_rate[v] * frames / sr) % two_pi)
            inst = freq * VOICE_OCTAVES[v] * self._detune[v] * vib
            phase = self._phases[slot, v] + np.cumsum(two_pi * inst / sr)
            self._phases[slot, v] = float(phase[-1] % two_pi)
            a_v = HARM_A[v] * (HARMONIC_K * freq * VOICE_OCTAVES[v] < ANTIALIAS_HZ)
            harm = np.sin(phase[:, None] * HARMONIC_K[None, :] + self._harm_off[v][None, :])
            voice = (harm @ a_v) * VOICE_GAINS[v]
            left += voice * self._pan_l[v]
            right += voice * self._pan_r[v]
        return left * _TIMBRE_NORM, right * _TIMBRE_NORM

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            target_vol = self._target_vol
            chord_freqs = self._chord_freqs.copy()
            slot_target = self._slot_target.copy()

        sr = self.sample_rate
        idx = np.arange(frames)
        two_pi = 2.0 * math.pi
        fade_step = frames / (CHORD_FADE_SEC * sr)

        left = np.zeros(frames)
        right = np.zeros(frames)
        active_gain = 0.0
        for s in range(MAX_CHORD):
            g0 = self._slot_gain[s]
            g1 = g0 + float(np.clip(slot_target[s] - g0, -fade_step, fade_step))
            self._slot_gain[s] = g1
            if g0 <= 1e-4 and g1 <= 1e-4:
                continue                          # silent slot -> skip (phases frozen)
            l, r = self._render_note(s, chord_freqs[s], frames, idx, two_pi, sr)
            gain = np.linspace(g0, g1, frames)
            left += l * gain
            right += r * gain
            active_gain += g1

        # Normalize by how many notes are sounding so a chord isn't louder than
        # a single note, then ramp master volume per sample.
        norm = 1.0 / math.sqrt(max(1.0, active_gain))
        amp = np.linspace(self._cur_vol, target_vol, frames) * norm
        self._cur_vol = target_vol

        outdata[:, 0] = np.clip(left * amp * MAX_AMPLITUDE, -1.0, 1.0).astype(np.float32)
        outdata[:, 1] = np.clip(right * amp * MAX_AMPLITUDE, -1.0, 1.0).astype(np.float32)

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


def _draw_ruler_row(frame, keys, y_frac, active_key, is_active_row, finger_x):
    h, w, _ = frame.shape
    y = int(h * y_frac)
    x_lo, x_hi = int(PITCH_X_LEFT * w), int(PITCH_X_RIGHT * w)
    base = (235, 235, 235) if is_active_row else (120, 120, 120)
    cv2.line(frame, (x_lo, y), (x_hi, y), base, 3 if is_active_row else 2)
    for key in keys:
        x = int(key["x"] * w)
        active = (key is active_key)
        color = (0, 255, 255) if active else base
        tick = 20 if active else 12
        cv2.line(frame, (x, y - tick), (x, y + tick), color, 4 if active else 2)
        fs = 0.7 if active else 0.5
        (tw, _th), _ = cv2.getTextSize(key["name"], cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(frame, key["name"], (x - tw // 2, y - tick - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
        if active:
            cv2.circle(frame, (x, y), 7, (0, 255, 255), cv2.FILLED)
    if is_active_row and finger_x is not None:
        fx = int(float(np.clip(finger_x, PITCH_X_LEFT, PITCH_X_RIGHT)) * w)
        cv2.drawMarker(frame, (fx, y + 24), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 24, 3)


def draw_pitch_ruler(frame, active_key, is_black, finger_x=None):
    """Two rows: upper line = black keys (sharps), lower = white keys. The active
    row (chosen by the finger's height) is bright; the current note highlighted."""
    _draw_ruler_row(frame, BLACK_KEYS, RULER_Y_BLACK,
                    active_key if is_black else None, is_black,
                    finger_x if is_black else None)
    _draw_ruler_row(frame, WHITE_KEYS, RULER_Y_WHITE,
                    None if is_black else active_key, not is_black,
                    None if is_black else finger_x)


def main():
    lo, hi = OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    cap = open_camera()

    with TonePlayer() as tone, mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        smoothed_vol = 0.0
        committed_key = WHITE_KEYS[len(WHITE_KEYS) // 2]   # currently pointed key
        committed_black = False
        chord_label = "single"
        prev_pt = None                           # last frame's fingertip (x, y)
        tone.set_chord([committed_key["freq"]])
        current_note = committed_key["name"]

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            pitch_hand, volume_hand = split_hands(results)

            # RIGHT hand: index x picks the note, y picks the row (up=black keys,
            # down=white). Finger count picks the chord (index only = single note).
            if pitch_hand is not None:
                tip = pitch_hand.landmark[INDEX_TIP]
                p = np.array([tip.x, tip.y])
                speed = 0.0 if prev_pt is None else float(np.hypot(*(p - prev_pt)))
                prev_pt = p
                target_key, target_black = pick_key(tip.x, tip.y)
                # Hold while moving fast (no glissando); commit once it settles.
                if speed <= SETTLE_SPEED:
                    committed_key, committed_black = target_key, target_black
                    current_note = committed_key["name"]
                offsets, chord_label = CHORDS.get(count_extended(pitch_hand.landmark),
                                                  ([0], "single"))
                root = committed_key["freq"]
                tone.set_chord([root * (SEMITONE ** o) for o in offsets])
                mp_drawing.draw_landmarks(
                    frame, pitch_hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                h0, w0, _ = frame.shape
                cv2.circle(frame, (int(tip.x * w0), int(tip.y * h0)), 10, (0, 255, 255), 2)
            else:
                prev_pt = None

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
            tone.set_volume(smoothed_vol)

            # ---- HUD ----
            h, w, _ = frame.shape
            # Two-row note ruler (black keys upper, white keys lower).
            finger_x = pitch_hand.landmark[INDEX_TIP].x if pitch_hand is not None else None
            draw_pitch_ruler(frame, committed_key, committed_black, finger_x)
            pct = int(round(smoothed_vol * 100))
            bar_w = int(smoothed_vol * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"Note: {current_note}   Chord: {chord_label}   Volume: {pct}%",
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "R index: x=note  up=black/down=white  fingers=chord   "
                        "L hand=volume   ESC quit  c calibrate",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

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
