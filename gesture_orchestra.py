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
Pitch: the horizontal position (x) of the RIGHT index fingertip, mapped
CONTINUOUSLY (theremin-style glide, no snapping) across G3..C5 -- far left =
G3, far right = C5.
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
# Portamento glide done INSIDE the audio callback (per block + ramped per
# sample), so pitch changes are smooth and independent of the camera frame
# rate -- this is what removes the "stepped/discrete" feel. 0..1 per block;
# lower = smoother/slower glide.
PITCH_GLIDE = 0.40

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


# ---- Right index-finger x -> pitch (continuous glide, G3..C5) ----------
# The RIGHT hand's index-fingertip horizontal position sets the pitch
# CONTINUOUSLY (theremin-style, no snapping to the scale): far left = G3, far
# right = C5, interpolated logarithmically so equal finger moves feel like
# equal musical intervals. The pitch is smoothed so it glides between notes.
SWAP_HANDS = False               # set True if your left/right come out swapped
INDEX_TIP = 8                    # MediaPipe landmark: index fingertip
_CHROM = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_midi(note):
    """Note name -> MIDI number, e.g. 'A4' -> 69, 'G#3' -> 56."""
    semis, i = _NOTE_SEMITONE[note[0].upper()], 1
    if note[i] in "#b":
        semis += 1 if note[i] == "#" else -1
        i += 1
    return (int(note[i:]) + 1) * 12 + semis


def chromatic_range(low, high):
    """Every semitone note name from low to high inclusive, e.g. 'G3'..'C5'."""
    return [f"{_CHROM[m % 12]}{m // 12 - 1}"
            for m in range(_note_midi(low), _note_midi(high) + 1)]


# Ruler notes are the full chromatic scale, so equal spacing = equal interval
# (one semitone per tick), matching the continuous log pitch mapping exactly.
SCALE_NOTES = chromatic_range("G3", "C5")              # G3, G#3, A3, ... B4, C5
SCALE_FREQS = [note_to_freq(n) for n in SCALE_NOTES]
PITCH_LOW_HZ = SCALE_FREQS[0]    # frequency at the far left  (G3)
PITCH_HIGH_HZ = SCALE_FREQS[-1]  # frequency at the far right (C5)
PITCH_X_LEFT = 0.15              # index x at/left of this  -> lowest pitch (G3)
PITCH_X_RIGHT = 0.85            # index x at/right of this -> highest pitch (C5)
PITCH_RULER_Y = 0.22             # vertical position of the on-screen note ruler
PITCH_SMOOTHING = 0.6            # light smoothing of the finger x: removes fast
                                 # jitter but keeps the slow natural tremble
# Pitch "gravity": pull the continuous pitch toward the nearest semitone so a
# held note locks in tune -- but the pull is only partial and FADES OUT as the
# finger moves. Result: slides stay smooth (glissando), a held note is in tune,
# and the small tremble that survives the pull becomes a gentle vibrato.
#   PITCH_ATTRACT        : max pull when the finger is still (0 = off/pure glide,
#                          1 = hard snap). 0.85 keeps ~15% of the tremble.
#   PITCH_ATTRACT_RELEASE: finger speed (semitones/frame) at which the pull
#                          reaches 0, freeing the slide.
PITCH_ATTRACT = 0.85
PITCH_ATTRACT_RELEASE = 0.5


def index_x_to_freq(x):
    """Continuous frequency from index fingertip x (log/musical interpolation)."""
    t = (x - PITCH_X_LEFT) / (PITCH_X_RIGHT - PITCH_X_LEFT)   # 0 left, 1 right
    t = float(np.clip(t, 0.0, 1.0))
    return PITCH_LOW_HZ * (PITCH_HIGH_HZ / PITCH_LOW_HZ) ** t


def attracted_freq(freq_raw, vel):
    """Pull a continuous frequency toward the nearest semitone by a velocity-
    dependent amount: strong when still (locks in tune, tremble -> vibrato),
    fading to zero as the finger moves (keeps the slide smooth)."""
    pos = 12.0 * math.log2(freq_raw / PITCH_LOW_HZ)      # semitones above G3
    attract = PITCH_ATTRACT * max(0.0, 1.0 - vel / PITCH_ATTRACT_RELEASE)
    nearest = round(pos)
    pos_out = nearest + (pos - nearest) * (1.0 - attract)  # partial pull to note
    return PITCH_LOW_HZ * 2.0 ** (pos_out / 12.0)


# ---- Openness -> volume calibration ------------------------------------
# Raw ratio values that map to 0% and 100%. Defaults work for most hands;
# press 'c' in the app to recalibrate to yours.
OPEN_RATIO_FIST = 0.55   # ratio when making a fist  -> 0%
OPEN_RATIO_OPEN = 1.30   # ratio when fully spread   -> 100%

SMOOTHING = 0.25         # 0..1, lower = smoother/slower volume changes

# Finger (tip_idx, mcp_idx) pairs for the 4 long fingers.
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]


class TonePlayer:
    """Plays a continuous orchestral-ensemble tone; pitch & volume set live."""

    def __init__(self, sample_rate=SAMPLE_RATE, freq=TONE_HZ):
        self.sample_rate = sample_rate
        self._freq = freq        # current (glided) frequency, updated in callback
        self._target_freq = freq # target set from the main thread (under _lock)
        self._target_vol = 0.0   # 0..1, set from the main thread
        self._cur_vol = 0.0      # smoothed inside the audio callback
        self._lock = threading.Lock()

        # Fixed random character for each player in the ensemble (seeded so the
        # timbre is consistent between runs).
        rng = np.random.default_rng(7)
        self._detune = 2.0 ** (rng.normal(0.0, DETUNE_SPREAD_CENTS, N_VOICES) / 1200.0)
        self._vib_rate = rng.uniform(*VIB_RATE_RANGE, size=N_VOICES)
        self._vib_depth = rng.uniform(*VIB_DEPTH_RANGE, size=N_VOICES)
        self._harm_off = rng.uniform(0.0, 2.0 * np.pi, size=(N_VOICES, N_HARMONICS))
        # Equal-power stereo pan, voices spread left->right for width.
        pos = np.linspace(0.0, 1.0, N_VOICES)
        rng.shuffle(pos)
        theta = pos * (np.pi / 2.0)
        self._pan_l = np.cos(theta)
        self._pan_r = np.sin(theta)

        # Running phase state (advance every block to stay click-free).
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)
        self._vib_phases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)

        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=2,
            callback=self._callback,
            blocksize=1024,
        )

    def set_volume(self, vol):
        with self._lock:
            self._target_vol = float(np.clip(vol, 0.0, 1.0))

    def set_freq(self, freq):
        # Sets the TARGET; the callback glides the actual frequency toward it.
        with self._lock:
            self._target_freq = float(np.clip(freq, PITCH_MIN_HZ, PITCH_MAX_HZ))

    @property
    def freq(self):
        with self._lock:
            return self._freq

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            target = self._target_vol
            target_freq = self._target_freq

        sr = self.sample_rate
        idx = np.arange(frames)
        two_pi = 2.0 * math.pi

        # Portamento: glide the base frequency toward the target and ramp it
        # smoothly across the block (per sample), so pitch never steps even if
        # the camera updates the target only ~30 times/sec.
        freq_start = self._freq
        freq_end = freq_start + PITCH_GLIDE * (target_freq - freq_start)
        base_freq = np.linspace(freq_start, freq_end, frames)   # per-sample Hz
        self._freq = freq_end

        left = np.zeros(frames)
        right = np.zeros(frames)
        for v in range(N_VOICES):
            # This player's own vibrato (independent rate/phase -> decorrelated,
            # which is what makes a section sound lush instead of mechanical).
            vib = 1.0 + self._vib_depth[v] * np.sin(
                self._vib_phases[v] + two_pi * self._vib_rate[v] * idx / sr)
            self._vib_phases[v] = float(
                (self._vib_phases[v] + two_pi * self._vib_rate[v] * frames / sr) % two_pi)

            # Instantaneous phase for this voice's fundamental, integrated so it
            # stays continuous across blocks even as pitch/vibrato move.
            inst_freq = base_freq * VOICE_OCTAVES[v] * self._detune[v] * vib
            phase = self._phases[v] + np.cumsum(two_pi * inst_freq / sr)
            self._phases[v] = float(phase[-1] % two_pi)

            # Harmonic stack with randomized harmonic phases (less "buzzy comb"),
            # dropping any overtone above Nyquist so the deep/high octaves don't
            # alias into harsh digital junk.
            fund = freq_end * VOICE_OCTAVES[v]
            a_v = HARM_A[v] * (HARMONIC_K * fund < ANTIALIAS_HZ)
            harm = np.sin(phase[:, None] * HARMONIC_K[None, :] + self._harm_off[v][None, :])
            voice = (harm @ a_v) * VOICE_GAINS[v]

            left += voice * self._pan_l[v]
            right += voice * self._pan_r[v]

        left *= _TIMBRE_NORM
        right *= _TIMBRE_NORM

        # Ramp amplitude smoothly within the block to avoid volume-jump clicks.
        amp = np.linspace(self._cur_vol, target, frames)
        self._cur_vol = target

        outdata[:, 0] = (left * amp * MAX_AMPLITUDE).astype(np.float32)
        outdata[:, 1] = (right * amp * MAX_AMPLITUDE).astype(np.float32)

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


def draw_pitch_ruler(frame, active_idx, finger_x=None):
    """Draw a labeled note ruler you can aim the index finger at.

    A horizontal line spans the pitch zone; each note gets a tick + label, the
    note nearest the current pitch is highlighted, and a red pointer marks the
    finger's x.
    """
    h, w, _ = frame.shape
    y = int(h * PITCH_RULER_Y)
    x_lo, x_hi = int(PITCH_X_LEFT * w), int(PITCH_X_RIGHT * w)
    cv2.line(frame, (x_lo, y), (x_hi, y), (235, 235, 235), 3)

    n = len(SCALE_NOTES)
    for i, name in enumerate(SCALE_NOTES):
        x = int(x_lo + (i / (n - 1)) * (x_hi - x_lo))
        active = (i == active_idx)
        sharp = "#" in name
        # White keys: long bold tick + always labeled. Black keys (semitones):
        # shorter tick, labeled only when active so the ruler stays readable.
        if active:
            color = (0, 255, 255)
        elif sharp:
            color = (150, 150, 150)
        else:
            color = (235, 235, 235)
        tick = 22 if active else (10 if sharp else 18)
        thick = 4 if active else (1 if sharp else 3)
        cv2.line(frame, (x, y - tick), (x, y + tick), color, thick)
        if active or not sharp:
            fs = 0.6 if active else 0.5
            (tw, _th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(frame, name, (x - tw // 2, y - tick - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
        if active:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), cv2.FILLED)

    if finger_x is not None:
        fx = int(float(np.clip(finger_x, PITCH_X_LEFT, PITCH_X_RIGHT)) * w)
        cv2.drawMarker(frame, (fx, y + 28), (0, 0, 255),
                       cv2.MARKER_TRIANGLE_UP, 24, 3)


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
        smoothed_x = 0.5             # smoothed index-finger x for the pitch glide
        prev_pos = None              # last frame's pitch (semitones) for velocity
        tone.set_freq(index_x_to_freq(smoothed_x))
        current_note = freq_to_note_name(tone.freq)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            pitch_hand, volume_hand = split_hands(results)

            # RIGHT hand index-finger x -> pitch, CONTINUOUS (left=low, right=high),
            # with velocity-dependent "pitch gravity": held notes are pulled in
            # tune (tremble -> vibrato) while slides stay smooth. See attracted_freq.
            if pitch_hand is not None:
                x = pitch_hand.landmark[INDEX_TIP].x
                smoothed_x += PITCH_SMOOTHING * (x - smoothed_x)
                freq_raw = index_x_to_freq(smoothed_x)
                pos = 12.0 * math.log2(freq_raw / PITCH_LOW_HZ)   # semitones
                vel = 0.0 if prev_pos is None else abs(pos - prev_pos)
                prev_pos = pos
                freq = attracted_freq(freq_raw, vel)
                tone.set_freq(freq)
                current_note = freq_to_note_name(freq)
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
                prev_pos = None       # reset velocity so re-entry isn't a false slide

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
            # Note ruler: labeled line to aim the index finger at. Highlight the
            # note nearest the current (continuous) pitch as a reference.
            finger_x = pitch_hand.landmark[INDEX_TIP].x if pitch_hand is not None else None
            nearest_idx = min(range(len(SCALE_FREQS)),
                              key=lambda i: abs(SCALE_FREQS[i] - tone.freq))
            draw_pitch_ruler(frame, nearest_idx, finger_x)
            pct = int(round(smoothed_vol * 100))
            bar_w = int(smoothed_vol * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
            pitch_state = "tracking" if pitch_hand is not None else "--"
            vol_state = "tracking" if volume_hand is not None else "--"
            cv2.putText(frame, f"Note: {current_note} ({tone.freq:.0f} Hz)   Volume: {pct}%",
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
