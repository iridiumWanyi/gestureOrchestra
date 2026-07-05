"""
Gesture-controlled volume.

A continuous tone plays through your speakers. The volume follows how open
your hand is:
    closed fist     -> 0%   (silent)
    fully spread     -> 100% (loudest)
    in between       -> scaled by how open the hand is

Run:
    ./venv/bin/python volume_control.py
Press 'q' in the video window to quit. Press 'c' to (re)calibrate the
fist/open range to your own hand (see on-screen hint).

How "openness" is measured
--------------------------
MediaPipe gives 21 landmarks per hand. We use a scale-invariant ratio:
for the 4 long fingers we take the distance from each fingertip to its
base knuckle (MCP), sum them, and divide by the palm size (wrist -> middle
knuckle). A curled fist makes that ratio small; a spread hand makes it big.
This ratio is independent of how far your hand is from the camera.
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
MAX_AMPLITUDE = 0.6      # overall loudness ceiling at 100% volume (0..1)
PITCH_MIN_HZ = 55.0      # keyboard pitch control clamps to this range
PITCH_MAX_HZ = 2000.0
SEMITONE = 2.0 ** (1.0 / 12.0)   # one-semitone frequency multiplier

# ---- Timbre: full orchestral tutti --------------------------------------
# To sound like a whole orchestra ("气势恢弘"), we layer several instrument
# SECTIONS at once, each spanning a different octave and brightness:
# contrabass/tuba weight at the bottom, strings + brass in the middle, violins
# and piccolo sheen on top. Each section is a pool of independent players --
# every voice has its OWN random detuning, vibrato rate/phase and harmonic
# phases, so they never fuse into a mechanical buzz but bloom into a big,
# breathing tutti. Voices are panned across stereo for orchestral width.
# It all renders live, so pitch (keyboard) and volume (hand) stay instant.
N_HARMONICS = 16
HARMONIC_K = np.arange(1, N_HARMONICS + 1)
ANTIALIAS_HZ = 0.45 * SAMPLE_RATE           # drop harmonics above this (no aliasing)

# Each section: (octave multiplier, brightness rolloff, n_voices, gain/voice).
# Bigger rolloff = brighter; lower octave = deeper.
SECTIONS = [
    (0.25,  4.0, 3, 0.95),   # contrabass / tuba   -> foundation & weight
    (0.5,   5.0, 4, 0.85),   # cello / low brass   -> body
    (1.0,   7.0, 6, 0.70),   # violas + strings    -> warm core
    (1.0,  11.0, 4, 0.60),   # horns / brass       -> epic blaze
    (2.0,   9.0, 4, 0.45),   # violins / woodwinds -> brilliance
    (4.0,  10.0, 2, 0.18),   # piccolo             -> air / sheen
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

DETUNE_SPREAD_CENTS = 8.0                   # std-dev of random per-voice detune
VIB_RATE_RANGE = (4.5, 6.8)                 # each player's vibrato speed (Hz)
VIB_DEPTH_RANGE = (0.003, 0.007)            # each player's vibrato depth

# Make-up gain after normalization; tuned so the worst-case peak across all
# notes stays clear of clipping (verified by a sweep).
_TIMBRE_GAIN = 6.5
_TIMBRE_NORM = _TIMBRE_GAIN / (HARM_A.sum(axis=1) * VOICE_GAINS).sum()

# ---- Musical keyboard layout --------------------------------------------
# Two rows act like a piano. The QWERTY... row is the white keys (naturals);
# the number row is the black keys (sharps), sitting between them just like
# on a real keyboard. Keys 4, 7 and '-' are intentionally unmapped -- a piano
# has no black key between B-C or E-F.
#
#   sharps:  1   2   3       5   6       8   9   0
#   nats:    Q   W   E   R   T   Y   U   I   O   P   L
#            G3  A3  B3  C4  D4  E4  F4  G4  A4  B4  C5
# (C5 is on L because the bracket key right of P doesn't register reliably.)
KEY_TO_NOTE = {
    # white keys (naturals)
    "q": "G3", "w": "A3", "e": "B3", "r": "C4", "t": "D4", "y": "E4",
    "u": "F4", "i": "G4", "o": "A4", "p": "B4", "l": "C5",
    # black keys (sharps)
    "1": "F#3", "2": "G#3", "3": "A#3", "5": "C#4", "6": "D#4",
    "8": "F#4", "9": "G#4", "0": "A#4",
}

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


KEY_TO_FREQ = {k: note_to_freq(v) for k, v in KEY_TO_NOTE.items()}

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
        self._freq = freq        # read/written under _lock (audio thread reads it)
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
        with self._lock:
            self._freq = float(np.clip(freq, PITCH_MIN_HZ, PITCH_MAX_HZ))

    @property
    def freq(self):
        with self._lock:
            return self._freq

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            target = self._target_vol
            freq = self._freq

        sr = self.sample_rate
        idx = np.arange(frames)
        two_pi = 2.0 * math.pi

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
            inst_freq = freq * VOICE_OCTAVES[v] * self._detune[v] * vib
            phase = self._phases[v] + np.cumsum(two_pi * inst_freq / sr)
            self._phases[v] = float(phase[-1] % two_pi)

            # Harmonic stack with randomized harmonic phases (less "buzzy comb"),
            # dropping any overtone above Nyquist so the deep/high octaves don't
            # alias into harsh digital junk.
            fund = freq * VOICE_OCTAVES[v]
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


def open_camera():
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"Using camera index {index}.")
                return cap
        cap.release()
    raise RuntimeError(
        "Could not open any webcam (tried indices 0-2). Check\n"
        "System Settings -> Privacy & Security -> Camera, then fully quit\n"
        "and reopen your terminal app."
    )


def main():
    lo, hi = OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    cap = open_camera()

    with TonePlayer() as tone, mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        smoothed_vol = 0.0
        last_ratio = None
        current_note = "A3"          # matches the 220 Hz starting pitch

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            target_vol = 0.0   # no hand -> silent
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                ratio = hand_openness(hand_landmarks.landmark)
                last_ratio = ratio
                target_vol = ratio_to_volume(ratio, lo, hi)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # Smooth so volume glides instead of jittering.
            smoothed_vol += SMOOTHING * (target_vol - smoothed_vol)
            tone.set_volume(smoothed_vol)

            # ---- HUD ----
            h, w, _ = frame.shape
            pct = int(round(smoothed_vol * 100))
            bar_w = int(smoothed_vol * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"Volume: {pct}%   Note: {current_note} ({tone.freq:.0f} Hz)", (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "ESC quit   c calibrate   Q-row = notes   1-row = sharps", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Volume - ESC quit, c calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                    # ESC -> quit
                break
            elif key == ord("c") and last_ratio is not None:
                lo, hi = calibrate(cap, hands)
            elif 0 <= key < 128 and chr(key) in KEY_TO_FREQ:  # piano keyboard
                ch = chr(key)
                tone.set_freq(KEY_TO_FREQ[ch])
                current_note = KEY_TO_NOTE[ch]

    cap.release()
    cv2.destroyAllWindows()


def calibrate(cap, hands):
    """Capture a fist sample then an open-hand sample; return (lo, hi)."""
    samples = {}
    for stage, prompt in (("FIST", "Make a FIST, hold still, press SPACE"),
                          ("OPEN", "SPREAD your hand, hold still, press SPACE")):
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            ratio = None
            if res.multi_hand_landmarks:
                ratio = hand_openness(res.multi_hand_landmarks[0].landmark)
            cv2.putText(frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if ratio is not None:
                cv2.putText(frame, f"ratio={ratio:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Volume - q quit, c calibrate", frame)
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
