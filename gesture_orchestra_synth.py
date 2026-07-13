"""
Two-handed gesture orchestra (theremin-style).

A continuous, rich orchestral tutti plays through your speakers, controlled by
both hands at once:
    RIGHT index finger x -> pitch   (move left = lower, move right = higher)
    LEFT hand openness   -> volume  (fist = silent, spread = full)

Run:
    ./venv/bin/python gesture_orchestra.py
Press ESC in the video window to quit. Press 'c' to (re)calibrate the left
hand's fist/open range to your own hand. Press 'b' to turn the black keys
(semitones/sharps) on or off -- off gives a white-keys-only scale.

Which hand is which
-------------------
We mirror the camera (selfie view). HandRouter keeps right hand = pitch and
left hand = volume by TRACKING each hand across frames (a hand barely moves
between frames), so the left hand can never take over the pitch -- not when the
hands cross, not when one drops out, and not when MediaPipe's flickery
handedness label misfires for a frame. Labels are only used to classify a
brand-new lone hand (with a several-frame vote before pitch is granted) and to
heal a wrong seed after many consecutive disagreeing frames. If your setup
comes out reversed, flip SWAP_HANDS below.

How the controls are measured
-----------------------------
Pitch: the horizontal position (x) of the RIGHT index fingertip over the scale
G3..C5 (far left = G3, far right = C5). The fingertip's HEIGHT gates it. On the
ruler line, a still finger holds the exact ("pure") note while a slow horizontal
pan glides through the pitches in between (glissando). Drop below the line to
travel silently in an arc to another note, then rise back to the line to sound
that destination note cleanly. Only the right index finger affects the pitch.
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
# Built once -- get_default_*_style() allocates a fresh spec dict on every call.
LANDMARK_STYLE = mp_styles.get_default_hand_landmarks_style()
CONNECTION_STYLE = mp_styles.get_default_hand_connections_style()

# ---- Audio config -------------------------------------------------------
SAMPLE_RATE = 44100
TONE_HZ = 220.0          # starting pitch of the continuous tone (A3)
MAX_AMPLITUDE = 0.42     # overall loudness ceiling at 100% volume (soft)
PITCH_MIN_HZ = 55.0      # keyboard pitch control clamps to this range
PITCH_MAX_HZ = 2000.0
SEMITONE = 2.0 ** (1.0 / 12.0)   # one-semitone frequency multiplier

# ---- Timbre: warm, full string orchestra --------------------------------
# A rich but SOFT string orchestra ("浑厚丰富但柔和") -- no brass, no piercing
# highs. The body comes from stacking many string sections across octaves, from
# a deep contrabass foundation up through cellos, violas and violins, so it
# stays full and powerful without any harsh edge. Each section is a pool of
# independent players (own detuning, vibrato rate/phase and harmonic phases);
# with many voices per section they melt into a broad, breathing string tutti.
# The LOW rolloff values keep every voice close to a mellow sine with only a
# gentle overtone bloom -- warm, never bright or edgy. Voices are panned across
# stereo for orchestral width; it all renders live so pitch/volume stay instant.
N_HARMONICS = 16
HARMONIC_K = np.arange(1, N_HARMONICS + 1)
ANTIALIAS_HZ = 0.45 * SAMPLE_RATE           # drop harmonics above this (no aliasing)

# Each section: (octave multiplier, brightness rolloff, n_voices, gain/voice).
# Bigger rolloff = brighter; lower octave = deeper. These rolloffs are kept LOW
# (soft, string-like) so nothing gets harsh; the fullness comes from the octave
# spread (0.25x .. 2x) and the large voice count, not from bright harmonics.
SECTIONS = [
    (0.25, 2.0, 1, 0.30),    # contrabasses         -> deep, round foundation
    (0.5,  2.6, 3, 0.52),    # cellos & basses       -> warm lower body
    (1.0,  3.2, 7, 0.74),    # violas & violins (core) -> full, soft mid weight
    (2.0,  2.8, 4, 0.40),    # upper strings         -> gentle warmth, not piercing
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

DETUNE_SPREAD_CENTS = 10.0                  # std-dev of random per-voice detune (big ensemble)
VIB_RATE_RANGE = (4.5, 6.0)                 # orchestral string vibrato speed (Hz)
VIB_DEPTH_RANGE = (0.004, 0.007)            # gentle vibrato depth

# Make-up gain after normalization; tuned so the worst-case peak across all
# notes stays clear of clipping (verified by a sweep).
_TIMBRE_GAIN = 5.8
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
# lower it for the white keys. Range G3..C5.
SWAP_HANDS = False               # set True if your left/right come out swapped
INDEX_TIP = 8                    # MediaPipe landmark: index fingertip
_WHITE = ["C", "D", "E", "F", "G", "A", "B"]

PITCH_X_LEFT = 0.10              # note zone: far left  = G3 (kept off the frame
PITCH_X_RIGHT = 0.90            #            far right = C5  edges, where hand
                                #            tracking is least accurate)
RULER_Y_BLACK = 0.57             # upper line (black keys / sharps), lower part of frame
RULER_Y_WHITE = 0.72             # lower line (white keys / naturals), toward the bottom
ROW_SPLIT_Y = 0.645              # finger y above this -> black row, below -> white
# Height-gated pitch, driven ONLY by the RIGHT index fingertip.
#   - Below the ruler line (DISENGAGED): the pitch FREEZES, so you can arc down
#     and swing across to another note silently -- nothing in between sounds.
#   - On/above the line (ENGAGED): whether you hear a glissando depends on the
#     fingertip's HORIZONTAL motion:
#       * arriving from below, or holding still -> snap to and HOLD the exact
#         note: a clean, in-tune "pure" tone (this also hides fingertip jitter);
#       * panning slowly along the line -> glide continuously through the
#         pitches in between (a glissando).
# So: return to the line -> a pure note; slow pan on the line -> a glissando.
# Hysteresis (ENGAGE_Y..DISENGAGE_Y) keeps engage/disengage from flickering.
# glide = 0 when the horizontal speed |dx| <= MOVE_LO (still), rising to 1 at
# MOVE_HI (panning); it blends the snapped note and the continuous pitch.
# MOVE_SMOOTH smooths the speed reading; OUT_SMOOTH is a click-free glide on the
# result.
ENGAGE_Y = RULER_Y_WHITE + 0.03      # rise to here (or above) -> engaged (on the line)
DISENGAGE_Y = RULER_Y_WHITE + 0.08   # drop below here -> disengaged (silent travel)
MOVE_LO = 0.0025                     # |dx|/frame at/below this = still -> pure note
MOVE_HI = 0.0060                     # at/above this = panning -> glissando
MOVE_SMOOTH = 0.5                    # EMA on the horizontal-speed reading
OUT_SMOOTH = 0.5
POS_SMOOTH = 0.6                 # fingertip-position EMA (1 = none). Tames landmark
                                 # jitter for steadier, more precise notes; kept high
                                 # so it adds well under a frame of lag.


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
    """White row G3..C5 (evenly spaced) + black row (sharps at white midpoints)."""
    names = white_range("G3", "C5")
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


def pick_key(fx, fy, allow_black=True):
    """Nearest key to finger x, in the row chosen by finger y (up = black).

    With allow_black False the black (sharp) row is disabled: white keys only,
    regardless of finger height."""
    row = BLACK_KEYS if (allow_black and fy < ROW_SPLIT_Y) else WHITE_KEYS
    best = min(row, key=lambda k: abs(k["x"] - fx))
    return best, (row is BLACK_KEYS)


def continuous_freq(fx):
    """Un-snapped pitch at finger x, log-interpolated between the two neighbouring
    white keys -- the continuous target a glissando sweep follows."""
    n = len(WHITE_KEYS)
    span = PITCH_X_RIGHT - PITCH_X_LEFT
    pos = (float(np.clip(fx, PITCH_X_LEFT, PITCH_X_RIGHT)) - PITCH_X_LEFT) / span * (n - 1)
    i = min(int(pos), n - 2)
    t = pos - i
    lo, hi = WHITE_KEYS[i]["freq"], WHITE_KEYS[i + 1]["freq"]
    return math.exp((1.0 - t) * math.log(lo) + t * math.log(hi))


# ---- Openness -> volume calibration ------------------------------------
# Raw ratio values that map to 0% and 100%. Defaults work for most hands;
# press 'c' in the app to recalibrate to yours.
OPEN_RATIO_FIST = 0.55   # ratio when making a fist  -> 0%
OPEN_RATIO_OPEN = 1.30   # ratio when fully spread   -> 100%

SMOOTHING = 0.25         # 0..1, lower = smoother/slower volume changes

# Finger (tip_idx, mcp_idx) pairs for the 4 long fingers.
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]


TABLE_SIZE = 4096        # wavetable resolution; interp error at 16 harmonics is ~-80 dB


class TonePlayer:
    """Rich orchestral-tutti synth playing one continuous note. Set its frequency
    and volume from the video thread; the audio callback renders the full
    ensemble live and keeps every voice's phase continuous so pitch changes stay
    click-free.

    Rendering is wavetable-based for speed: each voice's waveform (its harmonic
    stack) is a fixed periodic function, so it is precomputed ONCE into a table
    and the callback just looks it up at the running phase (linear interp),
    vectorized across all voices. That replaces ~250k np.sin calls per block
    with one table gather -- far less CPU and far less GIL time stolen from the
    video loop. Anti-aliasing is preserved exactly: tables are cumulative per
    harmonic count, and each voice picks the table with only its harmonics that
    fall below ANTIALIAS_HZ at the current pitch."""

    def __init__(self, sample_rate=SAMPLE_RATE, freq=TONE_HZ):
        self.sample_rate = sample_rate
        self._target_vol = 0.0
        self._cur_vol = 0.0
        self._freq = float(freq)
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
        self._pan_l = np.cos(theta).astype(np.float32)
        self._pan_r = np.sin(theta).astype(np.float32)
        self._gains32 = VOICE_GAINS.astype(np.float32)

        # Cumulative wavetables: tables[v, n] = voice v's waveform built from its
        # first n harmonics, so the antialias cutoff can drop high harmonics by
        # just picking a lower n -- same result as masking them, no recompute.
        x = np.arange(TABLE_SIZE) * (2.0 * np.pi / TABLE_SIZE)
        tables = np.zeros((N_VOICES, N_HARMONICS + 1, TABLE_SIZE), dtype=np.float32)
        for v in range(N_VOICES):
            acc = np.zeros(TABLE_SIZE)
            for k in range(N_HARMONICS):
                acc = acc + HARM_A[v, k] * np.sin((k + 1) * x + self._harm_off[v, k])
                tables[v, k + 1] = acc
        self._tables = tables
        self._n_ok = None             # harmonics-under-cutoff per voice (cached)
        self._vt = None               # the (V, TABLE_SIZE) tables selected by _n_ok

        # Per-voice phase state, so the note stays click-free across callbacks.
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)
        self._vib_phases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)

        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=2,
            callback=self._callback, blocksize=1024)

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
            target_vol = self._target_vol
            freq = self._freq

        sr = self.sample_rate
        two_pi = 2.0 * math.pi
        t = np.arange(frames)

        # Vibrato and phase accumulation for ALL voices at once (V, frames).
        vib = 1.0 + self._vib_depth[:, None] * np.sin(
            self._vib_phases[:, None] + (two_pi / sr) * self._vib_rate[:, None] * t)
        self._vib_phases = (self._vib_phases + two_pi * self._vib_rate * frames / sr) % two_pi
        inst = (freq * VOICE_OCTAVES * self._detune)[:, None] * vib
        phase = self._phases[:, None] + np.cumsum((two_pi / sr) * inst, axis=1)
        self._phases = phase[:, -1] % two_pi

        # Pick each voice's table for the harmonics under the antialias cutoff
        # (the cutoff mask is a prefix of k, so a count selects the same set).
        n_ok = (HARMONIC_K[None, :] * (freq * VOICE_OCTAVES)[:, None]
                < ANTIALIAS_HZ).sum(axis=1)
        if self._n_ok is None or not np.array_equal(n_ok, self._n_ok):
            self._n_ok = n_ok
            self._vt = self._tables[np.arange(N_VOICES), n_ok]

        # Wavetable lookup with linear interpolation (phase is always >= 0).
        pos = phase * (TABLE_SIZE / two_pi)
        ip = pos.astype(np.int64)
        frac = (pos - ip).astype(np.float32)
        i0 = ip % TABLE_SIZE
        i1 = (ip + 1) % TABLE_SIZE
        w0 = np.take_along_axis(self._vt, i0, axis=1)
        w1 = np.take_along_axis(self._vt, i1, axis=1)
        sig = (w0 + frac * (w1 - w0)) * self._gains32[:, None]

        left = self._pan_l @ sig
        right = self._pan_r @ sig

        # Ramp master volume per sample (no click when the volume jumps).
        amp = np.linspace(self._cur_vol, target_vol, frames,
                          dtype=np.float32) * np.float32(_TIMBRE_NORM * MAX_AMPLITUDE)
        self._cur_vol = target_vol

        outdata[:, 0] = np.clip(left * amp, -1.0, 1.0)
        outdata[:, 1] = np.clip(right * amp, -1.0, 1.0)

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


def _index_tip(lms):
    t = lms.landmark[INDEX_TIP]
    return np.array([t.x, t.y])


TRACK_MATCH_DIST = 0.25   # max wrist jump (normalized) to count as the same hand
TRACK_MAX_AGE = 10        # frames a lost hand's track stays valid for re-matching
SOLO_PITCH_VOTES = 3      # consistent right-hand labels a lone NEW hand needs for pitch
SWAP_HEAL_FRAMES = 10     # frames of steady label disagreement before roles swap


def _wrist(lms):
    w = lms.landmark[0]
    return np.array([w.x, w.y])


class HandRouter:
    """Assigns hands to the pitch (physical RIGHT) and volume (physical LEFT)
    roles so the LEFT hand can NEVER move the pitch.

    Identity comes from CONTINUITY: a hand barely moves between frames, so each
    role tracks its wrist position and every new frame is matched to the nearest
    track. MediaPipe's handedness label is never trusted per frame (it flickers,
    and a single mislabeled frame must not hand the pitch to the left hand); it
    is only used (a) to decide what a brand-new lone hand is -- and then a lone
    hand must vote 'right' several frames in a row before it is GRANTED pitch,
    while one 'left' vote immediately makes it volume (erring on the safe side)
    -- and (b) as a slow healer: if, with both hands visible, the labels
    consistently contradict the current assignment for many consecutive frames,
    the roles swap once. The label->hand mapping itself is learned from the
    unambiguous two-hand case (right-most on the mirrored screen = physical
    right). SWAP_HANDS inverts the roles."""

    def __init__(self):
        self._label_bias = 0.0    # >0: MediaPipe's 'Right' label = physical right hand
        self._tracks = {"pitch": None, "vol": None}    # role -> [wrist_pos, age]
        self._solo_vote = 0.0     # lone-new-hand handedness evidence
        self._swap_evid = 0       # steady-disagreement counter (role healing)

    def _fresh(self, role):
        t = self._tracks[role]
        return t is not None and t[1] <= TRACK_MAX_AGE

    def _is_right_label(self, label):
        right = "Right" if self._label_bias >= 0.0 else "Left"
        if SWAP_HANDS:
            right = "Left" if right == "Right" else "Right"
        return label == right

    def route(self, results):
        """Return (pitch_hand, volume_hand); either may be None. Pitch is only ever
        the tracked physical right hand -- never the left one."""
        for t in self._tracks.values():
            if t is not None:
                t[1] += 1
        lms = results.multi_hand_landmarks
        handed = results.multi_handedness
        if not lms or not handed:
            self._solo_vote = 0.0
            return None, None
        hands = [(l, h.classification[0].label) for l, h in zip(lms, handed)]

        if len(hands) >= 2:
            (h0, lab0), (h1, lab1) = hands[0], hands[1]
            p0, p1 = _wrist(h0), _wrist(h1)
            # Learn the label mapping only when it is unambiguous: hands clearly
            # apart and labelled differently (right-most = physical right).
            if abs(p0[0] - p1[0]) > 0.15 and lab0 != lab1:
                right_lab = lab0 if p0[0] > p1[0] else lab1
                self._label_bias += 0.15 * ((1.0 if right_lab == "Right" else -1.0)
                                            - self._label_bias)
            if self._fresh("pitch") and self._fresh("vol"):
                # Both roles tracked -> pure continuity (handles crossing; immune
                # to label flicker).
                pp, vp = self._tracks["pitch"][0], self._tracks["vol"][0]
                keep = (np.linalg.norm(p0 - pp) + np.linalg.norm(p1 - vp)
                        <= np.linalg.norm(p1 - pp) + np.linalg.norm(p0 - vp))
            elif lab0 != lab1:
                # (Re)seeding with informative labels -> the right-labelled hand.
                keep = self._is_right_label(lab0)
            else:
                # Same label on both -> seed by screen side (right-most = pitch).
                keep = p0[0] >= p1[0]
                if SWAP_HANDS:
                    keep = not keep
            pitch_hand, pitch_lab = (h0, lab0) if keep else (h1, lab1)
            volume_hand, vol_lab = (h1, lab1) if keep else (h0, lab0)

            # Healing: if the labels steadily contradict this assignment, a seed
            # went wrong (e.g. a hand re-entered on the far side) -- swap ONCE
            # after SWAP_HEAL_FRAMES consecutive disagreeing frames. Transient
            # label flicker never gets that far.
            if pitch_lab != vol_lab:
                if self._is_right_label(pitch_lab):
                    self._swap_evid = 0
                else:
                    self._swap_evid += 1
                    if self._swap_evid >= SWAP_HEAL_FRAMES:
                        pitch_hand, volume_hand = volume_hand, pitch_hand
                        self._swap_evid = 0
            self._tracks["pitch"] = [_wrist(pitch_hand), 0]
            self._tracks["vol"] = [_wrist(volume_hand), 0]
            self._solo_vote = 0.0
            return pitch_hand, volume_hand

        # --- one hand visible ---
        h, lab = hands[0]
        p = _wrist(h)
        d_pitch = (np.linalg.norm(p - self._tracks["pitch"][0])
                   if self._fresh("pitch") else np.inf)
        d_vol = (np.linalg.norm(p - self._tracks["vol"][0])
                 if self._fresh("vol") else np.inf)
        if min(d_pitch, d_vol) <= TRACK_MATCH_DIST:
            # Continues a tracked hand -> keep its role, whatever the label says.
            role = "pitch" if d_pitch <= d_vol else "vol"
            self._tracks[role] = [p, 0]
            return (h, None) if role == "pitch" else (None, h)
        # A brand-new lone hand: vote on its handedness. Pitch needs several
        # consistent right-hand labels; a single left-hand label makes it volume
        # (the safe direction -- the left hand must never take the pitch).
        self._solo_vote += 1.0 if self._is_right_label(lab) else -1.0
        if self._solo_vote >= SOLO_PITCH_VOTES:
            self._tracks["pitch"] = [p, 0]
            self._solo_vote = 0.0
            return h, None
        if self._solo_vote <= -1.0:
            self._tracks["vol"] = [p, 0]
            self._solo_vote = 0.0
            return None, h
        return None, None            # undecided for a frame or two -> hold as-is


# Camera index. None = auto-detect the first camera that delivers live
# frames (skips a black/frozen Continuity Camera). Set an int to force one.
CAMERA_INDEX = None

# Capture and processing sizes. We grab 720p (lighter to read/draw than 1080p)
# and run hand detection on a downscaled copy -- landmarks are normalized, so
# the display stays full-res while detection gets faster and lower-latency.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
PROC_WIDTH = 640


MIN_BRIGHTNESS = 50.0    # reject feeds dimmer than this mean (a covered/face-down
                         # Continuity Camera reads ~30 with a max pixel near 45)


def open_camera():
    """Open a webcam that delivers a LIVE, BRIGHT (non-black, non-frozen) image.

    On macOS index 0 is often a Continuity Camera (iPhone) that hands back a
    black, frozen, or very dark image (e.g. lying face-down) -- the usual cause
    of a black or "strange static" window. It can still show faint sensor noise,
    so a first-acceptable scan wrongly grabs it. Instead we warm up EVERY index,
    keep only feeds that are actually changing over time, and pick the BRIGHTEST
    of those -- the real webcam pointed at a lit room wins over the dark iPhone.
    """
    indices = (CAMERA_INDEX,) if CAMERA_INDEX is not None else (0, 1, 2)
    best = None                                   # (brightness, index, cap)
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        motion = 0.0
        brightness = 0.0
        prev = None
        for _ in range(40):                       # ~1.3s warm-up
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            brightness = float(frame.mean())      # last settled frame's brightness
            if prev is not None:
                motion = max(motion, float(np.mean(
                    np.abs(frame.astype(np.int16) - prev.astype(np.int16)))))
            prev = frame
        forced = CAMERA_INDEX is not None
        live = motion > 0.4                        # live sensor noise/movement
        if live and (brightness >= MIN_BRIGHTNESS or forced):
            print(f"Camera index {index}: live, brightness {brightness:.0f}.")
            if best is None or brightness > best[0]:
                if best is not None:
                    best[2].release()
                best = (brightness, index, cap)
                continue
        else:
            if not live:
                reason = "black/frozen image (no motion)"
            else:
                reason = f"too dark (brightness {brightness:.0f} < {MIN_BRIGHTNESS:.0f}), likely a covered Continuity Camera"
            print(f"Camera index {index}: {reason}; skipping.")
        cap.release()
    if best is not None:
        print(f"Using camera index {best[1]} (brightest live feed).")
        return best[2]
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


_TEXT_W = {}             # (label, font_scale) -> pixel width, cached across frames


def _text_width(label, fs):
    key = (label, fs)
    if key not in _TEXT_W:
        _TEXT_W[key] = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
    return _TEXT_W[key]


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
        tw = _text_width(key["name"], fs)
        cv2.putText(frame, key["name"], (x - tw // 2, y - tick - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
        if active:
            cv2.circle(frame, (x, y), 7, (0, 255, 255), cv2.FILLED)
    if is_active_row and finger_x is not None:
        fx = int(float(np.clip(finger_x, PITCH_X_LEFT, PITCH_X_RIGHT)) * w)
        cv2.drawMarker(frame, (fx, y + 24), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 24, 3)


def draw_pitch_ruler(frame, active_key, is_black, finger_x=None, show_black=True):
    """Two rows: upper line = black keys (sharps), lower = white keys. The active
    row (chosen by the finger's height) is bright; the current note highlighted.
    With show_black False only the white row is drawn (semitones disabled)."""
    if show_black:
        _draw_ruler_row(frame, BLACK_KEYS, RULER_Y_BLACK,
                        active_key if is_black else None, is_black,
                        finger_x if is_black else None)
    _draw_ruler_row(frame, WHITE_KEYS, RULER_Y_WHITE,
                    None if is_black else active_key, not is_black,
                    None if is_black else finger_x)


WINDOW = "Gesture Orchestra - ESC quit, c calibrate"


def detect_hands(hands, frame_bgr):
    """Run MediaPipe on a downscaled RGB copy of the frame. Landmarks are
    normalized (size-independent), so the full-res frame is still what we draw.
    Downscale FIRST so the colour conversion only touches 1/4 of the pixels."""
    if frame_bgr.shape[1] > PROC_WIDTH:
        h = round(frame_bgr.shape[0] * PROC_WIDTH / frame_bgr.shape[1])
        frame_bgr = cv2.resize(frame_bgr, (PROC_WIDTH, h))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return hands.process(rgb)


class VolumeControl:
    """Maps the LEFT hand's openness (fist..spread) to a smoothed volume 0..1."""

    def __init__(self, lo=OPEN_RATIO_FIST, hi=OPEN_RATIO_OPEN):
        self.lo, self.hi = lo, hi
        self.value = 0.0

    def update(self, volume_hand):
        target = 0.0
        if volume_hand is not None:
            target = ratio_to_volume(hand_openness(volume_hand.landmark), self.lo, self.hi)
        self.value += SMOOTHING * (target - self.value)
        return self.value


class PitchControl:
    """Maps the RIGHT index fingertip to a pitch. Vertical position gates it (with
    hysteresis): below the ruler line the pitch FREEZES (silent travel -- arc
    across notes without sounding them); on/above the line a still finger holds
    the exact "pure" note while a slow horizontal pan glides through the pitches
    in between (glissando), and returning to the line always re-sounds the exact
    note. Only the fingertip's x/y are read, so nothing else -- not the other
    fingers, not the left hand -- can move the pitch."""

    def __init__(self):
        mid = WHITE_KEYS[len(WHITE_KEYS) // 2]
        self.key = mid                # note the ruler points at (held while travelling)
        self.is_black = False
        self.freq = mid["freq"]       # actually-sounding pitch
        self.engaged = False
        self.finger_x = None          # smoothed fingertip x/y (None when no pitch hand)
        self.finger_y = None
        self._tip = None              # EMA-smoothed (x, y)
        self._prev_engaged = False
        self._prev_x = None
        self._dx = 0.0                # smoothed horizontal speed (still vs. panning)

    def update(self, pitch_hand, semitones_on):
        """Advance one frame; refreshes freq / key / engaged / finger_x / finger_y."""
        if pitch_hand is None:
            self._tip = self._prev_x = None
            self._dx = 0.0
            self.engaged = self._prev_engaged = False
            self.finger_x = self.finger_y = None
            return                    # hold the last freq

        # Smooth the fingertip to tame per-frame landmark jitter.
        raw = _index_tip(pitch_hand)
        self._tip = raw if self._tip is None else self._tip + POS_SMOOTH * (raw - self._tip)
        sx, sy = float(self._tip[0]), float(self._tip[1])
        self.finger_x, self.finger_y = sx, sy

        # Engage / disengage by finger height, with hysteresis.
        if self.engaged and sy > DISENGAGE_Y:
            self.engaged = False
        elif not self.engaged and sy < ENGAGE_Y:
            self.engaged = True

        if self.engaged:
            self.key, self.is_black = pick_key(sx, sy, allow_black=semitones_on)
            snap = self.key["freq"]
            if not self._prev_engaged or self._prev_x is None:
                # Returned to the line -> sound the exact note (the arc here was silent).
                self.freq = snap
                self._dx = 0.0
            else:
                # Still -> hold the pure note; panning -> glissando between notes.
                self._dx += MOVE_SMOOTH * (abs(sx - self._prev_x) - self._dx)
                glide = float(np.clip((self._dx - MOVE_LO) / (MOVE_HI - MOVE_LO), 0.0, 1.0))
                cont = snap if self.is_black else continuous_freq(sx)
                target = math.exp((1.0 - glide) * math.log(snap) + glide * math.log(cont))
                self.freq = math.exp(math.log(self.freq)
                                     + OUT_SMOOTH * (math.log(target) - math.log(self.freq)))
        # else: disengaged -> hold self.freq (silent travel between notes).
        self._prev_x = sx
        self._prev_engaged = self.engaged


def draw_hud(frame, pitch, volume, semitones_on):
    """Draw the fingertip marker, note ruler, volume bar and text overlays."""
    h, w, _ = frame.shape
    # Fingertip marker: cyan on the line, orange while travelling below it.
    if pitch.finger_x is not None:
        dot = (0, 255, 255) if pitch.engaged else (0, 140, 255)
        cv2.circle(frame, (int(pitch.finger_x * w), int(pitch.finger_y * h)), 10, dot, 2)
    # Note ruler (white row; black row too when semitones are on).
    draw_pitch_ruler(frame, pitch.key, pitch.is_black, pitch.finger_x, show_black=semitones_on)
    # Volume bar.
    pct = int(round(volume * 100))
    bar_w = int(volume * (w - 40))
    cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
    cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
    # Text.
    semi = "on" if semitones_on else "off"
    state = "" if pitch.engaged else "  (travel)"
    cv2.putText(frame, f"Note: {pitch.key['name']}{state}   Semitones: {semi}   Volume: {pct}%",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    row_hint = "up=black/down=white  " if semitones_on else ""
    cv2.putText(frame, f"R index: on line=note / arc below=jump  {row_hint}L hand=volume   "
                "b semitones   ESC quit  c calibrate",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    cap = open_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # keep only the latest frame -> less lag

    router = HandRouter()
    pitch = PitchControl()
    volume = VolumeControl()
    semitones_on = False                         # 'b' toggles the black-key (sharp) row

    with TonePlayer(freq=pitch.freq) as tone, mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)           # selfie mirror
            results = detect_hands(hands, frame)

            # Route hands, then drive pitch (right index) and volume (left openness).
            pitch_hand, volume_hand = router.route(results)
            pitch.update(pitch_hand, semitones_on)
            tone.set_freq(pitch.freq)
            tone.set_volume(volume.update(volume_hand))

            for hand in (pitch_hand, volume_hand):
                if hand is not None:
                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        LANDMARK_STYLE, CONNECTION_STYLE)
            draw_hud(frame, pitch, volume.value, semitones_on)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                        # ESC -> quit
                break
            elif key == ord("b"):                # toggle semitones (black keys)
                semitones_on = not semitones_on
            elif key == ord("c"):                # recalibrate the volume-hand range
                volume.lo, volume.hi = calibrate(cap, hands)

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
            res = detect_hands(hands, frame)
            # Calibration shows a single (left) hand -> just read whichever hand
            # is detected, no role assignment needed.
            volume_hand = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None
            ratio = hand_openness(volume_hand.landmark) if volume_hand is not None else None
            cv2.putText(frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if ratio is not None:
                cv2.putText(frame, f"ratio={ratio:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
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
