"""
Gesture melody conductor (synth timbre).

A fixed melody (randomly generated) is "conducted" by your two hands:

  RIGHT hand:
    - fist -> open              -> STARTS playback and plays the first note
                                   (nothing sounds until you do this)
    - quick move to a new spot  -> triggers the NEXT note of the melody (rhythm)
    - openness (fist..spread)   -> volume (fist = silent)

  LEFT hand (expression on the current note):
    - raise / lower it          -> glissando up / down by up to a whole tone
    - tremble up & down fast     -> vibrato

The sound is the soft synthesized string ensemble (no samples).

Run:
    ./venv/bin/python gesture_melody.py
Keys:  SPACE = restart from the beginning (then fist->open to start again)
       r     = load a new random melody
       ESC   = quit

We mirror the camera (selfie view) so MediaPipe handedness matches your real
hands; flip SWAP_HANDS if they come out reversed.
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
MAX_AMPLITUDE = 0.42     # overall loudness ceiling at 100% volume (soft)
PITCH_MIN_HZ = 40.0
PITCH_MAX_HZ = 2000.0
SEMITONE = 2.0 ** (1.0 / 12.0)
PITCH_GLIDE = 0.40       # audio-rate portamento (used for the glissando bend)
EXP_VIB_HZ = 5.5         # rate of the expressive (left-hand) vibrato
ATTACK_SEC = 0.04        # note onset (re-articulation) time on each trigger

# ---- Timbre: soft string ensemble (same engine as gesture_orchestra) ----
N_HARMONICS = 16
HARMONIC_K = np.arange(1, N_HARMONICS + 1)
ANTIALIAS_HZ = 0.45 * SAMPLE_RATE
SECTIONS = [
    (0.5, 4.0, 3, 0.45),     # soft lower octave -> warmth
    (1.0, 5.5, 10, 0.70),    # warm strings      -> body
    (2.0, 5.0, 3, 0.22),     # soft octave up    -> sheen
]
_oct, _gain, _roll = [], [], []
for _octave, _rolloff, _n, _g in SECTIONS:
    _oct += [_octave] * _n
    _gain += [_g] * _n
    _roll += [_rolloff] * _n
VOICE_OCTAVES = np.array(_oct)
VOICE_GAINS = np.array(_gain)
N_VOICES = len(VOICE_OCTAVES)
HARM_A = np.array([(1.0 / HARMONIC_K) * np.exp(-(HARMONIC_K - 1) / r) for r in _roll])
DETUNE_SPREAD_CENTS = 6.0
VIB_RATE_RANGE = (4.5, 6.0)
VIB_DEPTH_RANGE = (0.003, 0.006)
_TIMBRE_GAIN = 4.5
_TIMBRE_NORM = _TIMBRE_GAIN / (HARM_A.sum(axis=1) * VOICE_GAINS).sum()

# ---- Scale + melody -----------------------------------------------------
_NOTE_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_WHITE = ["C", "D", "E", "F", "G", "A", "B"]


def note_to_freq(note):
    semis, i = _NOTE_SEMITONE[note[0].upper()], 1
    if note[i] in "#b":
        semis += 1 if note[i] == "#" else -1
        i += 1
    midi = (int(note[i:]) + 1) * 12 + semis
    return 440.0 * (SEMITONE ** (midi - 69))


def freq_to_note_name(freq):
    midi = round(69 + 12 * math.log2(freq / 440.0))
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def white_range(low, high):
    notes, (letter, octv) = [], (low[0], int(low[1:]))
    i = _WHITE.index(letter)
    while True:
        notes.append(f"{_WHITE[i]}{octv}")
        if notes[-1] == high:
            return notes
        i = (i + 1) % 7
        if i == 0:
            octv += 1


SCALE_NOTES = white_range("G3", "C5")
SCALE_FREQS = [note_to_freq(n) for n in SCALE_NOTES]


# ======================= EDIT YOUR MELODY HERE =======================
# Write your tune as space- (or comma-) separated note names. Any octave works,
# sharps as 'C#4', flats as 'Db4'. Example is the opening of "Ode to Joy".
# Leave MELODY = "" to get a random melody instead. Press 'r' while running to
# generate a fresh random one.
MELODY = "A3 B3 C4 A3 B3 C4 E4 D#4 E4 A3 B3 C4 A3 B3 C4 G4 F#4 B3"
# =====================================================================


def parse_melody(text):
    """Parse 'E4 F4 G#4' -> ['E4', 'F4', 'G#4'], skipping anything invalid."""
    out = []
    for nm in text.replace(",", " ").split():
        try:
            note_to_freq(nm)                 # validate it's a real note name
            out.append(nm)
        except Exception:
            print(f"[melody] skipping unrecognized note: {nm!r}")
    return out


def make_melody(n=24, seed=None):
    """A gentle random-walk melody over the scale; returns a list of note names."""
    rng = np.random.default_rng(seed)
    idx = len(SCALE_NOTES) // 2
    out = []
    for _ in range(n):
        out.append(SCALE_NOTES[idx])
        step = int(rng.choice([-2, -1, -1, 0, 1, 1, 2]))
        idx = int(np.clip(idx + step, 0, len(SCALE_NOTES) - 1))
    return out


# ---- Control config -----------------------------------------------------
SWAP_HANDS = False
CAMERA_INDEX = None              # None = auto-detect the first live camera
INDEX_TIP = 8

# Right hand: trigger the next note on a quick move (with re-arm hysteresis).
TRIGGER_SPEED = 0.045            # normalized 2D fingertip speed/frame to fire
REARM_SPEED = 0.020             # must drop below this before it can fire again

# Right hand: openness -> volume, and a fist->open starts the very first note.
OPEN_RATIO_FIST = 0.55
OPEN_RATIO_OPEN = 1.30
VOL_SMOOTHING = 0.25
FIST_LEVEL = 0.20               # openness (in volume units) below this = "fist"
OPEN_LEVEL = 0.60               # ...then rising above this = "opened" -> start

# Left hand: vertical position -> pitch bend (glissando), tremble -> vibrato.
BEND_RANGE = 0.15               # vertical fraction of frame for a FULL bend
MAX_BEND = 1.0                  # semitones up/down (a half step / 半度)
LEFT_SLOW = 0.15                # smoothing for the slow (bend) component
# Vibrato only kicks in once the fast up/down wobble is CLEARLY above a deadzone,
# so small hand jitter / tracking noise no longer produces vibrato.
VIB_THRESHOLD = 0.016           # residual motion below this -> no vibrato (deadzone)
VIB_GAIN = 3000.0               # above-threshold tremble -> vibrato depth (cents)
MAX_VIB_CENTS = 45.0
VIB_SMOOTH = 0.2

FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]


class TonePlayer:
    """Soft ensemble synth: a base note that can be bent (glissando), given an
    expressive vibrato, and re-articulated (attack) on each melody trigger."""

    def __init__(self, sample_rate=SAMPLE_RATE, freq=None):
        self.sample_rate = sample_rate
        freq = freq or SCALE_FREQS[len(SCALE_FREQS) // 2]
        self._freq = freq            # current glided frequency
        self._target_freq = freq     # target (base note * bend)
        self._target_vol = 0.0
        self._cur_vol = 0.0
        self._vib_depth = 0.0        # expressive vibrato depth (ratio)
        self._vib_phase = 0.0
        self._env = 1.0              # attack envelope (0 on trigger -> 1)
        self._pending_trigger = False
        self._lock = threading.Lock()

        rng = np.random.default_rng(7)
        self._detune = 2.0 ** (rng.normal(0.0, DETUNE_SPREAD_CENTS, N_VOICES) / 1200.0)
        self._vrate = rng.uniform(*VIB_RATE_RANGE, size=N_VOICES)
        self._vdepth = rng.uniform(*VIB_DEPTH_RANGE, size=N_VOICES)
        self._harm_off = rng.uniform(0.0, 2.0 * np.pi, size=(N_VOICES, N_HARMONICS))
        pos = np.linspace(0.0, 1.0, N_VOICES)
        rng.shuffle(pos)
        theta = pos * (np.pi / 2.0)
        self._pan_l, self._pan_r = np.cos(theta), np.sin(theta)
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)
        self._vphases = rng.uniform(0.0, 2.0 * np.pi, size=N_VOICES)

        self.stream = sd.OutputStream(samplerate=sample_rate, channels=2,
                                      callback=self._callback, blocksize=1024)

    def set_freq(self, freq):
        with self._lock:
            self._target_freq = float(np.clip(freq, PITCH_MIN_HZ, PITCH_MAX_HZ))

    def set_volume(self, vol):
        with self._lock:
            self._target_vol = float(np.clip(vol, 0.0, 1.0))

    def set_vibrato(self, cents):
        with self._lock:
            self._vib_depth = 2.0 ** (max(0.0, cents) / 1200.0) - 1.0

    def trigger(self):
        """Re-articulate: snap to the new base note and restart the attack."""
        with self._lock:
            self._pending_trigger = True

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            target_vol = self._target_vol
            target_freq = self._target_freq
            vib_depth = self._vib_depth
            if self._pending_trigger:
                self._freq = target_freq     # snap (new note, no slur)
                self._env = 0.0              # restart attack
                self._pending_trigger = False

        sr = self.sample_rate
        idx = np.arange(frames)
        two_pi = 2.0 * math.pi

        # Glissando glide (per-sample ramp) + expressive vibrato on the base freq.
        freq_end = self._freq + PITCH_GLIDE * (target_freq - self._freq)
        base = np.linspace(self._freq, freq_end, frames)
        self._freq = freq_end
        vib = 1.0 + vib_depth * np.sin(self._vib_phase + two_pi * EXP_VIB_HZ * idx / sr)
        self._vib_phase = float((self._vib_phase + two_pi * EXP_VIB_HZ * frames / sr) % two_pi)
        base_freq = base * vib

        left = np.zeros(frames)
        right = np.zeros(frames)
        for v in range(N_VOICES):
            vv = 1.0 + self._vdepth[v] * np.sin(
                self._vphases[v] + two_pi * self._vrate[v] * idx / sr)
            self._vphases[v] = float(
                (self._vphases[v] + two_pi * self._vrate[v] * frames / sr) % two_pi)
            inst = base_freq * VOICE_OCTAVES[v] * self._detune[v] * vv
            phase = self._phases[v] + np.cumsum(two_pi * inst / sr)
            self._phases[v] = float(phase[-1] % two_pi)
            fund = freq_end * VOICE_OCTAVES[v]
            a_v = HARM_A[v] * (HARMONIC_K * fund < ANTIALIAS_HZ)
            harm = np.sin(phase[:, None] * HARMONIC_K[None, :] + self._harm_off[v][None, :])
            voice = (harm @ a_v) * VOICE_GAINS[v]
            left += voice * self._pan_l[v]
            right += voice * self._pan_r[v]
        left *= _TIMBRE_NORM
        right *= _TIMBRE_NORM

        # Attack envelope (per note onset) and volume, both ramped per sample.
        env_step = frames / (ATTACK_SEC * sr)
        env_end = min(1.0, self._env + env_step)
        env = np.linspace(self._env, env_end, frames)
        self._env = env_end
        amp = np.linspace(self._cur_vol, target_vol, frames) * env
        self._cur_vol = target_vol

        outdata[:, 0] = (left * amp * MAX_AMPLITUDE).astype(np.float32)
        outdata[:, 1] = (right * amp * MAX_AMPLITUDE).astype(np.float32)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


# ---- Hand helpers -------------------------------------------------------
def split_hands(results):
    """Return (right_hand, left_hand) landmark objects, or None for each.
    Right = trigger + volume; Left = expression (bend + vibrato)."""
    right = left = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label
            if SWAP_HANDS:
                label = "Right" if label == "Left" else "Left"
            if label == "Right":
                right = lms
            else:
                left = lms
    return right, left


def hand_openness(landmarks):
    def pt(i):
        lm = landmarks[i]
        return np.array([lm.x, lm.y, lm.z])
    palm = np.linalg.norm(pt(0) - pt(9))
    if palm < 1e-6:
        return 0.0
    total = sum(np.linalg.norm(pt(t) - pt(m)) for t, m in FINGERS)
    return (total / len(FINGERS)) / palm


def ratio_to_volume(r, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((r - lo) / (hi - lo), 0.0, 1.0))


def open_camera():
    indices = (CAMERA_INDEX,) if CAMERA_INDEX is not None else (0, 1, 2)
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        nonblack, motion, prev = False, 0.0, None
        for _ in range(40):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if float(frame.std()) > 5.0:
                nonblack = True
            if prev is not None:
                motion = max(motion, float(np.mean(np.abs(
                    frame.astype(np.int16) - prev.astype(np.int16)))))
            prev = frame
        if nonblack and motion > 0.4:
            print(f"Using camera index {index} (live).")
            return cap
        print(f"Camera index {index}: black/frozen; skipping.")
        cap.release()
    raise RuntimeError("No live camera. Try CAMERA_INDEX=None, disable Continuity "
                       "Camera, or run: sudo killall VDCAssistant")


# ---- HUD ----------------------------------------------------------------
def draw_melody_strip(frame, melody, playhead):
    h, w, _ = frame.shape
    y = int(h * 0.16)
    n = len(melody)
    x0, x1 = int(0.05 * w), int(0.95 * w)
    cv2.line(frame, (x0, y), (x1, y), (120, 120, 120), 2)
    for k in range(n):
        x = int(x0 + (k / (n - 1)) * (x1 - x0))
        played = (k == playhead)
        color = (0, 255, 255) if played else (170, 170, 170)
        cv2.circle(frame, (x, y), 8 if played else 4, color, cv2.FILLED)
        if played or k == (playhead + 1) % n:
            name = melody[k]
            (tw, _t), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, name, (x - tw // 2, y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def main():
    lo, hi = OPEN_RATIO_FIST, OPEN_RATIO_OPEN
    cap = open_camera()
    melody = parse_melody(MELODY) or make_melody()
    print(f"[melody] {len(melody)} notes: {' '.join(melody)}")

    with TonePlayer() as tone, mp_hands.Hands(
        model_complexity=0, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands:
        smoothed_vol = 0.0
        playhead = -1            # -1 = not started; first flick plays melody[0]
        started = False
        base_name = None
        # Right-hand trigger state.
        prev_r = None
        armed = True
        fist_ready = False        # saw a fist; a following open will start playback
        reset_flash = 0
        # Left-hand expression state.
        left_slow_y = None
        left_neutral_y = None
        vib_level = 0.0
        bend = 0.0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            right_hand, left_hand = split_hands(results)
            h, w, _ = frame.shape

            # ---- RIGHT hand: start (fist->open), advance notes (flick), volume --
            target_vol = 0.0
            if right_hand is not None:
                # Track the WRIST for movement, not a fingertip: opening/closing
                # the hand barely moves the wrist, so spreading fingers (for
                # volume) no longer counts as a "move" and won't advance the note.
                anchor = right_hand.landmark[0]
                p = np.array([anchor.x, anchor.y])
                speed = 0.0 if prev_r is None else float(np.hypot(*(p - prev_r)))
                prev_r = p
                vol_now = ratio_to_volume(hand_openness(right_hand.landmark), lo, hi)
                if not started:
                    # The FIRST note starts on a fist -> open of the right hand.
                    if vol_now < FIST_LEVEL:
                        fist_ready = True
                    if fist_ready and vol_now > OPEN_LEVEL:
                        started, playhead, base_name = True, 0, melody[0]
                        tone.set_freq(note_to_freq(base_name) * (SEMITONE ** bend))
                        tone.trigger()
                        armed, fist_ready = False, False
                else:
                    # Once started, a quick move triggers the next note.
                    if armed and speed > TRIGGER_SPEED:
                        playhead = (playhead + 1) % len(melody)
                        base_name = melody[playhead]
                        tone.set_freq(note_to_freq(base_name) * (SEMITONE ** bend))
                        tone.trigger()
                        armed = False
                    elif speed < REARM_SPEED:
                        armed = True
                if started:
                    target_vol = vol_now
                mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                # Marker on the wrist -- the point whose movement triggers notes.
                cv2.circle(frame, (int(anchor.x * w), int(anchor.y * h)), 10, (0, 255, 255), 2)
            else:
                prev_r = None
                armed = True
                fist_ready = False
            smoothed_vol += VOL_SMOOTHING * (target_vol - smoothed_vol)
            tone.set_volume(smoothed_vol)

            # ---- LEFT hand: bend (position) + vibrato (fast tremble) ----
            if left_hand is not None:
                ly = left_hand.landmark[INDEX_TIP].y
                if left_slow_y is None:
                    left_slow_y = ly
                    left_neutral_y = ly           # where the hand entered = neutral
                left_slow_y += LEFT_SLOW * (ly - left_slow_y)
                residual = ly - left_slow_y       # fast wiggle (tremble)
                vib_level += VIB_SMOOTH * (abs(residual) - vib_level)
                # Deadzone: only clear trembling (above VIB_THRESHOLD) makes vibrato.
                vib_cents = min(MAX_VIB_CENTS, VIB_GAIN * max(0.0, vib_level - VIB_THRESHOLD))
                # bend from slow vertical displacement vs. where the hand entered
                bend = float(np.clip((left_neutral_y - left_slow_y) / BEND_RANGE,
                                     -1.0, 1.0)) * MAX_BEND
                tone.set_vibrato(vib_cents)
                mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
            else:
                left_slow_y = left_neutral_y = None
                vib_level = 0.0
                bend = 0.0
                tone.set_vibrato(0.0)

            # Apply current base note * bend every frame (continuous glissando).
            if base_name is not None:
                tone.set_freq(note_to_freq(base_name) * (SEMITONE ** bend))

            # ---- HUD ----
            draw_melody_strip(frame, melody, playhead)
            pct = int(round(smoothed_vol * 100))
            bar_w = int(smoothed_vol * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 20), (0, 255, 0), cv2.FILLED)
            nxt = melody[(playhead + 1) % len(melody)]
            if started:
                status = (f"Note {playhead + 1}/{len(melody)}: {base_name}   next: {nxt}"
                          f"   bend: {bend:+.2f}st   vol: {pct}%")
            else:
                status = f"Ready -- make a FIST then OPEN your right hand.  First note: {nxt}"
            cv2.putText(frame, status, (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "R: flick=next note, openness=volume   L: up/down=glissando, "
                        "tremble=vibrato   SPACE=restart  r=new melody  ESC=quit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
            if reset_flash > 0:
                reset_flash -= 1
                (tw, _t), _ = cv2.getTextSize("RESTART", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.putText(frame, "RESTART", ((w - tw) // 2, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3, cv2.LINE_AA)

            cv2.imshow("Gesture Melody - ESC quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):                       # SPACE = restart current melody
                playhead, started, base_name, fist_ready = -1, False, None, False
                reset_flash = 18
            elif key == ord("r"):                       # r = brand-new random melody
                melody = make_melody()
                playhead, started, base_name, fist_ready = -1, False, None, False
                print(f"[melody] {len(melody)} notes: {' '.join(melody)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
