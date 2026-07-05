"""
Camera diagnostic. Run:  ./venv/bin/python camera_test.py

For each camera index it reports whether frames arrive, how bright they are
(std dev ~0 = black), and whether they CHANGE over time (motion ~0 = frozen /
static image). This isolates a camera/system problem from the app.
"""
import cv2
import numpy as np

for index in (0, 1, 2, 3):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"index {index}: did NOT open")
        cap.release()
        continue
    brightness = 0.0
    motion = 0.0
    prev = None
    frames = 0
    for _ in range(40):                 # warm up + sample
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames += 1
        brightness = float(frame.std())
        if prev is not None:
            motion = max(motion, float(np.mean(
                np.abs(frame.astype(np.int16) - prev.astype(np.int16)))))
        prev = frame
    cap.release()
    if frames == 0:
        print(f"index {index}: opened but read 0 frames")
        continue
    if brightness <= 5.0:
        verdict = "BLACK / no image"
    elif motion <= 0.4:
        verdict = "FROZEN / static image"
    else:
        verdict = "LIVE (moving)"
    print(f"index {index}: {frames} frames  brightness={brightness:6.2f}  "
          f"motion={motion:6.2f}  ->  {verdict}")

print("\nUse the index reported as LIVE. If your only 'camera' is FROZEN or "
      "BLACK, it's a Continuity Camera / stuck daemon issue -- disable "
      "Continuity Camera or run  sudo killall VDCAssistant  and retry.")
