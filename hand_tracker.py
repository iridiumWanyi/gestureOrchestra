"""
Real-time hand / fingertip tracking with MediaPipe + OpenCV.

Reads from your webcam, detects hands, draws the skeleton, and prints/draws
the coordinates of each fingertip. Press 'q' to quit.

MediaPipe returns 21 landmarks per hand. The 5 fingertips are landmarks:
    THUMB_TIP   = 4
    INDEX_TIP   = 8
    MIDDLE_TIP  = 12
    RING_TIP    = 16
    PINKY_TIP   = 20

Each landmark has:
    x, y : normalized to [0, 1] relative to the image width/height
    z    : depth relative to the wrist (smaller = closer to camera)
We convert x, y to pixel coordinates for display/use.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark index -> human-readable fingertip name
FINGERTIPS = {
    4: "THUMB",
    8: "INDEX",
    12: "MIDDLE",
    16: "RING",
    20: "PINKY",
}


def open_camera():
    """Try a few camera indices using the macOS AVFoundation backend."""
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            # Confirm we can actually read a frame, not just "open".
            ok, _ = cap.read()
            if ok:
                print(f"Using camera index {index}.")
                return cap
        cap.release()
    raise RuntimeError(
        "Could not open any webcam (tried indices 0-2).\n"
        "On macOS this is usually a camera-permission problem:\n"
        "  System Settings -> Privacy & Security -> Camera -> enable your\n"
        "  terminal app (Terminal / iTerm / VS Code), then fully quit and\n"
        "  reopen it. Also close any app already using the camera."
    )


def main():
    cap = open_camera()

    with mp_hands.Hands(
        model_complexity=0,            # 0 fastest, 1 more accurate
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Ignoring empty camera frame.")
                continue

            # Mirror so it feels natural; convert BGR->RGB for MediaPipe.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False        # small perf win
            results = hands.process(rgb)

            h, w, _ = frame.shape
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Left / Right label.
                    label = "Hand"
                    if results.multi_handedness:
                        label = results.multi_handedness[hand_idx].classification[0].label

                    # Draw the full skeleton.
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Extract + report each fingertip.
                    for lm_id, name in FINGERTIPS.items():
                        lm = hand_landmarks.landmark[lm_id]
                        px, py = int(lm.x * w), int(lm.y * h)

                        cv2.circle(frame, (px, py), 8, (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, name, (px + 10, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        print(
                            f"{label:5s} {name:6s} "
                            f"norm=({lm.x:.3f}, {lm.y:.3f})  "
                            f"px=({px:4d}, {py:4d})  z={lm.z:+.3f}"
                        )
                    print("-" * 60)

            cv2.imshow("Hand Tracker - press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
