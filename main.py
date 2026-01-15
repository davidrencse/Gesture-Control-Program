import time
import cv2
import mediapipe as mp
import pyautogui
import os
import sys

from keypoint_preprocess import pre_process_landmark
from gesture_classifier import KeyPointClassifier

# ====== Settings ======
WIDTH, HEIGHT = 1280, 720

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)

MODEL_PATH = resource_path(r"model\keypoint_classifier\keypoint_classifier.tflite")


# default labels: 0=open, 1=close(fist), 2=pointing (ignore)
OPEN_LABEL = 0
FIST_LABEL = 1

STABLE_FRAMES = 6          # increase if flickery (8–12)
SCROLL_INTERVAL = 0.15     # seconds between scroll ticks
SCROLL_STEP = 160          # scroll amount per tick (adjust)


def main():
    pyautogui.FAILSAFE = True  # slam mouse to corner to stop

    # Load classifier
    classifier = KeyPointClassifier(MODEL_PATH)

    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera failed to open. Try index 1/2 or remove CAP_DSHOW.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    draw = mp.solutions.drawing_utils

    stable_label = None
    run_len = 0
    last_scroll = 0.0

    while True:
        success, img = cap.read()
        if not success:
            print("Camera read failed")
            break

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        status = "NO_HAND"
        conf_note = ""

        if res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]
            draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Extract normalized (x,y) landmarks
            landmark_xy = [(lm.x, lm.y) for lm in hand_lms.landmark]

            # Preprocess -> (42,)
            feat = pre_process_landmark(landmark_xy)

            # Predict class
            label = classifier(feat)

            if label == OPEN_LABEL:
                status = "OPEN"
            elif label == FIST_LABEL:
                status = "FIST"
            else:
                status = "OTHER"  # ignore

            # Debounce (stability)
            if label == stable_label:
                run_len += 1
            else:
                stable_label = label
                run_len = 1

            # Scroll only when stable and only for OPEN/FIST
            now = time.time()
            if run_len >= STABLE_FRAMES and (now - last_scroll) >= SCROLL_INTERVAL:
                if stable_label == OPEN_LABEL:
                    pyautogui.scroll(+SCROLL_STEP)
                    last_scroll = now
                elif stable_label == FIST_LABEL:
                    pyautogui.scroll(-SCROLL_STEP)
                    last_scroll = now

        cv2.putText(img, f"{status} stable={run_len}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, f"{status} stable={run_len}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(img, "Press Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, "Press Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gesture Controller", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
