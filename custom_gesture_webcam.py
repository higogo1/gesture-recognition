"""훈련한 나만의 제스처로 웹캠 실시간 추론

실행: python custom_gesture_webcam.py [--hands 2] [--min-score 0.7]
종료: q 또는 ESC
"""
import argparse
import time

import cv2
import joblib
import mediapipe as mp

from gesture_features import CLASSIFIER_PATH, create_recognizer, landmarks_to_features
from gesture_webcam import draw_hand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--model", default=CLASSIFIER_PATH)
    parser.add_argument("--min-score", type=float, default=0.7, help="이보다 낮으면 Unknown")
    args = parser.parse_args()

    clf = joblib.load(args.model)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {args.camera}번을 열 수 없습니다.")

    start = time.monotonic()
    with create_recognizer(args.hands) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_image, int((time.monotonic() - start) * 1000))

            for i, world in enumerate(result.hand_world_landmarks):
                pts = draw_hand(frame, result.hand_landmarks[i])
                feats = landmarks_to_features(world, result.handedness[i][0].category_name)
                proba = clf.predict_proba([feats])[0]
                best = proba.argmax()
                name = clf.classes_[best] if proba[best] >= args.min_score else "Unknown"
                label = f"{name} ({proba[best]:.2f})"

                x = min(p[0] for p in pts)
                y = max(min(p[1] for p in pts) - 10, 20)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Custom Gesture (q/ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
