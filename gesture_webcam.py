"""MediaPipe Gesture Recognizer - 웹캠 실시간 제스처 인식

실행: python gesture_webcam.py [--camera 0] [--hands 2]
종료: q 또는 ESC
"""
import argparse
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")

# 손 랜드마크 21개 연결선
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),
]


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--hands", type=int, default=2, help="최대 손 개수")
    args = parser.parse_args()

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {args.camera}번을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            for i, landmarks in enumerate(result.hand_landmarks):
                pts = draw_hand(frame, landmarks)
                label = ""
                if i < len(result.gestures) and result.gestures[i]:
                    g = result.gestures[i][0]
                    label = f"{g.category_name} ({g.score:.2f})"
                if i < len(result.handedness) and result.handedness[i]:
                    # 거울 모드라 좌우가 뒤집혀 보이므로 반대로 표시
                    side = result.handedness[i][0].category_name
                    side = {"Left": "Right", "Right": "Left"}.get(side, side)
                    label = f"{side}: {label}"
                x = min(p[0] for p in pts)
                y = max(min(p[1] for p in pts) - 10, 20)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2, cv2.LINE_AA)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Recognizer (q/ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
