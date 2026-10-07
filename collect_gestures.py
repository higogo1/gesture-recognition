"""나만의 제스처 데이터 수집 (웹캠)

실행: python collect_gestures.py --labels none fist_heart rock ok
조작:
  1~9      : 수집할 라벨 선택 (--labels 순서대로)
  SPACE    : 녹화 시작/정지 (녹화 중에는 손이 보이는 매 프레임 저장)
  q / ESC  : 종료
저장: data/gestures.csv (실행할 때마다 이어서 추가)
"""
import argparse
import csv
import os
import time
from collections import Counter

import cv2
import mediapipe as mp

from gesture_features import DATA_PATH, NUM_FEATURES, create_recognizer, landmarks_to_features
from gesture_webcam import draw_hand


def load_counts():
    if not os.path.exists(DATA_PATH):
        return Counter()
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return Counter(row[0] for row in csv.reader(f) if row and row[0] != "label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True, help="수집할 제스처 이름들 (최대 9개)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1, help="저장 간격(초). 비슷한 프레임 중복 방지")
    args = parser.parse_args()
    labels = args.labels[:9]

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)
    counts = load_counts()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {args.camera}번을 열 수 없습니다.")

    current = 0
    recording = False
    last_saved = 0.0
    start = time.monotonic()

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f, create_recognizer() as recognizer:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_image, int((time.monotonic() - start) * 1000))

            hand_found = bool(result.hand_world_landmarks)
            if hand_found:
                draw_hand(frame, result.hand_landmarks[0])
                now = time.monotonic()
                if recording and now - last_saved >= args.interval:
                    feats = landmarks_to_features(
                        result.hand_world_landmarks[0], result.handedness[0][0].category_name)
                    writer.writerow([labels[current]] + [f"{v:.6f}" for v in feats])
                    counts[labels[current]] += 1
                    last_saved = now

            # 상태 표시
            color = (0, 0, 255) if recording else (200, 200, 200)
            status = "REC" if recording else "PAUSE"
            cv2.putText(frame, f"[{status}] {current + 1}:{labels[current]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            if recording and not hand_found:
                cv2.putText(frame, "no hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2, cv2.LINE_AA)
            for i, name in enumerate(labels):
                cv2.putText(frame, f"{i + 1}:{name} = {counts[name]}", (10, 100 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Collect gestures (1-9 label, SPACE rec, q quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
            elif ord("1") <= key < ord("1") + len(labels):
                current = key - ord("1")
                recording = False  # 라벨을 바꾸면 실수 방지를 위해 일시정지

    cap.release()
    cv2.destroyAllWindows()
    print("수집 현황:", dict(counts))
    print("저장 위치:", DATA_PATH)


if __name__ == "__main__":
    main()
