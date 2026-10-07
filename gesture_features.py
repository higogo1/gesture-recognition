"""수집 · 훈련 · 추론에서 공통으로 쓰는 랜드마크 → 특징 벡터 변환"""
import os

import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_recognizer.task")
DATA_PATH = os.path.join(BASE_DIR, "data", "gestures.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "custom_gesture.joblib")

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def create_recognizer(num_hands=1):
    """손 랜드마크 추출용으로 기본 Gesture Recognizer를 VIDEO 모드로 생성"""
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.GestureRecognizer.create_from_options(options)


def landmarks_to_features(world_landmarks, handedness):
    """월드 랜드마크(미터 단위, 화면 비율 영향 없음) 21개를 63차원 벡터로 변환.

    - 손목(0번)을 원점으로 이동 → 손 위치에 무관
    - 손목에서 가장 먼 점까지 거리로 나눔 → 손 크기 · 카메라 거리에 무관
    - 왼손은 x축을 뒤집음 → 오른손/왼손을 한 모델로 학습
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()
