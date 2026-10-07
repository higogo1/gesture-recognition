"""수집한 데이터로 제스처 분류기 훈련

실행: python train_gestures.py
입력: data/gestures.csv
출력: models/custom_gesture.joblib
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_features import CLASSIFIER_PATH, DATA_PATH


def train(data_path=DATA_PATH, out_path=CLASSIFIER_PATH, test_size=0.2, log=print):
    """CSV를 읽어 분류기를 훈련 · 평가 · 저장하고 모델을 반환. 진행 상황은 log()로 출력"""
    if not os.path.exists(data_path):
        raise ValueError(f"데이터 파일이 없습니다: {data_path}")
    raw = np.genfromtxt(data_path, delimiter=",", dtype=str, skip_header=1, encoding="utf-8")
    if raw.size == 0:
        raise ValueError("수집된 데이터가 없습니다.")
    raw = np.atleast_2d(raw)
    y = raw[:, 0]
    X = raw[:, 1:].astype(np.float32)

    labels, counts = np.unique(y, return_counts=True)
    log("클래스별 샘플 수:")
    for name, n in zip(labels, counts):
        log(f"  {name:<15} {n}")
    if len(labels) < 2:
        raise ValueError("제스처가 2개 이상 필요합니다. (none 클래스 포함 권장)")
    if counts.min() < 10:
        raise ValueError("샘플이 10개 미만인 클래스가 있습니다. 클래스당 200개 이상 권장.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                      early_stopping=True, random_state=42),
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    log("\n=== 테스트 결과 ===")
    log(classification_report(y_test, pred, digits=3))
    log(f"혼동 행렬 (행=정답, 열=예측): {[str(c) for c in model.classes_]}")
    log(str(confusion_matrix(y_test, pred, labels=model.classes_)))

    # 최종 모델은 전체 데이터로 다시 학습
    model.fit(X, y)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    log(f"\n저장 완료: {out_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--out", default=CLASSIFIER_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    try:
        train(args.data, args.out, args.test_size)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
