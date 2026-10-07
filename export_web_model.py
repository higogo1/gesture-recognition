"""훈련한 분류기(joblib)를 웹에서 쓸 수 있도록 JSON으로 내보내기

실행: python export_web_model.py
출력: web/model.json  (StandardScaler 값 + MLP 가중치)
"""
import json
import os

import joblib

from gesture_features import BASE_DIR, CLASSIFIER_PATH

OUT_PATH = os.path.join(BASE_DIR, "web", "model.json")


def main():
    model = joblib.load(CLASSIFIER_PATH)
    scaler, mlp = model[0], model[-1]
    data = {
        "classes": [str(c) for c in model.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,          # 은닉층: relu
        "outActivation": mlp.out_activation_,  # 출력층: softmax(3개 이상) / logistic(2개)
        "weights": [w.tolist() for w in mlp.coefs_],
        "biases": [b.tolist() for b in mlp.intercepts_],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"내보내기 완료: {OUT_PATH}  클래스={data['classes']}")


if __name__ == "__main__":
    main()
