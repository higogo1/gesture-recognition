// 파이썬 gesture_features.py + scikit-learn MLP 추론을 JS로 옮긴 것

// 월드 랜드마크 21개 → 63차원 특징 (파이썬 landmarks_to_features와 동일)
export function landmarksToFeatures(world, handedness) {
  const pts = world.map((p) => [p.x - world[0].x, p.y - world[0].y, p.z - world[0].z]);
  if (handedness === "Left") pts.forEach((p) => (p[0] *= -1));
  const scale = Math.max(...pts.map((p) => Math.hypot(p[0], p[1], p[2])));
  return pts.flat().map((v) => (scale > 0 ? v / scale : v));
}

// model.json(export_web_model.py 출력)을 받아 확률을 계산하는 분류기
export class GestureClassifier {
  constructor(model) {
    this.m = model;
    this.classes = model.classes;
  }

  predictProba(features) {
    const { mean, scale, weights, biases, outActivation } = this.m;
    let x = features.map((v, i) => (v - mean[i]) / scale[i]);
    weights.forEach((W, layer) => {
      const out = biases[layer].slice();
      for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      const last = layer === weights.length - 1;
      x = last ? out : out.map((v) => Math.max(0, v)); // 은닉층 relu
    });
    if (outActivation === "logistic") {
      const p = 1 / (1 + Math.exp(-x[0]));
      return [1 - p, p];
    }
    const max = Math.max(...x);
    const exps = x.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }
}
