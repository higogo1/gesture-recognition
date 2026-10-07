import {
  FilesetResolver,
  GestureRecognizer,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import { GestureClassifier, landmarksToFeatures } from "./classifier.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task";
const STABLE_FRAMES = 5; // 같은 결과가 연속 N프레임이어야 오버레이 표시

// 제스처 라벨 → 화면에 띄울 오버레이 (라벨은 소문자로 비교)
// 로고 이미지 파일이 있으면 { html: '<img src="assets/nike.png" width="280">' } 처럼 바꿔도 됩니다.
const OVERLAYS = {
  nike: {
    html: `<svg viewBox="0 0 300 110" aria-label="Nike swoosh">
      <path fill="#fff" d="M38 70 C 10 88, 4 108, 34 106 C 52 104, 80 94, 296 6
        C 120 70, 78 84, 60 84 C 40 84, 36 74, 38 70 Z"/>
    </svg>`,
  },
  ok: { emoji: "👌" },
  "멈춰": { emoji: "😡" },
};

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const stage = $("stage");

let recognizer, classifier, drawer;
let lastVideoTime = -1;
const history = [];
const overlayEls = {};
let stableLabel = null;

// ---------- 초기화 ----------
async function loadClassifier() {
  const res = await fetch("model.json", { cache: "no-store" });
  if (!res.ok) throw new Error("model.json이 없습니다. python export_web_model.py를 먼저 실행하세요.");
  classifier = new GestureClassifier(await res.json());
  $("bars").innerHTML = classifier.classes
    .map((c) => `<div class="bar-row"><span>${c}</span><div class="bar"><i id="bar-${c}"></i></div><span id="pct-${c}">0%</span></div>`)
    .join("");
}

function buildOverlays() {
  for (const [label, cfg] of Object.entries(OVERLAYS)) {
    const el = document.createElement("div");
    el.className = "overlay" + (cfg.emoji ? " emoji" : "");
    el.innerHTML = cfg.emoji ?? cfg.html;
    stage.appendChild(el);
    overlayEls[label] = el;
  }
}

async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  btn.textContent = "모델 불러오는 중…";
  try {
    await loadClassifier();
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    recognizer = await GestureRecognizer.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    await video.play();
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`; // 잘림 없이 → 오버레이 위치 정확
    drawer = new DrawingUtils(ctx);
    buildOverlays();
    $("start").remove();
    $("hint").innerHTML = "nike → Nike 로고 · ok → 👌 · 멈춰 → 😡<br />제스처가 5프레임 연속 같아야 표시됩니다.";
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    btn.disabled = false;
    btn.textContent = "다시 시도";
    const msg = e.name === "NotReadableError" ? "카메라를 다른 프로그램이 사용 중입니다. (gesture_studio.py 등을 닫고 다시 시도)" : e.message;
    $("hint").innerHTML = `<span class="error">${msg}</span>`;
  }
}

// ---------- 매 프레임 ----------
function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = recognizer.recognizeForVideo(video, performance.now());
    render(result);
  }
  requestAnimationFrame(loop);
}

function render(result) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const handed = result.handedness ?? result.handednesses;
  if (!result.worldLandmarks?.length) {
    update(null, null);
    return;
  }
  const lm = result.landmarks[0];
  drawer.drawConnectors(lm, GestureRecognizer.HAND_CONNECTIONS, { color: "#3ddc84", lineWidth: 3 });
  drawer.drawLandmarks(lm, { color: "#ff4d4d", radius: 3 });

  const feats = landmarksToFeatures(result.worldLandmarks[0], handed[0][0].categoryName);
  update(classifier.predictProba(feats), lm);
}

function update(proba, lm) {
  const thr = parseFloat($("thr").value);
  let label = null;
  if (proba) {
    classifier.classes.forEach((c, i) => {
      $(`bar-${c}`).style.width = `${proba[i] * 100}%`;
      $(`pct-${c}`).textContent = `${Math.round(proba[i] * 100)}%`;
    });
    const best = proba.indexOf(Math.max(...proba));
    label = proba[best] >= thr ? classifier.classes[best] : "Unknown";
    $("result").textContent = label;
    $("score").textContent = `확신도 ${proba[best].toFixed(2)}`;
  } else {
    classifier.classes.forEach((c) => {
      $(`bar-${c}`).style.width = "0%";
      $(`pct-${c}`).textContent = "0%";
    });
    $("result").textContent = "-";
    $("score").textContent = "손이 보이지 않음";
  }

  // 최근 N프레임이 모두 같은 라벨일 때만 바꿈 → 깜빡임 방지
  history.push(label);
  if (history.length > STABLE_FRAMES) history.shift();
  if (history.length === STABLE_FRAMES && history.every((h) => h === label)) {
    stableLabel = label?.toLowerCase() ?? null;
  }
  showOverlay(stableLabel, lm);
}

function showOverlay(key, lm) {
  for (const [label, el] of Object.entries(overlayEls)) {
    const on = label === key;
    el.classList.toggle("show", on);
    if (on && lm) {
      // 손 위쪽에 띄우기 (영상이 거울 모드라 x를 반전)
      const xs = lm.map((p) => p.x), ys = lm.map((p) => p.y);
      const cx = 1 - (Math.min(...xs) + Math.max(...xs)) / 2;
      const top = Math.min(...ys);
      el.style.left = `${cx * 100}%`;
      el.style.top = `${Math.max(top - 0.18, 0.15) * 100}%`;
    }
  }
}

$("thr").addEventListener("input", (e) => ($("thrText").textContent = Number(e.target.value).toFixed(2)));
$("startBtn").addEventListener("click", start);
