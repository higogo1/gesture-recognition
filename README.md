# 🖐️ MediaPipe 제스처 인식 — 웹캠 실습 강의노트

> **작성일:** 2026-10-07
> **주제:** Google MediaPipe Gesture Recognizer로 웹캠에서 손 제스처를 실시간으로 인식하기
> **환경:** Windows 11 · Python 3.14 · mediapipe 1.1.0 · opencv-python 5.0.0

---

## 📌 한눈에 보기 (TL;DR)

```
1. 모델 파일(.task) 다운로드   →  gesture_recognizer.task
2. 라이브러리 설치             →  pip install mediapipe opencv-python
3. 실행                       →  python gesture_webcam.py
4. 종료                       →  q 또는 ESC
```

| 구성 요소 | 역할 |
|---|---|
| `gesture_recognizer.task` | 학습이 끝난 AI 모델 묶음 (손 찾기 + 제스처 분류) |
| `mediapipe` | 모델을 불러와 추론을 실행하는 라이브러리 |
| `opencv-python` | 웹캠 영상 읽기 · 화면에 그리기 · 창 띄우기 |
| `gesture_webcam.py` | 위 셋을 연결한 실행 코드 |

---

## 📚 목차

1. [학습 목표](#1-학습-목표)
2. [핵심 개념](#2-핵심-개념)
3. [실습 1 — 환경 준비](#3-실습-1--환경-준비)
4. [실습 2 — 코드 이해하기](#4-실습-2--코드-이해하기)
5. [실습 3 — 실행과 결과](#5-실습-3--실행과-결과)
6. [트러블슈팅](#6-트러블슈팅)
7. [용어 정리](#7-용어-정리-glossary)
8. [더 해보기](#8-더-해보기)
9. [참고 자료](#9-참고-자료)

---

## 1. 학습 목표

이번 실습을 마치면 다음을 할 수 있습니다.

- [x] MediaPipe **Task** 구조(모델 파일 + Task API)를 이해한다
- [x] 사전 학습된 `.task` 모델을 내려받아 Python에서 불러온다
- [x] OpenCV로 웹캠 프레임을 읽고, MediaPipe가 원하는 이미지 형식으로 바꾼다
- [x] `VIDEO` 모드로 프레임마다 제스처를 인식하고 결과를 화면에 그린다

---

## 2. 핵심 개념

### 2-1. 제스처 인식은 2단계 파이프라인

```
 웹캠 프레임 (BGR)
      │
      ▼  색상 변환 BGR → RGB
 ┌─────────────────────────────┐
 │  ① Hand Landmark 모델        │  손을 찾고 관절 21개 좌표 추출
 └─────────────────────────────┘
      │  21개 랜드마크 (x, y, z)
      ▼
 ┌─────────────────────────────┐
 │  ② Gesture Classifier 모델   │  좌표 모양을 보고 제스처 분류
 └─────────────────────────────┘
      │
      ▼
 결과: 제스처 이름 + 점수, 왼손/오른손, 랜드마크
```

> 💡 **포인트:** 제스처 분류기는 *이미지*가 아니라 *관절 좌표*를 보고 판단합니다.
> 그래서 배경·조명이 달라져도 비교적 안정적으로 동작합니다.

### 2-2. 손 랜드마크 21개

```
          8   12  16  20        ← 손가락 끝 (TIP)
          |   |   |   |
          7   11  15  19
          |   |   |   |
     4    6   10  14  18
     |    |   |   |   |
     3    5---9---13--17        ← 손가락 뿌리 (MCP)
      \   |            /
       2  |           /
        \ |          /
         1|         /
           \       /
            \     /
               0                ← 손목 (WRIST)
```

| 번호 | 부위 | 번호 | 부위 |
|---|---|---|---|
| 0 | 손목 | 9–12 | 중지 |
| 1–4 | 엄지 | 13–16 | 약지 |
| 5–8 | 검지 | 17–20 | 새끼 |

### 2-3. 기본 제공 제스처 7종 (+ None)

| 라벨 | 의미 | 모양 |
|---|---|---|
| `Closed_Fist` | 주먹 | ✊ |
| `Open_Palm` | 편 손바닥 | ✋ |
| `Pointing_Up` | 검지 위로 | ☝️ |
| `Thumb_Down` | 엄지 아래 | 👎 |
| `Thumb_Up` | 엄지 위 | 👍 |
| `Victory` | 브이 | ✌️ |
| `ILoveYou` | 사랑해 | 🤟 |
| `None` | 해당 없음 | — |

### 2-4. Running Mode 3가지

| 모드 | 입력 | 호출 함수 | 언제 쓰나 |
|---|---|---|---|
| `IMAGE` | 사진 1장 | `recognize()` | 정지 이미지 |
| `VIDEO` | 연속 프레임 + 타임스탬프 | `recognize_for_video()` | 동영상·웹캠 (**이번 실습**, 동기식이라 코드가 단순) |
| `LIVE_STREAM` | 연속 프레임 + 타임스탬프 | `recognize_async()` + 콜백 | 지연을 최소화하고 싶은 실시간 앱 |

> ⚠️ `VIDEO`/`LIVE_STREAM` 모드에서 **타임스탬프(ms)는 반드시 계속 증가**해야 합니다.

---

## 3. 실습 1 — 환경 준비

### Step 1. 모델 파일 다운로드

공식 문서 → *Models* 섹션 → `HandGestureClassifier` 링크.

```powershell
# 브라우저 대신 명령어로 받을 수도 있음
curl -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

> 📁 파일 크기 약 8.4MB. **`gesture_webcam.py`와 같은 폴더**에 두세요.
> (이 저장소에는 모델 파일을 포함하지 않았습니다 — 위 명령으로 받으세요.)

### Step 2. 라이브러리 설치

```powershell
python -m pip install mediapipe opencv-python
```

> 💡 Windows에서 `pip` 명령이 안 잡히면 `python -m pip` 형태로 실행하세요.

### Step 3. 설치 확인

```powershell
python -c "import mediapipe as mp, cv2; print(mp.__version__, cv2.__version__)"
# 출력 예: 1.1.0 5.0.0
```

---

## 4. 실습 2 — 코드 이해하기

전체 코드: [`gesture_webcam.py`](gesture_webcam.py)

### ① 옵션 설정 → 인식기 생성

```python
options = vision.GestureRecognizerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),  # 모델 경로
    running_mode=vision.RunningMode.VIDEO,                            # 실행 모드
    num_hands=2,                                                      # 최대 손 개수
    min_hand_detection_confidence=0.5,   # 손 '검출' 최소 신뢰도
    min_hand_presence_confidence=0.5,    # 손 '존재' 최소 신뢰도
    min_tracking_confidence=0.5,         # 프레임 간 '추적' 최소 신뢰도
)
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    ...
```

### ② 웹캠 프레임 → MediaPipe 이미지

```python
ok, frame = cap.read()                       # OpenCV는 BGR 순서
frame = cv2.flip(frame, 1)                   # 거울 모드
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe는 RGB 필요
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
```

> ⚠️ **흔한 실수:** BGR→RGB 변환을 빼먹으면 인식률이 크게 떨어집니다.

### ③ 인식 실행

```python
timestamp_ms = int((time.monotonic() - start) * 1000)  # 단조 증가 시계
result = recognizer.recognize_for_video(mp_image, timestamp_ms)
```

### ④ 결과 구조

```
result
├── gestures[i][0].category_name   → "Thumb_Up"
├── gestures[i][0].score           → 0.87
├── handedness[i][0].category_name → "Left" / "Right"
└── hand_landmarks[i][0..20]       → .x .y .z (0~1 정규화 좌표)
```

> 💡 랜드마크 좌표는 0~1로 **정규화**되어 있으므로 픽셀 좌표로 쓰려면
> `x * 이미지너비`, `y * 이미지높이`를 곱합니다.

### ⑤ 거울 모드와 왼손/오른손

화면을 `flip`으로 좌우 반전했기 때문에, 모델이 말하는 `Left`/`Right`가
사용자 입장과 반대로 보입니다. 코드에서 서로 바꿔서 표시합니다.

---

## 5. 실습 3 — 실행과 결과

```powershell
python gesture_webcam.py                 # 기본 (카메라 0번, 손 2개)
python gesture_webcam.py --camera 1      # 다른 카메라
python gesture_webcam.py --hands 1       # 손 1개만
```

**화면에 표시되는 것**
- 🟢 손 관절 연결선 / 🔴 관절 점
- 🔵 `Right: Thumb_Up (0.87)` 형태의 라벨
- 🟡 왼쪽 위 FPS

**종료:** `q` 또는 `ESC`

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `'pip' is not recognized` | PATH에 pip 없음 | `python -m pip ...` 사용 |
| `웹캠 0번을 열 수 없습니다` | 카메라 번호 다름 / 다른 앱이 사용 중 | `--camera 1` 시도, 화상회의 앱 종료 |
| 모델 파일을 못 찾음 | `.task`가 다른 폴더에 있음 | 스크립트와 같은 폴더로 이동 |
| 콘솔에 `Feedback manager...` 경고 다수 | MediaPipe 내부 로그 | **무시해도 됨** |
| 인식이 잘 안 됨 | 손이 너무 작거나 어두움 | 카메라에 가깝게, 밝은 곳에서 |

---

## 7. 용어 정리 (Glossary)

| 용어 | 한 줄 설명 |
|---|---|
| **MediaPipe** | Google의 온디바이스 ML 프레임워크. 손·얼굴·자세 인식 등 완성된 솔루션 제공 |
| **MediaPipe Tasks** | MediaPipe의 최신 API. `모델 파일 + Task 클래스` 조합으로 사용 |
| **Gesture Recognizer** | 손 제스처를 인식하는 MediaPipe Task |
| **`.task` 파일** | 여러 모델(손 검출·랜드마크·분류기)을 하나로 묶은 **모델 번들** |
| **모델 번들 (Model Bundle)** | 여러 개의 모델과 메타데이터를 하나의 파일로 패키징한 것 |
| **랜드마크 (Landmark)** | 신체의 특징점. 손은 21개 |
| **정규화 좌표** | 이미지 크기와 무관하게 0~1 사이로 표현한 좌표 |
| **Handedness** | 왼손/오른손 판별 결과 |
| **추론 (Inference)** | 학습된 모델에 새 입력을 넣어 결과를 얻는 과정 |
| **신뢰도 (Confidence / Score)** | 모델이 결과를 얼마나 확신하는지 (0~1) |
| **Running Mode** | 입력 종류에 따른 실행 방식: `IMAGE` / `VIDEO` / `LIVE_STREAM` |
| **타임스탬프 (Timestamp)** | 프레임의 시각(ms). VIDEO·LIVE_STREAM 모드에서 필수, 단조 증가해야 함 |
| **동기 / 비동기** | 결과를 바로 반환받는 방식(동기) vs 콜백으로 나중에 받는 방식(비동기) |
| **OpenCV (`cv2`)** | 영상 처리 라이브러리. 카메라 입력, 그리기, 창 표시 담당 |
| **BGR / RGB** | 색상 채널 순서. OpenCV는 BGR, MediaPipe는 RGB 사용 |
| **프레임 (Frame)** | 영상을 구성하는 한 장의 이미지 |
| **FPS** | Frames Per Second. 초당 처리 프레임 수 |
| **float16 양자화** | 모델 가중치를 16비트 실수로 줄여 파일 크기·속도를 개선하는 기법 |
| **TFLite / XNNPACK** | 모델 실행 엔진(TensorFlow Lite)과 그 CPU 가속기 |
| **CAP_DSHOW** | Windows의 DirectShow 카메라 백엔드. 웹캠 열기가 빠르고 안정적 |

---

## 8. 더 해보기

- [ ] `LIVE_STREAM` 모드 + 콜백으로 바꿔 FPS 비교하기
- [ ] 특정 제스처에 동작 연결하기 (예: 👍 → 스크린샷 저장)
- [ ] `canned_gestures_classifier_options`로 원하는 제스처만 허용하기
- [ ] **MediaPipe Model Maker**로 나만의 제스처 학습시키기

---

## 9. 참고 자료

- 공식 가이드: <https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer>
- Python 가이드: <https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python>
- 모델 다운로드: <https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task>

---

## 10. 심화 — 나만의 제스처 학습하기

> 🌐 **라이브 데모:** <https://higogo1.github.io/gesture-recognition/>
> (nike → Nike 로고 · ok → 👌 · 멈춰 → 😡)

### 10-1. 아이디어: "손 찾기"는 빌리고, "분류"만 새로 배운다

```
 웹캠 ─▶ MediaPipe (기존 모델) ─▶ 손 관절 21개 ─▶ 특징 63개 ─▶ 내 분류기 ─▶ "멈춰"
         └──── 그대로 재사용 ────┘                         └ 새로 학습 ┘
```

- 손을 찾고 관절 좌표를 뽑는 어려운 일은 기존 `gesture_recognizer.task`가 담당
- 우리는 **좌표 → 제스처 이름**만 학습 → 데이터가 제스처당 200개 정도면 충분
- 분류기는 scikit-learn의 작은 신경망(`MLPClassifier`) 사용
  (공식 도구 *MediaPipe Model Maker*는 구버전 TensorFlow가 필요해 Windows · Python 3.14에서 설치가 까다로움)

### 10-2. 특징 만들기 (정규화)

| 단계 | 하는 일 | 효과 |
|---|---|---|
| ① 월드 랜드마크 사용 | 화면 좌표 대신 미터 단위 3D 좌표 | 화면 비율 영향 없음 |
| ② 손목 기준 이동 | 모든 점 − 0번(손목) | 손이 화면 **어디에** 있든 같음 |
| ③ 크기로 나누기 | 손목에서 가장 먼 점까지 거리로 나눔 | 손이 **크든 작든**, 카메라와 **멀든 가깝든** 같음 |
| ④ 왼손 x축 반전 | Left이면 x × −1 | 왼손·오른손을 **한 모델**로 학습 |

→ 21개 × (x, y, z) = **63차원 벡터**. 이 코드는 `gesture_features.py`, `web/classifier.js`에 똑같이 들어 있습니다.

### 10-3. 실습: 수집 → 훈련 → 인식 (UI 앱)

```powershell
pip install scikit-learn joblib pillow
python gesture_studio.py
```

| 단계 | 조작 |
|---|---|
| ① 수집 | 제스처 이름 입력 → **추가** → 목록에서 선택 → `Space`로 녹화 시작/정지 |
| ② 훈련 | **훈련 시작** → 로그에 정확도 · 혼동 행렬 출력 → 자동으로 인식 모드 |
| ③ 인식 | 큰 글씨로 결과 + 클래스별 확률 막대, 최소 확신도 슬라이더 |

> 💡 **꼭 `none` 클래스를 만드세요.** 없으면 아무 손이나 억지로 다른 제스처로 분류합니다.
> 💡 각도 · 거리 · 손 방향을 바꿔가며 제스처당 **200개 이상** 모으면 좋습니다.

CLI 버전도 있습니다: `collect_gestures.py --labels ...` → `train_gestures.py` → `custom_gesture_webcam.py`

### 10-4. 웹으로 옮기기

```powershell
python export_web_model.py                      # models/*.joblib → web/model.json
python -m http.server 8000 --directory web      # http://localhost:8000
```

- 브라우저에서는 **MediaPipe JS(`@mediapipe/tasks-vision`)** 가 손 관절을 뽑고
- `model.json`(정규화 평균/표준편차 + 신경망 가중치)으로 **JS가 직접 행렬 계산** → 서버 불필요
- 검증: 같은 입력에 대해 Python과 JS 확률 차이 ≈ 1e-7
- 제스처별 오버레이는 `web/app.js`의 `OVERLAYS`에서 설정 (`"멈춰": { emoji: "😡" }`)
- 5프레임 연속 같은 결과일 때만 바뀌게 해서 깜빡임 방지

> ⚠️ 웹캠은 **보안 컨텍스트(https 또는 localhost)** 에서만 동작합니다. HTML 파일을 더블클릭해서 열면 카메라가 안 켜집니다.

### 10-5. GitHub Pages 배포

`.github/workflows/pages.yml` 이 `web/` 폴더를 자동 배포합니다.
`web/` 안의 파일을 바꿔 `main`에 push하면 1~2분 뒤 사이트에 반영됩니다.

```powershell
python export_web_model.py
git add web/model.json
git commit -m "Update model"
git push
```

### 10-6. 추가 용어

| 용어 | 한 줄 설명 |
|---|---|
| **특징 (Feature)** | 모델에 넣는 숫자 벡터. 여기서는 정규화된 관절 좌표 63개 |
| **월드 랜드마크** | 손 중심 기준 미터 단위 3D 좌표 (화면 크기와 무관) |
| **클래스 / 라벨** | 분류할 범주의 이름 (`nike`, `ok`, `멈춰`, `none`) |
| **MLP** | Multi-Layer Perceptron. 층이 여러 개인 기본 신경망 (63 → 64 → 32 → 클래스 수) |
| **StandardScaler** | 각 특징을 평균 0, 표준편차 1로 맞추는 전처리 |
| **ReLU / Softmax** | 은닉층 활성화 함수(음수→0) / 출력을 확률(합=1)로 바꾸는 함수 |
| **train/test split** | 데이터를 학습용 80% · 평가용 20%로 나누기 |
| **혼동 행렬** | 정답(행) vs 예측(열) 표. 어떤 제스처끼리 헷갈리는지 보여줌 |
| **joblib** | scikit-learn 모델을 파일로 저장/불러오는 도구 |
| **Tkinter** | 파이썬 기본 GUI 라이브러리 |
| **보안 컨텍스트** | https 또는 localhost. 카메라 같은 민감한 API가 허용되는 환경 |
| **GitHub Pages / Actions** | GitHub의 정적 웹 호스팅 / 자동 실행(CI) 도구 |

---

## 📂 파일 구성

```
gesture recognition/
├── README.md                  ← 이 강의노트
├── gesture_webcam.py          ← 기본 제스처 7종 웹캠 인식
├── gesture_features.py        ← 공통: 관절 → 특징 63개
├── gesture_studio.py          ← UI 앱: 수집 · 훈련 · 인식
├── collect_gestures.py        ← (CLI) 수집
├── train_gestures.py          ← (CLI) 훈련 / train() 함수
├── custom_gesture_webcam.py   ← (CLI) 내 제스처 인식
├── export_web_model.py        ← 모델 → web/model.json
├── web/                       ← 웹 버전 (GitHub Pages 배포)
│   ├── index.html
│   ├── app.js                 ← 웹캠 · 인식 · 오버레이
│   ├── classifier.js          ← 특징 계산 + MLP 추론 (JS)
│   └── model.json             ← 훈련된 가중치
├── .github/workflows/pages.yml
├── gesture_recognizer.task    ← (직접 다운로드, git 미포함)
├── data/                      ← 수집 데이터 (git 미포함)
└── models/                    ← 훈련 모델 .joblib (git 미포함)
```
