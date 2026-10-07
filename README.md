# 🖐️ MediaPipe 제스처 인식 — 웹캠부터 나만의 제스처 학습 · 웹 배포까지

> **작성일:** 2026-10-07
> **주제:** MediaPipe로 손 제스처를 인식하고, 원하는 제스처를 직접 학습시켜 웹에 배포하기
> **환경:** Windows 11 · Python 3.14 · mediapipe 1.1.0 · opencv-python 5.0.0 · scikit-learn 1.9
> 🌐 **라이브 데모:** <https://higogo1.github.io/gesture-recognition/> (nike → Nike 로고 · ok → 👌 · 멈춰 → 😡)

---

## 📌 한눈에 보기 (TL;DR)

```
[기본]   python gesture_webcam.py          기본 제스처 7종 인식
[학습]   python gesture_studio.py          ① 수집 → ② 훈련 → ③ 인식 (UI)
[웹]     python export_web_model.py        모델 → web/model.json
         python -m http.server 8000 --directory web   →  http://localhost:8000
[배포]   git add web/model.json && git commit -m "Update model" && git push
```

### 오늘 만든 것 — 6단계 흐름

| # | 단계 | 결과물 | 장 |
|---|---|---|---|
| 1 | 기본 제스처 인식 | `gesture_webcam.py` | [2–4장](#2-핵심-개념) |
| 2 | 나만의 제스처 수집 · 훈련 (CLI) | `collect_gestures.py` · `train_gestures.py` · `custom_gesture_webcam.py` | [5장](#5-심화--나만의-제스처-학습) |
| 3 | 수집 · 훈련 · 인식을 한 창에서 (UI) | `gesture_studio.py` | [5장](#5-심화--나만의-제스처-학습) |
| 4 | 웹 버전 (nike → 로고, ok → 👌) | `web/` | [6장](#6-웹으로-옮기기) |
| 5 | 새 동작 추가 (멈춰 → 😡) | `web/app.js` 한 줄 | [7장](#7-️-새-동작-추가--훈련하는-법-step-by-step) |
| 6 | GitHub Pages 배포 | `.github/workflows/pages.yml` | [8장](#8-github-pages-배포) |

---

## 📚 목차

1. [학습 목표](#1-학습-목표)
2. [핵심 개념](#2-핵심-개념)
3. [실습 1 — 환경 준비](#3-실습-1--환경-준비)
4. [실습 2 — 기본 제스처 인식 코드](#4-실습-2--기본-제스처-인식-코드)
5. [심화 — 나만의 제스처 학습](#5-심화--나만의-제스처-학습)
6. [웹으로 옮기기](#6-웹으로-옮기기)
7. [⭐ 새 동작 추가 · 훈련하는 법 (Step by Step)](#7-️-새-동작-추가--훈련하는-법-step-by-step)
8. [GitHub Pages 배포](#8-github-pages-배포)
9. [트러블슈팅](#9-트러블슈팅)
10. [용어 정리](#10-용어-정리-glossary)
11. [파일 구성](#11-파일-구성)
12. [더 해보기 · 참고 자료](#12-더-해보기--참고-자료)

---

## 1. 학습 목표

- [x] MediaPipe **Task** 구조(모델 파일 + Task API)를 이해한다
- [x] 웹캠 프레임을 MediaPipe로 추론하고 결과를 화면에 그린다
- [x] 손 관절 좌표를 **특징**으로 바꿔 나만의 제스처 분류기를 학습시킨다
- [x] 학습한 모델을 **브라우저(JS)** 로 옮겨 실행한다
- [x] GitHub Actions로 **GitHub Pages** 에 자동 배포한다

---

## 2. 핵심 개념

### 2-1. 제스처 인식은 2단계 파이프라인

```
 웹캠 프레임 (BGR)
      │  BGR → RGB
      ▼
 ┌──────────────────────────┐
 │ ① Hand Landmark 모델      │  손을 찾고 관절 21개 좌표 추출
 └──────────────────────────┘
      │  21개 랜드마크 (x, y, z)
      ▼
 ┌──────────────────────────┐
 │ ② Gesture Classifier     │  좌표 모양을 보고 제스처 분류
 └──────────────────────────┘
      ▼
 제스처 이름 + 점수, 왼손/오른손, 랜드마크
```

> 💡 분류기는 *이미지*가 아니라 *관절 좌표*를 봅니다 → 배경 · 조명 변화에 강함.
> 👉 **이 구조 덕분에 ②만 내 것으로 바꾸면 나만의 제스처를 인식할 수 있습니다** (5장).

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
        1 |          /
           \        /
               0                ← 손목 (WRIST)
```

| 번호 | 부위 | 번호 | 부위 |
|---|---|---|---|
| 0 | 손목 | 9–12 | 중지 |
| 1–4 | 엄지 | 13–16 | 약지 |
| 5–8 | 검지 | 17–20 | 새끼 |

### 2-3. 기본 제공 제스처 7종

| 라벨 | 모양 | 라벨 | 모양 |
|---|---|---|---|
| `Closed_Fist` | ✊ | `Thumb_Up` | 👍 |
| `Open_Palm` | ✋ | `Victory` | ✌️ |
| `Pointing_Up` | ☝️ | `ILoveYou` | 🤟 |
| `Thumb_Down` | 👎 | `None` | 해당 없음 |

### 2-4. Running Mode 3가지

| 모드 | 호출 함수 | 언제 쓰나 |
|---|---|---|
| `IMAGE` | `recognize()` | 사진 1장 |
| `VIDEO` | `recognize_for_video()` | 동영상 · 웹캠 (**오늘 사용**, 동기식이라 단순) |
| `LIVE_STREAM` | `recognize_async()` + 콜백 | 지연 최소화가 중요한 실시간 앱 |

> ⚠️ `VIDEO` / `LIVE_STREAM`에서 **타임스탬프(ms)는 반드시 계속 증가**해야 합니다.

---

## 3. 실습 1 — 환경 준비

```powershell
# ① 모델 파일 (약 8.4MB, 스크립트와 같은 폴더에)
curl -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

# ② 라이브러리
python -m pip install mediapipe opencv-python scikit-learn joblib pillow

# ③ 확인
python -c "import mediapipe as mp, cv2, sklearn; print(mp.__version__, cv2.__version__, sklearn.__version__)"
```

> 💡 Windows에서 `pip`가 안 잡히면 `python -m pip` 으로 실행.
> 📁 `gesture_recognizer.task`, `data/`, `models/` 는 git에 올리지 않습니다 (`.gitignore`).

---

## 4. 실습 2 — 기본 제스처 인식 코드

```powershell
python gesture_webcam.py                 # 카메라 0번, 손 2개
python gesture_webcam.py --camera 1      # 다른 카메라
```

### 코드 핵심 4줄 흐름 ([`gesture_webcam.py`](gesture_webcam.py))

```python
# ① 인식기 생성
options = vision.GestureRecognizerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ② 프레임 변환 (OpenCV=BGR, MediaPipe=RGB)
rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# ③ 추론
result = recognizer.recognize_for_video(mp_image, timestamp_ms)

# ④ 결과 사용
result.gestures[i][0].category_name     # "Thumb_Up"
result.handedness[i][0].category_name   # "Left" / "Right"
result.hand_landmarks[i]                # 화면 기준 0~1 좌표 21개
result.hand_world_landmarks[i]          # 미터 단위 3D 좌표 21개 (5장에서 사용)
```

> ⚠️ **흔한 실수:** BGR → RGB 변환 누락 → 인식률 급감
> 💡 `cv2.flip`으로 거울 모드를 쓰면 Left/Right가 반대로 보이므로 표시할 때 바꿔줍니다.

---

## 5. 심화 — 나만의 제스처 학습

### 5-1. 아이디어: "손 찾기"는 빌리고, "분류"만 새로 배운다

```
 웹캠 ─▶ MediaPipe (기존 모델) ─▶ 손 관절 21개 ─▶ 특징 63개 ─▶ 내 분류기 ─▶ "멈춰"
         └───── 그대로 재사용 ─────┘                        └ 새로 학습 ┘
```

- 어려운 일(손 찾기 · 관절 추출)은 기존 `gesture_recognizer.task`가 담당
- 우리는 **좌표 → 제스처 이름**만 학습 → 제스처당 **200개** 정도면 충분
- 분류기: scikit-learn `StandardScaler` + `MLPClassifier(64, 32)`
- 공식 도구 *MediaPipe Model Maker* 는 구버전 TensorFlow가 필요해 Windows · Python 3.14에선 설치가 까다로워 이 방식을 선택

### 5-2. 특징 만들기 — 정규화 4단계 ([`gesture_features.py`](gesture_features.py))

| 단계 | 하는 일 | 효과 |
|---|---|---|
| ① 월드 랜드마크 사용 | 화면 좌표 대신 미터 단위 3D 좌표 | 화면 비율 영향 없음 |
| ② 손목 기준 이동 | 모든 점 − 0번(손목) | 손이 화면 **어디에** 있든 같음 |
| ③ 크기로 나누기 | 손목에서 가장 먼 점까지 거리로 나눔 | 손 **크기** · 카메라 **거리** 무관 |
| ④ 왼손 x축 반전 | Left이면 x × −1 | 왼손 · 오른손을 **한 모델**로 |

→ 21개 × (x, y, z) = **63차원 벡터**

> ⚠️ **수집 · 훈련 · 추론(파이썬, 웹)이 반드시 같은 특징 함수를 써야 합니다.**
> 그래서 `gesture_features.py` 와 `web/classifier.js` 가 똑같은 계산을 합니다.

### 5-3. 두 가지 사용 방식

| | UI 앱 (추천) | CLI |
|---|---|---|
| 수집 | `gesture_studio.py` ① | `collect_gestures.py --labels none ok 멈춰` |
| 훈련 | `gesture_studio.py` ② 버튼 | `train_gestures.py` |
| 인식 | `gesture_studio.py` ③ | `custom_gesture_webcam.py` |

### 5-4. Gesture Studio 화면

```
┌─────────────────────────┬──────────────────────┐
│                         │ 카메라 [0] [열기]      │
│      웹캠 영상            │ 모드 ◉①수집 ○③인식    │
│   (손 관절 표시)           ├──────────────────────┤
│                         │ ① 수집                │
│                         │  제스처      샘플 수    │
├─────────────────────────┤  none          221   │
│ [수집] 멈춰 · 녹화 중      │  멈춰          229   │
├─────────────────────────┤ [이름 입력    ][추가]   │
│ 로그 (훈련 결과)           │ [ ● 녹화 시작 (Space) ]│
│                         ├──────────────────────┤
│                         │ ② 훈련 [훈련 시작]      │
│                         ├──────────────────────┤
│                         │ ③ 인식     멈춰        │
│                         │ 최소 확신도 ━━●━ 0.70   │
└─────────────────────────┴──────────────────────┘
```

### 5-5. 훈련 결과 읽는 법

오늘 실제 훈련 결과 (클래스당 약 220개 수집, 20%로 평가):

```
              precision    recall  f1-score   support
        nike      1.000     1.000     1.000        44
        none      0.933     0.955     0.944        44
          ok      0.957     0.957     0.957        46
          멈춰      0.978     0.957     0.967        46
    accuracy                          0.967       180

혼동 행렬 (행=정답, 열=예측): ['nike', 'none', 'ok', '멈춰']
[[44  0  0  0]     ← nike 44개 전부 정답
 [ 0 42  1  1]     ← none 중 1개는 ok, 1개는 멈춰로 착각
 [ 0  2 44  0]     ← ok 중 2개를 none으로 착각
 [ 0  1  1 44]]
```

→ 전체 정확도 **96.7%**. `none` ↔ `ok` 가 가장 많이 헷갈림 → 두 클래스 데이터를 더 모으면 개선

| 지표 | 의미 | 목표 |
|---|---|---|
| **precision** | "멈춰"라고 예측한 것 중 진짜 멈춰 비율 | 0.95 ↑ |
| **recall** | 진짜 멈춰 중 멈춰라고 맞힌 비율 | 0.95 ↑ |
| **혼동 행렬** | 대각선 밖 숫자 = 헷갈린 개수 | 대각선에 몰릴수록 좋음 |

> 💡 두 제스처가 서로 자주 헷갈리면 → 그 두 제스처 데이터를 더 다양하게 모으세요.

---

## 6. 웹으로 옮기기

```powershell
python export_web_model.py                      # models/*.joblib → web/model.json
python -m http.server 8000 --directory web      # 크롬에서 http://localhost:8000
```

### 6-1. 구조

```
 [Python]  훈련 ─▶ custom_gesture.joblib ─▶ export_web_model.py ─▶ model.json
                                                                     │
 [브라우저] 웹캠 ─▶ MediaPipe JS ─▶ 관절 ─▶ classifier.js (JS 행렬 계산) ─▶ 오버레이
```

- 브라우저에서 **`@mediapipe/tasks-vision`** 이 손 관절 추출
- `model.json` = 정규화 평균/표준편차 + 신경망 가중치 → **JS가 직접 계산 (서버 불필요)**
- 검증 결과: 같은 입력에 대해 Python vs JS 확률 차이 **≈ 1e-7**
- **5프레임 연속** 같은 결과일 때만 오버레이가 바뀌어 깜빡임 방지
- 오버레이는 손 위쪽을 따라다니며 팝업

### 6-2. 오버레이 설정 ([`web/app.js`](web/app.js))

```js
const OVERLAYS = {
  nike: { html: `<svg ...>` },   // SVG / 이미지
  ok:   { emoji: "👌" },          // 이모지
  "멈춰": { emoji: "😡" },
};
```

> ⚠️ 웹캠은 **https 또는 localhost** 에서만 동작합니다. HTML 파일을 더블클릭하면 카메라가 안 켜집니다.

---

## 7. ⭐ 새 동작 추가 · 훈련하는 법 (Step by Step)

> 예시: **"따봉" 제스처를 추가하고, 웹에서 👍 를 띄우기**

### ✅ 체크리스트

```
□ 1. 카메라 쓰는 다른 프로그램/웹 탭 닫기
□ 2. gesture_studio.py 실행
□ 3. 제스처 이름 추가 → 200개 이상 수집
□ 4. 훈련 → 정확도 · 혼동 행렬 확인
□ 5. 인식 모드에서 직접 테스트
□ 6. export_web_model.py 로 웹용 내보내기
□ 7. web/app.js OVERLAYS 에 한 줄 추가
□ 8. localhost:8000 에서 확인
□ 9. git push → GitHub Pages 자동 배포
```

### Step 1. 앱 실행

```powershell
cd "C:\Users\sejin\Desktop\gesture recognition"
python gesture_studio.py
```

### Step 2. 제스처 추가 · 수집

1. 오른쪽 **① 수집** 입력칸에 `따봉` 입력 → **추가** (Enter도 됨)
2. 목록에서 `따봉` 이 선택된 상태 확인
3. 카메라 앞에서 동작을 취하고 **`Space`** → 녹화 시작 (0.1초마다 1개 저장)
4. 녹화 중 **손을 조금씩 돌리고, 기울이고, 멀리/가까이** 움직이기
5. 샘플 수가 **200 이상**이 되면 `Space` → 정지

> 💡 **좋은 데이터 수집 요령**
> - 한 자세로만 찍으면 실제 사용할 때 인식이 안 됩니다 → **다양하게**
> - 왼손 · 오른손 둘 다 OK (자동으로 맞춰줌)
> - 제스처끼리 샘플 수를 **비슷하게** 맞추기
> - `none`(아무 동작 아님: 편하게 둔 손, 애매한 손 모양)은 **필수**. 새 제스처를 만들 때 비슷한데 아닌 손 모양도 `none` 에 추가하면 오인식이 줄어듭니다.
> - 잘못 녹화했다면 → 제스처 선택 → **선택한 제스처 데이터 삭제** → 다시 수집

### Step 3. 훈련

1. **② 훈련 → 훈련 시작**
2. 하단 로그에서 `따봉` 의 precision / recall 이 **0.95 이상**인지 확인
3. 끝나면 자동으로 **③ 인식** 모드로 바뀜

### Step 4. 테스트

- 동작을 해서 큰 글씨로 `따봉` 이 뜨는지, 확률 막대가 높은지 확인
- 다른 제스처와 헷갈리면 → **Step 2로 돌아가** 데이터 추가 → 재훈련
- `Unknown` 이 자주 뜨면 → 최소 확신도 슬라이더를 낮추거나 데이터 추가

### Step 5. 웹에 반영

```powershell
# (gesture_studio 창을 닫아 카메라를 놓아준 뒤)
python export_web_model.py
```

출력에 `클래스=[..., '따봉']` 이 보이면 성공. 이어서 [`web/app.js`](web/app.js) 의 `OVERLAYS` 에 한 줄 추가:

```js
const OVERLAYS = {
  nike: { html: `...` },
  ok: { emoji: "👌" },
  "멈춰": { emoji: "😡" },
  "따봉": { emoji: "👍" },          // ← 추가
  // 이미지로 하려면: rock: { html: '<img src="assets/rock.png" width="240">' }
};
```

> ⚠️ **키 이름 = 수집할 때 입력한 라벨과 똑같이.** 영어 라벨은 **소문자**로 적으세요 (비교할 때 소문자로 바꿈).
> 💡 `OVERLAYS` 에 없는 제스처도 인식 · 확률 막대는 표시됩니다 (오버레이만 안 뜸).

### Step 6. 로컬 확인

```powershell
python -m http.server 8000 --directory web
```
크롬에서 `localhost:8000` → **F5 새로고침** → 카메라 시작 → 동작 → 👍 확인

### Step 7. 배포

```powershell
git add web/model.json web/app.js
git commit -m "Add 따봉 gesture"
git push
```
1~2분 뒤 <https://higogo1.github.io/gesture-recognition/> 에 반영 (브라우저 강력 새로고침 `Ctrl+F5`).

### 🔁 요약 한 장

```
 gesture_studio.py ─(수집·훈련)─▶ models/custom_gesture.joblib
                                         │ export_web_model.py
                                         ▼
                                  web/model.json  +  web/app.js (OVERLAYS)
                                         │ git push
                                         ▼
                                  GitHub Pages 자동 배포
```

> 📦 **백업 주의:** `data/gestures.csv`(수집 데이터)는 git에 올리지 않습니다. 지우면 처음부터 다시 수집해야 하니 따로 백업하세요.

---

## 8. GitHub Pages 배포

- [`.github/workflows/pages.yml`](.github/workflows/pages.yml): `main` 브랜치의 **`web/` 폴더가 바뀌면** GitHub Actions가 자동 배포
- 진행 상황: 저장소 → **Actions** 탭 (✅ 초록 체크면 완료)
- 무료 계정은 **공개(Public) 저장소**에서만 Pages 사용 가능
- Pages는 https 이므로 **휴대폰에서도** 카메라가 동작

```
git push ─▶ Actions: checkout → upload web/ → deploy ─▶ https://higogo1.github.io/gesture-recognition/
```

---

## 9. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `'pip' is not recognized` | PATH에 pip 없음 | `python -m pip ...` |
| 웹캠을 열 수 없음 / `Could not start video source` | 다른 프로그램이 카메라 사용 중 | gesture_studio · 다른 웹 탭 · 화상회의 앱 닫기 |
| 카메라 번호가 다름 | 노트북 + 외장 카메라 | 앱에서 카메라 번호 변경 / `--camera 1` |
| 웹에서 카메라가 안 켜짐 | HTML 파일을 직접 열었음 | `http.server` 로 `localhost` 접속 또는 Pages 사용 |
| 새 제스처가 웹에 안 나옴 | `export_web_model.py` 안 함 / 캐시 | 내보내기 → `Ctrl+F5` |
| 인식은 되는데 오버레이가 안 뜸 | `OVERLAYS` 키 불일치 | 라벨과 똑같이, 영어는 소문자 |
| 아무 손이나 특정 제스처로 인식 | `none` 클래스 부족 | `none` 데이터 다양하게 추가 |
| 훈련 실패: 샘플 10개 미만 | 데이터 부족 | 클래스당 200개 이상 수집 |
| `Feedback manager...` 경고 다수 | MediaPipe 내부 로그 | 무시해도 됨 |
| 한글 라벨이 영상 위에서 깨짐 | OpenCV `putText` 는 한글 미지원 | Studio · 웹은 패널에 표시하므로 문제 없음 |

---

## 10. 용어 정리 (Glossary)

### 🧠 AI · 머신러닝

| 용어 | 한 줄 설명 |
|---|---|
| **추론 (Inference)** | 학습된 모델에 새 입력을 넣어 결과를 얻는 것 |
| **훈련 (Training)** | 데이터로 모델의 가중치를 조정하는 것 |
| **특징 (Feature)** | 모델에 넣는 숫자 벡터. 여기선 정규화된 관절 좌표 63개 |
| **정규화 (Normalization)** | 위치 · 크기 등 불필요한 차이를 없애 값을 비교 가능하게 맞추기 |
| **클래스 / 라벨** | 분류할 범주의 이름 (`nike`, `ok`, `멈춰`, `none`) |
| **none 클래스** | "어떤 제스처도 아님"을 학습시키는 클래스. 오인식 방지 |
| **신뢰도 (Confidence / Score)** | 모델의 확신 정도 (0~1) |
| **MLP** | 다층 퍼셉트론. 기본 신경망 (63 → 64 → 32 → 클래스 수) |
| **StandardScaler** | 각 특징을 평균 0, 표준편차 1로 맞추는 전처리 |
| **ReLU / Softmax** | 은닉층 활성화(음수→0) / 출력을 합이 1인 확률로 변환 |
| **가중치 (Weights)** | 신경망이 학습한 숫자들. `model.json` 에 저장 |
| **train/test split** | 학습용 80% · 평가용 20%로 데이터 나누기 |
| **precision / recall** | 예측의 정확성 / 놓치지 않는 정도 |
| **혼동 행렬** | 정답(행) vs 예측(열) 표. 헷갈리는 쌍을 보여줌 |
| **float16 양자화** | 가중치를 16비트로 줄여 크기 · 속도 개선 |

### 🖐️ MediaPipe

| 용어 | 한 줄 설명 |
|---|---|
| **MediaPipe** | Google의 온디바이스 ML 프레임워크 |
| **MediaPipe Tasks** | `모델 파일 + Task 클래스` 조합의 최신 API (Python · JS · Android · iOS) |
| **Gesture Recognizer** | 손 제스처 인식 Task |
| **`.task` 파일 / 모델 번들** | 여러 모델(손 검출 · 랜드마크 · 분류기)을 하나로 묶은 파일 |
| **랜드마크 (Landmark)** | 신체 특징점. 손은 21개 |
| **정규화 좌표** | 화면 크기와 무관한 0~1 좌표 (`hand_landmarks`) |
| **월드 랜드마크** | 손 중심 기준 미터 단위 3D 좌표 (`hand_world_landmarks`) |
| **Handedness** | 왼손 / 오른손 판별 |
| **Running Mode** | `IMAGE` / `VIDEO` / `LIVE_STREAM` |
| **타임스탬프** | 프레임 시각(ms). 단조 증가해야 함 |
| **Model Maker** | MediaPipe 공식 전이학습 도구 (TensorFlow 필요) |
| **tasks-vision** | MediaPipe의 웹(JS) 패키지 |

### 💻 개발 도구 · 웹

| 용어 | 한 줄 설명 |
|---|---|
| **OpenCV (`cv2`)** | 영상 처리 라이브러리 (카메라 · 그리기 · 창) |
| **BGR / RGB** | 색 채널 순서. OpenCV=BGR, MediaPipe=RGB |
| **프레임 / FPS** | 영상의 한 장 / 초당 처리 장수 |
| **scikit-learn** | 파이썬 머신러닝 라이브러리 |
| **joblib** | 파이썬 모델 저장/불러오기 도구 (`.joblib`) |
| **Tkinter** | 파이썬 기본 GUI 라이브러리 |
| **CSV** | 쉼표로 구분된 표 형식 텍스트 (`data/gestures.csv`) |
| **JSON** | 웹에서 쓰는 데이터 형식 (`web/model.json`) |
| **오버레이 (Overlay)** | 영상 위에 겹쳐 띄우는 그림 · 이모지 |
| **보안 컨텍스트** | https 또는 localhost. 카메라 API가 허용되는 환경 |
| **http.server** | 파이썬 내장 간이 웹 서버 |
| **CDN** | 라이브러리를 빠르게 받아오는 공용 서버 (jsDelivr) |
| **GitHub Pages** | GitHub의 무료 정적 웹 호스팅 |
| **GitHub Actions** | push 등에 반응해 자동 실행되는 CI/CD |
| **.gitignore** | git에 올리지 않을 파일 목록 |

---

## 11. 파일 구성

```
gesture recognition/
├── README.md                  ← 이 강의노트
├── gesture_webcam.py          ← [1] 기본 제스처 7종 웹캠 인식
├── gesture_features.py        ← 공통: 관절 → 특징 63개, 경로 설정
├── gesture_studio.py          ← [3] UI 앱: 수집 · 훈련 · 인식
├── collect_gestures.py        ← [2] (CLI) 수집
├── train_gestures.py          ← [2] (CLI) 훈련 / train() 함수
├── custom_gesture_webcam.py   ← [2] (CLI) 내 제스처 인식
├── export_web_model.py        ← [4] 모델 → web/model.json
├── web/                       ← [4] 웹 버전 (GitHub Pages 배포 대상)
│   ├── index.html             ←     화면
│   ├── app.js                 ←     웹캠 · 인식 · 오버레이(OVERLAYS)
│   ├── classifier.js          ←     특징 계산 + MLP 추론 (JS)
│   └── model.json             ←     훈련된 가중치
├── .github/workflows/pages.yml← [6] 자동 배포
├── .gitignore
├── gesture_recognizer.task    ← (직접 다운로드 · git 미포함)
├── data/gestures.csv          ← 수집 데이터 (git 미포함 · 백업 권장)
└── models/custom_gesture.joblib ← 훈련 모델 (git 미포함)
```

---

## 12. 더 해보기 · 참고 자료

### 더 해보기
- [ ] 두 손을 함께 쓰는 제스처 (특징 126개)
- [ ] 정지 자세가 아닌 **움직임**(손 흔들기) 인식 → 여러 프레임을 묶어 특징으로
- [ ] 제스처로 동작 실행 (예: 👍 → 스크린샷, 멈춰 → 음악 정지)
- [ ] 오버레이에 효과음 추가
- [ ] MediaPipe Model Maker로 학습해 결과 비교

### 참고 자료
- 공식 가이드: <https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer>
- Python 가이드: <https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python>
- Web 가이드: <https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/web_js>
- 모델 파일: <https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task>
