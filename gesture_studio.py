"""Gesture Studio — 나만의 제스처 수집 · 훈련 · 인식을 한 화면에서

실행: python gesture_studio.py
"""
import csv
import os
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import joblib
import mediapipe as mp
from PIL import Image, ImageTk

from gesture_features import (CLASSIFIER_PATH, DATA_PATH, NUM_FEATURES,
                              create_recognizer, landmarks_to_features)
from gesture_webcam import draw_hand
from train_gestures import train

SAVE_INTERVAL = 0.1  # 녹화 중 저장 간격(초)
HEADER = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]


# ---------- 데이터 파일 ----------
def read_rows():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f) if row and row[0] != "label"]


def append_row(label, feats):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)
    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerow([label] + [f"{v:.6f}" for v in feats])


def remove_label(label):
    rows = [r for r in read_rows() if r[0] != label]
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


# ---------- 앱 ----------
class GestureStudio:
    def __init__(self, root):
        self.root = root
        root.title("Gesture Studio")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.recognizer = create_recognizer(num_hands=1)
        self.start = time.monotonic()
        self.last_ts = -1
        self.recording = False
        self.last_saved = 0.0
        self.counts = Counter(r[0] for r in read_rows())
        self.clf = None
        self.prob_bars = {}
        self.log_queue = queue.Queue()
        self.training = False

        self.build_ui()
        self.refresh_labels()
        self.load_classifier()
        self.open_camera()
        root.bind("<space>", self.on_space)
        self.update_loop()

    # ----- UI 구성 -----
    def build_ui(self):
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Malgun Gothic", 22, "bold"))
        style.configure("Rec.TButton", font=("Malgun Gothic", 11, "bold"))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # 왼쪽: 영상 + 로그
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = ttk.Label(left)
        self.video.pack()
        self.status = ttk.Label(left, text="", foreground="gray")
        self.status.pack(anchor="w", pady=(4, 0))
        self.log_box = ScrolledText(left, height=10, font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        # 오른쪽: 컨트롤
        right = ttk.Frame(main, width=300)
        right.pack(side="left", fill="y", padx=(8, 0))

        cam = ttk.LabelFrame(right, text="카메라", padding=6)
        cam.pack(fill="x")
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=9, width=5, textvariable=self.cam_var).pack(side="left")
        ttk.Button(cam, text="열기", command=self.open_camera).pack(side="left", padx=4)

        self.mode = tk.StringVar(value="collect")
        mode = ttk.LabelFrame(right, text="모드", padding=6)
        mode.pack(fill="x", pady=6)
        ttk.Radiobutton(mode, text="① 수집", value="collect", variable=self.mode,
                        command=self.on_mode).pack(side="left")
        ttk.Radiobutton(mode, text="③ 인식", value="infer", variable=self.mode,
                        command=self.on_mode).pack(side="left", padx=8)

        # 수집
        col = ttk.LabelFrame(right, text="① 수집", padding=6)
        col.pack(fill="x")
        self.tree = ttk.Treeview(col, columns=("count",), height=7, selectmode="browse")
        self.tree.heading("#0", text="제스처")
        self.tree.heading("count", text="샘플 수")
        self.tree.column("#0", width=160)
        self.tree.column("count", width=80, anchor="e")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.stop_recording())

        row = ttk.Frame(col)
        row.pack(fill="x", pady=4)
        self.new_label = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.new_label)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(row, text="추가", width=6, command=self.add_label).pack(side="left", padx=(4, 0))
        ttk.Button(col, text="선택한 제스처 데이터 삭제", command=self.delete_label).pack(fill="x")
        self.rec_btn = ttk.Button(col, text="● 녹화 시작  (Space)", style="Rec.TButton",
                                  command=self.toggle_recording)
        self.rec_btn.pack(fill="x", pady=(6, 0), ipady=6)
        ttk.Label(col, text="※ 'none'(아무 동작 아님) 제스처를 꼭 포함하세요",
                  foreground="gray").pack(anchor="w", pady=(4, 0))

        # 훈련
        tr = ttk.LabelFrame(right, text="② 훈련", padding=6)
        tr.pack(fill="x", pady=6)
        self.train_btn = ttk.Button(tr, text="훈련 시작", command=self.start_training)
        self.train_btn.pack(fill="x", ipady=4)
        self.progress = ttk.Progressbar(tr, mode="indeterminate")
        self.progress.pack(fill="x", pady=(4, 0))

        # 인식
        inf = ttk.LabelFrame(right, text="③ 인식", padding=6)
        inf.pack(fill="both", expand=True)
        self.result_label = ttk.Label(inf, text="-", style="Big.TLabel", anchor="center")
        self.result_label.pack(fill="x")
        thr = ttk.Frame(inf)
        thr.pack(fill="x", pady=4)
        ttk.Label(thr, text="최소 확신도").pack(side="left")
        self.min_score = tk.DoubleVar(value=0.7)
        self.min_score_text = ttk.Label(thr, text="0.70", width=5)
        self.min_score_text.pack(side="right")
        ttk.Scale(thr, from_=0.0, to=1.0, variable=self.min_score,
                  command=lambda v: self.min_score_text.config(text=f"{float(v):.2f}")
                  ).pack(side="left", fill="x", expand=True, padx=4)
        self.prob_frame = ttk.Frame(inf)
        self.prob_frame.pack(fill="x")

    # ----- 로그 -----
    def log(self, msg):
        self.log_queue.put(msg)  # 훈련 스레드에서도 안전하게 호출

    def flush_log(self):
        while not self.log_queue.empty():
            self.log_box.insert("end", self.log_queue.get() + "\n")
            self.log_box.see("end")

    # ----- 카메라 -----
    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        idx = self.cam_var.get()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.log(f"카메라 {idx}번 연결됨")
        else:
            self.log(f"⚠️ 카메라 {idx}번을 열 수 없습니다. 다른 번호를 선택하세요.")

    # ----- 라벨 -----
    def refresh_labels(self, select=None):
        selected = select or self.selected_label()
        existing = set(self.tree.get_children())
        for name in sorted(set(self.counts) | existing):
            if self.tree.exists(name):
                self.tree.set(name, "count", self.counts[name])
            else:
                self.tree.insert("", "end", iid=name, text=name, values=(self.counts[name],))
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)
        elif self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])

    def selected_label(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def add_label(self):
        name = self.new_label.get().strip().replace(",", "_")
        if not name:
            return
        if not self.tree.exists(name):
            self.tree.insert("", "end", iid=name, text=name, values=(0,))
        self.tree.selection_set(name)
        self.new_label.set("")
        self.log(f"제스처 추가: {name}")

    def delete_label(self):
        name = self.selected_label()
        if not name:
            return
        if not messagebox.askyesno("삭제", f"'{name}' 데이터 {self.counts[name]}개를 삭제할까요?"):
            return
        self.stop_recording()
        if self.counts[name]:
            remove_label(name)
        del self.counts[name]
        self.tree.delete(name)
        self.refresh_labels()
        self.log(f"제스처 삭제: {name}")

    # ----- 녹화 -----
    def on_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        self.toggle_recording()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        if self.mode.get() != "collect":
            self.mode.set("collect")
        if not self.selected_label():
            messagebox.showinfo("안내", "먼저 제스처 이름을 추가하고 선택하세요.")
            return
        self.recording = True
        self.rec_btn.config(text="■ 녹화 중지  (Space)")
        self.log(f"녹화 시작: {self.selected_label()}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.rec_btn.config(text="● 녹화 시작  (Space)")
            self.log("녹화 중지")

    def on_mode(self):
        self.stop_recording()
        if self.mode.get() == "infer" and self.clf is None:
            self.log("⚠️ 훈련된 모델이 없습니다. 먼저 ② 훈련을 실행하세요.")

    # ----- 훈련 -----
    def start_training(self):
        if self.training:
            return
        self.stop_recording()
        self.training = True
        self.train_btn.config(state="disabled")
        self.progress.start(10)
        self.log("\n===== 훈련 시작 =====")
        threading.Thread(target=self.train_worker, daemon=True).start()

    def train_worker(self):
        try:
            train(log=self.log)
            ok = True
        except Exception as e:
            self.log(f"⚠️ 훈련 실패: {e}")
            ok = False
        self.root.after(0, self.training_done, ok)

    def training_done(self, ok):
        self.training = False
        self.train_btn.config(state="normal")
        self.progress.stop()
        if ok:
            self.load_classifier()
            self.mode.set("infer")
            self.log("✅ 훈련 완료 — 인식 모드로 전환했습니다.")

    def load_classifier(self):
        if not os.path.exists(CLASSIFIER_PATH):
            return
        self.clf = joblib.load(CLASSIFIER_PATH)
        for w in self.prob_frame.winfo_children():
            w.destroy()
        self.prob_bars = {}
        for i, name in enumerate(self.clf.classes_):
            ttk.Label(self.prob_frame, text=str(name), width=12).grid(row=i, column=0, sticky="w")
            bar = ttk.Progressbar(self.prob_frame, maximum=1.0, length=150)
            bar.grid(row=i, column=1, pady=1)
            self.prob_bars[str(name)] = bar
        self.log(f"모델 로드: {[str(c) for c in self.clf.classes_]}")

    # ----- 메인 루프 -----
    def update_loop(self):
        self.flush_log()
        ok, frame = (self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None))
        if ok:
            self.process(frame)
        self.root.after(15, self.update_loop)

    def process(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.recognizer.recognize_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        feats = None
        if result.hand_world_landmarks:
            draw_hand(frame, result.hand_landmarks[0])
            feats = landmarks_to_features(result.hand_world_landmarks[0],
                                          result.handedness[0][0].category_name)

        if self.mode.get() == "collect":
            self.handle_collect(frame, feats)
        else:
            self.handle_infer(feats)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=img)
        self.video.image = img

    def handle_collect(self, frame, feats):
        label = self.selected_label()
        if self.recording:
            cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)
            if feats is not None and time.monotonic() - self.last_saved >= SAVE_INTERVAL:
                append_row(label, feats)
                self.counts[label] += 1
                self.tree.set(label, "count", self.counts[label])
                self.last_saved = time.monotonic()
        hand = "손 감지됨" if feats is not None else "손이 보이지 않음"
        rec = "녹화 중" if self.recording else "대기"
        self.status.config(text=f"[수집] {label or '-'}  ·  {rec}  ·  {hand}")

    def handle_infer(self, feats):
        if self.clf is None:
            self.result_label.config(text="모델 없음")
            self.status.config(text="[인식] 먼저 훈련하세요")
            return
        if feats is None:
            self.result_label.config(text="-")
            for bar in self.prob_bars.values():
                bar["value"] = 0
            self.status.config(text="[인식] 손이 보이지 않음")
            return
        proba = self.clf.predict_proba([feats])[0]
        for name, p in zip(self.clf.classes_, proba):
            self.prob_bars[str(name)]["value"] = p
        best = proba.argmax()
        name = str(self.clf.classes_[best]) if proba[best] >= self.min_score.get() else "Unknown"
        self.result_label.config(text=name)
        self.status.config(text=f"[인식] {name}  ({proba[best]:.2f})")

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.recognizer.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    GestureStudio(root)
    root.mainloop()
