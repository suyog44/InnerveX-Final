#!/usr/bin/env python3
import time
import csv
import json
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np
import cv2
import yaml

from picamera2 import Picamera2

from ultralytics import YOLO

# MLX
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board, busio, adafruit_mlx90640


# =========================================================
# PATHS
# =========================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

STEREO_YAML = DATA_DIR / "stereo_alignment_matrix.yaml"
MLX_JSON = DATA_DIR / "mlx_manual_align.json"
CSV_PATH = DATA_DIR / "detections.csv"

TW, TH = 640, 640


# =========================================================
# UTILS
# =========================================================
def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def ensure_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "label", "confidence",
                 "threat_score", "camouflage",
                 "x1", "y1", "x2", "y2"]
            )

def load_homography():
    if not STEREO_YAML.exists():
        print("[WARN] Missing stereo yaml – using identity")
        return np.eye(3, dtype=np.float32)

    with open(STEREO_YAML) as f:
        H = yaml.safe_load(f)["homography_secondary_to_master"]
    H = np.array(H, np.float32)
    return H / H[2, 2]


# =========================================================
# CAMERA
# =========================================================
class PiCam:
    def __init__(self, idx):
        self.cam = Picamera2(idx)
        cfg = self.cam.create_preview_configuration(
            main={"size": (TW, TH), "format": "XBGR8888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(0.2)

    def read(self):
        frame = self.cam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.cam.stop()


def pick_cameras():
    info = Picamera2.global_camera_info()
    master = secondary = None
    for i, x in enumerate(info):
        s = str(x).lower()
        if "imx519" in s:
            master = i
        if "imx500" in s:
            secondary = i
    if master is None or secondary is None:
        print("[WARN] fallback to 0/1")
        return 0, 1
    return master, secondary


# =========================================================
# THERMAL
# =========================================================
class Thermal:
    def __init__(self):
        self.dx = self.dy = 0
        self.scale = 1.0
        if MLX_JSON.exists():
            d = json.load(open(MLX_JSON))
            self.dx = d.get("dx", 0)
            self.dy = d.get("dy", 0)
            self.scale = d.get("scale", 1.0)

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

    def read(self):
        self.mlx.getFrame(self.buf)
        raw = np.array(self.buf, np.float32).reshape(24, 32)
        u8 = np.clip((raw - 20) / 25 * 255, 0, 255).astype(np.uint8)
        up = cv2.resize(u8, (TW, TH), cv2.INTER_CUBIC)
        bgr = cv2.applyColorMap(up, cv2.COLORMAP_TURBO)
        M = np.array([
            [self.scale, 0, self.dx],
            [0, self.scale, self.dy]
        ], np.float32)
        return cv2.warpAffine(bgr, M, (TW, TH))


# =========================================================
# YOLO
# =========================================================
class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names

    def infer(self, img):
        dets = []
        r = self.model(img, verbose=False)[0]
        if r.boxes is None:
            return dets
        for b in r.boxes:
            dets.append({
                "label": self.names[int(b.cls)],
                "conf": float(b.conf),
                "xyxy": b.xyxy[0].tolist()
            })
        return dets


# =========================================================
# TK CANVAS (VISIBLE!)
# =========================================================
class VideoPanel:
    def __init__(self, parent, title):
        self.frame = ttk.LabelFrame(parent, text=title)
        self.canvas = tk.Label(self.frame)
        self.canvas.pack(fill="both", expand=True)
        self.photo = None

    def grid(self, **k):
        self.frame.grid(**k)

    def update(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(img.resize((420, 420)))
        self.canvas.configure(image=self.photo)


# =========================================================
# MAIN APP (THIS NOW SHOWS!)
# =========================================================
class SpectraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPECTRA – LIVE")
        self.geometry("1300x500")

        self.grid_columnconfigure((0,1,2), weight=1)

        self.p1 = VideoPanel(self, "IMX500 → IMX519")
        self.p2 = VideoPanel(self, "IMX519 MASTER")
        self.p3 = VideoPanel(self, "MLX THERMAL")

        self.p1.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.p2.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.p3.grid(row=0, column=2, padx=6, pady=6, sticky="nsew")

        self.H = load_homography()
        m, s = pick_cameras()
        self.cam_m = PiCam(m)
        self.cam_s = PiCam(s)
        self.thermal = Thermal()
        self.det = Detector()

        ensure_csv()
        self.after(30, self.loop)

    def loop(self):
        master = self.cam_m.read()
        sec = self.cam_s.read()
        sec_w = cv2.warpPerspective(sec, self.H, (TW, TH))
        thermal = self.thermal.read()

        dets = self.det.infer(master)
        for d in dets:
            x1,y1,x2,y2 = map(int, d["xyxy"])
            for img in (master, sec_w, thermal):
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            with open(CSV_PATH,"a",newline="") as f:
                csv.writer(f).writerow(
                    [now_iso(), d["label"], d["conf"], d["conf"], 0, x1,y1,x2,y2]
                )

        self.p1.update(sec_w)
        self.p2.update(master)
        self.p3.update(thermal)

        self.after(30, self.loop)


if __name__ == "__main__":
    SpectraApp().mainloop()
