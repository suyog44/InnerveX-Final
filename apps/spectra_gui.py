#!/usr/bin/env python3
import time
import json
from pathlib import Path
import tkinter as tk

import numpy as np
import cv2
import yaml
from PIL import Image, ImageTk

from picamera2 import Picamera2
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board, busio, adafruit_mlx90640

# ============================================================
# GLOBAL CONFIG
# ============================================================
TW = TH = 640

THERM_MIN = 20.0
THERM_MAX = 45.0
COLORMAP = cv2.COLORMAP_TURBO

FLIP_V = True
FLIP_H = False

STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

YOLO_CONF = 0.25

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

STEREO_YAML = DATA / "stereo_alignment_matrix.yaml"
MLX_JSON = DATA / "mlx_manual_align.json"

# ============================================================
# HOMOGRAPHY (NO INVERSION)
# ============================================================
def load_homography():
    if not STEREO_YAML.exists():
        print("[WARN] Missing homography YAML")
        return np.eye(3, dtype=np.float32)

    with open(STEREO_YAML, "r") as f:
        y = yaml.safe_load(f)

    H = np.array(y["homography_secondary_to_master"], dtype=np.float32)
    H /= H[2, 2]
    print("[INFO] Loaded stereo homography OK")
    return H

# ============================================================
# CAMERA – FORCE BOTH TO 1280×720 SENSOR GEOMETRY
# ============================================================
class PiCam:
    def __init__(self, cam_index, force_720p=False):
        self.picam = Picamera2(camera_num=cam_index)

        modes = self.picam.sensor_modes

        if force_720p:
            # ?? FORCE SENSOR MODE CLOSEST TO 1280×720
            mode = min(
                modes,
                key=lambda m: abs(m["size"][0] - 1280) + abs(m["size"][1] - 720)
            )
            print(f"[INFO] Forcing sensor mode {mode['size']} for IMX500")
            crop = (0, 0, mode["size"][0], mode["size"][1])
        else:
            # IMX519 already locked to 720p
            mode = modes[0]
            crop = None

        cfg = self.picam.create_preview_configuration(
            main={"size": (TW, TH), "format": "XBGR8888"},
            controls={"ScalerCrop": crop} if crop else {},
            buffer_count=8
        )

        try:
            cfg.enable_raw(False)
        except Exception:
            pass

        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.4)

        print("[INFO] Camera config:", self.picam.camera_configuration())

    def read(self):
        frame = self.picam.capture_array()
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def close(self):
        self.picam.stop()

def pick_cameras():
    info = Picamera2.global_camera_info()
    rgb = ir = None
    for i, d in enumerate(info):
        s = str(d).lower()
        if "imx500" in s:
            rgb = i
        if "imx519" in s:
            ir = i
    return rgb or 0, ir or 1

# ============================================================
# THERMAL (UNCHANGED, CORRECT)
# ============================================================
class ThermalMLX:
    def __init__(self):
        self.dx = 0
        self.dy = 0
        self.scale = 1.0

        if MLX_JSON.exists():
            d = json.load(open(MLX_JSON))
            self.dx = d["dx"]
            self.dy = d["dy"]
            self.scale = d["scale"]

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

    def read(self):
        self.mlx.getFrame(self.buf)
        raw = np.array(self.buf).reshape(24, 32)

        if FLIP_V:
            raw = raw[::-1]
        if FLIP_H:
            raw = raw[:, ::-1]

        raw = np.clip(raw, THERM_MIN, THERM_MAX)
        u8 = ((raw - THERM_MIN) / (THERM_MAX - THERM_MIN) * 255).astype(np.uint8)

        up = cv2.resize(u8, (TW, TH), interpolation=cv2.INTER_CUBIC)
        bgr = cv2.applyColorMap(up, COLORMAP)

        cx, cy = TW / 2, TH / 2
        s = self.scale
        M = np.array([
            [s, 0, (1 - s) * cx + self.dx],
            [0, s, (1 - s) * cy + self.dy]
        ], dtype=np.float32)

        return cv2.warpAffine(bgr, M, (TW, TH))

# ============================================================
# GUI
# ============================================================
class SpectraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPECTRA : HYPERSPECTRAL SURVEILLANCE")

        self.H = load_homography()

        rgb_idx, ir_idx = pick_cameras()

        # ?? IMX500 forced to 720p geometry
        self.cam_rgb = PiCam(rgb_idx, force_720p=True)

        # IMX519 untouched (already 720p)
        self.cam_ir = PiCam(ir_idx, force_720p=False)

        self.thermal = ThermalMLX()
        self.detector = YOLO("yolov8n.pt")

        self.lbl_rgb = tk.Label(self)
        self.lbl_ir = tk.Label(self)
        self.lbl_th = tk.Label(self)

        self.lbl_rgb.pack()
        self.lbl_ir.pack()
        self.lbl_th.pack()

        self.after(10, self.loop)

    def show(self, lbl, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl.img = im
        lbl.configure(image=im)

    def loop(self):
        rgb = self.cam_rgb.read()    # IMX500 (reference)
        ir  = self.cam_ir.read()     # IMX519

        ir_warp = cv2.warpPerspective(
            ir, self.H, (TW, TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        th = self.thermal.read()

        dets = self.detector(rgb, conf=YOLO_CONF, verbose=False)[0]

        for b in dets.boxes or []:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            for img in (rgb, ir_warp, th):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.show(self.lbl_rgb, rgb)
        self.show(self.lbl_ir, ir_warp)
        self.show(self.lbl_th, th)

        self.after(10, self.loop)

    def destroy(self):
        self.cam_rgb.close()
        self.cam_ir.close()
        super().destroy()

if __name__ == "__main__":
    SpectraApp().mainloop()
