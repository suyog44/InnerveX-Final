# -*- coding: utf-8 -*-

import os
import time
import json
import cv2
import numpy as np
import yaml
import warnings

from picamera2 import Picamera2
from ultralytics import YOLO

import board
import busio
import adafruit_mlx90640

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
YAML_PATHS = ["stereo_alignment_matrix.yaml", "stereo_alignment_matrix.yaml.bak"]

MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF_TH = 0.35

THERM_MIN = 20.0
THERM_MAX = 45.0
THERM_PERSON_DELTA = 3.0

COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

MLX_ALIGN_FILE = "mlx_manual_align.json"
t_dx, t_dy, t_scale = 0, 0, 1.0

# =========================================================
# YAML LOAD
# =========================================================
def load_stereo_yaml():
    for p in YAML_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = yaml.safe_load(f)
            H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
            ts = data.get("target_shape", [640, 640])  # [W,H]
            return H, (int(ts[0]), int(ts[1]))
    raise FileNotFoundError("Stereo alignment YAML not found")

# =========================================================
# CAMERA HELPERS
# =========================================================
def list_cameras():
    info = Picamera2.global_camera_info()
    for i, cam in enumerate(info):
        print(f"[CAM] {i} -> {cam}")
    return info

def pick_camera_index(info, needle):
    needle = needle.lower()
    for cam in info:
        if needle in str(cam).lower():
            return cam["Num"]  # <-- use the real camera Num
    return None

def build_config(cam, size):
    # Use preview config for stability (works well for live frames)
    cfg = cam.create_preview_configuration(
        main={"size": size, "format": "XBGR8888"},
        buffer_count=6
    )
    # Disable RAW if supported by your config object [5](https://picamera.readthedocs.io/en/release-1.13/faq.html)
    try:
        cfg.enable_raw(False)
    except Exception:
        pass
    return cfg

def capture_bgr(cam):
    frame = cam.capture_array("main")
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

# =========================================================
# THERMAL HELPERS
# =========================================================
def norm_thermal(t, lo, hi):
    t = np.clip(t, lo, hi)
    return ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)

def apply_colormap(gray):
    return cv2.applyColorMap(gray, getattr(cv2, COLORMAP))

def apply_thermal_transform(img, dx, dy, scale):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = np.array([
        [scale, 0, (1 - scale) * cx + dx],
        [0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h))

# =========================================================
# MLX ALIGN LOAD / SAVE
# =========================================================
def load_mlx_alignment():
    global t_dx, t_dy, t_scale
    if os.path.exists(MLX_ALIGN_FILE):
        with open(MLX_ALIGN_FILE, "r") as f:
            d = json.load(f)
            t_dx = d.get("dx", 0)
            t_dy = d.get("dy", 0)
            t_scale = d.get("scale", 1.0)

def save_mlx_alignment():
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump({"dx": t_dx, "dy": t_dy, "scale": t_scale}, f)
    print("[INFO] Saved MLX alignment")

# =========================================================
# INIT
# =========================================================
H_sec_to_master, (TW, TH) = load_stereo_yaml()

info = list_cameras()
idx_master = pick_camera_index(info, MASTER_NAME)
idx_secondary = pick_camera_index(info, SECONDARY_NAME)

# Fallback, but your print shows: imx500 Num=0, imx519 Num=1
if idx_master is None: idx_master = 1
if idx_secondary is None: idx_secondary = 0

print(f"[INFO] Using MASTER Num={idx_master} ({MASTER_NAME})")
print(f"[INFO] Using SECONDARY Num={idx_secondary} ({SECONDARY_NAME})")

# ✅ IMPORTANT: create BOTH Picamera2 instances first (before starting either)
print("[INFO] Creating camera objects...")
cam_master = Picamera2(camera_num=idx_master)
cam_secondary = Picamera2(camera_num=idx_secondary)

print("[INFO] Configuring cameras...")
cfg_master = build_config(cam_master, (TW, TH))
cfg_secondary = build_config(cam_secondary, (TW, TH))

cam_master.configure(cfg_master)
cam_secondary.configure(cfg_secondary)

print("[INFO] Starting cameras...")
cam_master.start()
cam_secondary.start()
time.sleep(1.0)

# MLX90640
print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_buf = [0.0] * 768

load_mlx_alignment()

# YOLO
print("[INFO] Loading YOLOv8...")
yolo = YOLO(YOLO_MODEL_PATH)

# =========================================================
# MAIN LOOP
# =========================================================
win = "Multispectral Fusion (IMX519 master)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, TW * 3, TH)

try:
    while True:
        t0 = time.time()

        master = capture_bgr(cam_master)
        secondary = capture_bgr(cam_secondary)

        master = cv2.resize(master, (TW, TH))
        secondary = cv2.resize(secondary, (TW, TH))

        warped_secondary = cv2.warpPerspective(secondary, H_sec_to_master, (TW, TH))

        # -------- YOLO (IMX519) --------
        results = yolo(master, conf=YOLO_CONF_TH, verbose=False)[0]

        # -------- THERMAL --------
        mlx.getFrame(mlx_buf)
        raw = np.array(mlx_buf, dtype=np.float32).reshape(24, 32)

        if FLIP_V: raw = np.flip(raw, axis=0)
        if FLIP_H: raw = np.flip(raw, axis=1)

        ambient = float(np.mean(raw))

        gray = norm_thermal(raw, THERM_MIN, THERM_MAX)
        gray_up = cv2.resize(gray, (TW, TH), interpolation=cv2.INTER_CUBIC)
        thermal = apply_colormap(gray_up)
        thermal = apply_thermal_transform(thermal, t_dx, t_dy, t_scale)

        # -------- FUSION --------
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            roi = gray_up[y1:y2, x1:x2]
            mean_temp = ambient
            if roi.size > 0:
                mean_temp = (np.mean(roi) / 255.0) * (THERM_MAX - THERM_MIN) + THERM_MIN

            fused_conf = conf
            if cls == 0 and mean_temp > ambient + THERM_PERSON_DELTA:
                fused_conf = min(1.0, conf + 0.25)

            label = f"{yolo.names[cls]} {fused_conf:.2f}"

            cv2.rectangle(master, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(master, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        view = cv2.hconcat([warped_secondary, master, thermal])

        fps = 1.0 / max(1e-6, time.time() - t0)
        cv2.putText(view, f"FPS {fps:.1f}", (10, TH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)

        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            save_mlx_alignment()

finally:
    print("[INFO] Shutting down...")
    for cam in [cam_master, cam_secondary]:
        try:
            cam.stop()
            cam.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
