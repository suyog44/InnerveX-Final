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

# MLX90640
warnings.filterwarnings("ignore", category=RuntimeWarning)
import board
import busio
import adafruit_mlx90640

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
THERM_PERSON_DELTA = 3.0  # deg C above ambient

COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

# Thermal alignment relative to IMX519 (manual)
MLX_ALIGN_FILE = "mlx_manual_align.json"
t_dx, t_dy, t_scale = 0, 0, 1.0
MOVE_STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

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
    print("\nDetected cameras:")
    for i, cam in enumerate(info):
        print(f"[CAM] {i} -> {cam}")
    return info

def pick_camera_num(info, model_name):
    model_name = model_name.lower()
    for cam in info:
        if model_name in str(cam).lower():
            return cam.get("Num", None)
    return None

def build_config(cam, size_wh):
    """
    Use preview configuration with XBGR8888 for stability.
    Disable raw if supported by config API. [3](https://docs.cirkitdesigner.com/component/ba38c125-bec5-8a89-cdb6-cd2d7e16a41f/adafruit-mlx90640-thermal-camera)
    """
    cfg = cam.create_preview_configuration(
        main={"size": size_wh, "format": "XBGR8888"},
        buffer_count=6
    )
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

def apply_colormap(gray_u8):
    return cv2.applyColorMap(gray_u8, getattr(cv2, COLORMAP))

def warp_gray_with_same_thermal_transform(gray_u8, dx, dy, scale):
    """
    Apply the same affine mapping used for thermal visualization to the grayscale.
    This ensures ROI temperature computation matches the displayed thermal panel.
    """
    h, w = gray_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale, 0.0, (1 - scale) * cx + dx],
        [0.0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)
    return cv2.warpAffine(
        gray_u8, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# =========================================================
# DRAW HELPERS
# =========================================================
def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def draw_annots(img, annots):
    for a in annots:
        x1, y1, x2, y2 = a["box"]
        color = a.get("color", (0, 255, 0))
        label = a.get("label", "")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def add_title(img, title, W):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out

# =========================================================
# MLX ALIGN LOAD / SAVE
# =========================================================
def load_mlx_alignment():
    global t_dx, t_dy, t_scale
    if os.path.exists(MLX_ALIGN_FILE):
        with open(MLX_ALIGN_FILE, "r") as f:
            d = json.load(f)
        t_dx = int(d.get("dx", 0))
        t_dy = int(d.get("dy", 0))
        t_scale = float(d.get("scale", 1.0))
        print("[INFO] Loaded MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})

def save_mlx_alignment():
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump({"dx": t_dx, "dy": t_dy, "scale": t_scale}, f)
    print("[INFO] Saved MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})

# =========================================================
# INIT
# =========================================================
H_sec_to_master, (TW, TH) = load_stereo_yaml()

info = list_cameras()
idx_master = pick_camera_num(info, MASTER_NAME)
idx_secondary = pick_camera_num(info, SECONDARY_NAME)

# Fallback if needed (based on your logs often: imx500 Num=0, imx519 Num=1)
if idx_master is None:
    idx_master = 1
if idx_secondary is None:
    idx_secondary = 0

print(f"[INFO] Using MASTER Num={idx_master} ({MASTER_NAME})")
print(f"[INFO] Using SECONDARY Num={idx_secondary} ({SECONDARY_NAME})")

# ✅ Important multi-cam pattern:
# Create BOTH Picamera2 objects first, THEN configure both, THEN start both. [4](https://docs.opencv.org/master/d9/dab/tutorial_homography.html)[5](https://github.com/Tarpit59/thermal-rgb-aligner)
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
win = "Multispectral Fusion (IMX519 master) - 3 Panels"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, TW * 3, TH)

print("\n[CONTROLS]")
print("  H / K / I / J : Move thermal left/right/up/down")
print("  + / -         : Zoom thermal in/out")
print("  p             : Save thermal alignment")
print("  q             : Quit\n")

try:
    while True:
        t0 = time.time()

        # --- Capture master & secondary ---
        master = capture_bgr(cam_master)
        secondary = capture_bgr(cam_secondary)

        master = cv2.resize(master, (TW, TH))
        secondary = cv2.resize(secondary, (TW, TH))

        # --- Map IMX500 -> IMX519 coordinate system ---
        warped_secondary = cv2.warpPerspective(
            secondary, H_sec_to_master, (TW, TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # --- YOLO inference ONLY on IMX519 ---
        results = yolo(master, conf=YOLO_CONF_TH, verbose=False)[0]

        # --- Thermal read ---
        try:
            mlx.getFrame(mlx_buf)
        except Exception as e:
            print("[WARN] MLX read error:", e)
            continue

        raw = np.array(mlx_buf, dtype=np.float32).reshape(24, 32)
        if FLIP_V:
            raw = np.flip(raw, axis=0)
        if FLIP_H:
            raw = np.flip(raw, axis=1)

        ambient = float(np.mean(raw))

        # Convert MLX raw-> gray-> upsample to IMX519 size
        gray = norm_thermal(raw, THERM_MIN, THERM_MAX)
        gray_up = cv2.resize(gray, (TW, TH), interpolation=cv2.INTER_CUBIC)

        # ✅ Align thermal grayscale into IMX519 coordinate system
        gray_aligned = warp_gray_with_same_thermal_transform(gray_up, t_dx, t_dy, t_scale)
        thermal_vis = apply_colormap(gray_aligned)

        # --- Build annotations ONCE (IMX519 coords) ---
        annots = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, TW, TH)

            # ROI temperature from the aligned thermal grayscale
            roi = gray_aligned[y1:y2, x1:x2]
            mean_temp = ambient
            if roi.size > 0:
                mean_temp = (np.mean(roi) / 255.0) * (THERM_MAX - THERM_MIN) + THERM_MIN

            fused_conf = conf
            if cls == 0 and mean_temp > ambient + THERM_PERSON_DELTA:
                fused_conf = min(1.0, conf + 0.25)

            label = f"{yolo.names[cls]} {fused_conf:.2f}"

            annots.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "color": (0, 200, 255) if cls == 0 else (0, 255, 0)
            })

        # --- Draw boxes on ALL panels ---
        imx519_vis = master.copy()
        imx500_vis = warped_secondary.copy()

        draw_annots(imx519_vis, annots)
        draw_annots(imx500_vis, annots)
        draw_annots(thermal_vis, annots)

        # Add titles
        imx500_vis = add_title(imx500_vis, "IMX500 mapped -> IMX519", TW)
        imx519_vis = add_title(imx519_vis, "IMX519 master (YOLO)", TW)
        thermal_vis = add_title(thermal_vis, f"MLX90640 mapped (dx={t_dx},dy={t_dy},s={t_scale:.2f})", TW)

        view = cv2.hconcat([imx500_vis, imx519_vis, thermal_vis])

        fps = 1.0 / max(1e-6, time.time() - t0)
        cv2.putText(view, f"FPS {fps:.1f}", (10, TH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)

        cv2.imshow(win, view)

        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('p'), ord('P')):
            save_mlx_alignment()

        # Move thermal (H/K/I/J)
        elif key in (ord('h'), ord('H')):
            t_dx -= MOVE_STEP
        elif key in (ord('k'), ord('K')):
            t_dx += MOVE_STEP
        elif key in (ord('i'), ord('I')):
            t_dy -= MOVE_STEP
        elif key in (ord('j'), ord('J')):
            t_dy += MOVE_STEP

        # Zoom thermal
        elif key in (ord('+'), ord('=')):
            t_scale = min(SCALE_MAX, t_scale + ZOOM_STEP)
        elif key in (ord('-'), ord('_')):
            t_scale = max(SCALE_MIN, t_scale - ZOOM_STEP)

finally:
    print("[INFO] Shutting down...")
    for cam in [cam_master, cam_secondary]:
        try:
            cam.stop()
            cam.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
