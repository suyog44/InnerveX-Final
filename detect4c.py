#!/usr/bin/env python3
import os
import time
import json
import cv2
import numpy as np
import yaml
import warnings
from picamera2 import Picamera2

warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board
import busio
import adafruit_mlx90640

# -----------------------------
# CONFIG
# -----------------------------
YAML_PATHS = ["stereo_alignment_matrix.yaml", "stereo_alignment_matrix.yaml.bak"]
MLX_ALIGN_FILE = "mlx_manual_align.json"

MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

# MLX settings
THERM_MIN = 20.0
THERM_MAX = 45.0
THERM_COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

# Display scaling (tile size will be derived from target)
SCALE_DISPLAY = 1.0   # set 0.75 if window is too big

# Optional runtime controls for MLX mapping
STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

# -----------------------------
# Load stereo YAML mapping
# -----------------------------
def load_stereo_yaml_optional(default_wh=(640, 480)):
    """
    Loads:
      - homography_secondary_to_master
      - target_shape [W,H]
    Falls back to identity if not found.
    """
    for p in YAML_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = yaml.safe_load(f)
            H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
            target_shape = data.get("target_shape", [default_wh[0], default_wh[1]])  # [W,H]
            TW, TH = int(target_shape[0]), int(target_shape[1])
            print(f"[INFO] Loaded stereo YAML: {p} | target={TW}x{TH}")
            return H, (TW, TH)

    print(f"[WARN] Stereo YAML not found: {YAML_PATHS}. Using identity mapping.")
    return np.eye(3, dtype=np.float32), default_wh


# -----------------------------
# MLX alignment JSON mapping
# -----------------------------
def load_mlx_alignment_optional():
    """
    Loads dx/dy/scale from mlx_manual_align.json.
    Falls back to dx=0,dy=0,scale=1.
    """
    dx, dy, sc = 0, 0, 1.0
    if os.path.exists(MLX_ALIGN_FILE):
        try:
            with open(MLX_ALIGN_FILE, "r") as f:
                d = json.load(f)
            dx = int(d.get("dx", 0))
            dy = int(d.get("dy", 0))
            sc = float(d.get("scale", 1.0))
            print("[INFO] Loaded MLX alignment:", {"dx": dx, "dy": dy, "scale": sc})
        except Exception as e:
            print("[WARN] Failed reading MLX alignment JSON, using defaults:", e)
    else:
        print("[WARN] MLX alignment JSON not found. Using defaults dx=0 dy=0 scale=1.0")

    return dx, dy, sc

def save_mlx_alignment(dx, dy, sc):
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump({"dx": dx, "dy": dy, "scale": sc}, f)
    print("[INFO] Saved MLX alignment:", {"dx": dx, "dy": dy, "scale": sc})


# -----------------------------
# Picamera2 helpers
# -----------------------------
def list_cameras():
    info = Picamera2.global_camera_info()
    print("\nDetected cameras:")
    for i, cam in enumerate(info):
        print(f"  index={i} -> {cam}")
    return info

def pick_camera_index(info, needle):
    needle = needle.lower()
    for i, cam in enumerate(info):
        if needle in str(cam).lower():
            return i
    return None

def open_camera(cam_index, size_wh):
    cam = Picamera2(camera_num=cam_index)
    cfg = cam.create_preview_configuration(main={"size": size_wh, "format": "XBGR8888"}, buffer_count=8)

    # best-effort disable raw stream
    try:
        cfg.enable_raw(False)
    except Exception:
        try:
            cfg["raw"] = None
        except Exception:
            pass

    cam.configure(cfg)
    cam.start()
    time.sleep(0.25)
    return cam

def capture_bgr(cam):
    frame = cam.capture_array()
    # XBGR8888 -> BGRA -> BGR
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# -----------------------------
# Thermal helpers
# -----------------------------
def norm_thermal(t, lo, hi):
    t = np.clip(t, lo, hi)
    return ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)

def apply_colormap(gray_u8, cmap_name):
    cmap_id = getattr(cv2, cmap_name)
    return cv2.applyColorMap(gray_u8, cmap_id)

def apply_thermal_transform(img, dx, dy, scale):
    """
    WarpAffine using dx/dy/scale.
    Works on grayscale or BGR.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale, 0.0, (1 - scale) * cx + dx],
        [0.0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)
    border_val = 0 if img.ndim == 2 else (0, 0, 0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_val)

# -----------------------------
# Visualization helpers
# -----------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

def label_tile(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 22), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out

def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def channel_color_view(bgr_img, which):
    """
    Show only one channel as a colored image.
    which: 'r' / 'g' / 'b'
    Input is BGR.
    """
    b, g, r = cv2.split(bgr_img)
    z = np.zeros_like(b)
    if which == 'r':
        return cv2.merge([z, z, r])   # red-only image (B=0, G=0, R=r)
    if which == 'g':
        return cv2.merge([z, g, z])   # green-only
    if which == 'b':
        return cv2.merge([b, z, z])   # blue-only
    return bgr_img

# -----------------------------
# MAIN
# -----------------------------
# Load mappings
H_sec_to_master, (TW, TH) = load_stereo_yaml_optional(default_wh=(640, 480))
t_dx, t_dy, t_scale = load_mlx_alignment_optional()

# Detect cameras
info = list_cameras()
idx_master = pick_camera_index(info, MASTER_NAME)
idx_secondary = pick_camera_index(info, SECONDARY_NAME)

if idx_master is None or idx_secondary is None:
    print("[WARN] Could not auto-detect IMX519/IMX500 by name. Falling back to 0=IMX519, 1=IMX500")
    idx_master, idx_secondary = 0, 1

print(f"[INFO] IMX519 index={idx_master}")
print(f"[INFO] IMX500 index={idx_secondary}")

# Start cameras at target size (mapping-friendly)
cam_master = open_camera(idx_master, (TW, TH))
cam_secondary = open_camera(idx_secondary, (TW, TH))

# Start MLX
print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768

# Window
WIN = "detect4c (mapped): IMX500 RGB(channels) | IMX519 gray | MLX thermal"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

tile_w = int(TW * SCALE_DISPLAY)
tile_h = int(TH * SCALE_DISPLAY)
cv2.resizeWindow(WIN, 3 * tile_w, 2 * tile_h)

print("\n[CONTROLS]")
print("  q : Quit")
print("  h/k/i/j : move MLX left/right/up/down")
print("  + / -   : zoom MLX in/out")
print("  0       : reset MLX transform")
print("  p       : save MLX transform\n")

fps = 0.0

try:
    while True:
        t0 = time.time()

        # Capture
        master = capture_bgr(cam_master)       # IMX519
        secondary = capture_bgr(cam_secondary) # IMX500

        # Warp IMX500 -> IMX519 using YAML homography
        warped_secondary = cv2.warpPerspective(
            secondary, H_sec_to_master, (TW, TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Build colored channel tiles from *mapped* IMX500
        red_only   = channel_color_view(warped_secondary, 'r')
        green_only = channel_color_view(warped_secondary, 'g')
        blue_only  = channel_color_view(warped_secondary, 'b')

        # IMX519 grayscale
        master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
        master_gray_bgr = ensure_bgr(master_gray)

        # MLX thermal -> upscale to target -> apply dx/dy/scale mapping from JSON
        try:
            mlx.getFrame(buf)
        except Exception as e:
            print("[WARN] MLX read error:", e)
            continue

        raw = np.array(buf, dtype=np.float32).reshape(24, 32)
        if FLIP_V:
            raw = np.flip(raw, axis=0)
        if FLIP_H:
            raw = np.flip(raw, axis=1)

        therm_u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
        therm_up = cv2.resize(therm_u8, (TW, TH), interpolation=cv2.INTER_CUBIC)
        therm_up = apply_thermal_transform(therm_up, t_dx, t_dy, t_scale)
        therm_color = apply_colormap(therm_up, THERM_COLORMAP)

        # Resize to display tiles
        r_t = cv2.resize(red_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        g_t = cv2.resize(green_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        b_t = cv2.resize(blue_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        y_t = cv2.resize(master_gray_bgr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        t_t = cv2.resize(therm_color, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

        # One blank tile (2x3 grid has 6 slots, we show 5)
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        # Label tiles
        r_t = label_tile(r_t, "IMX500 mapped -> IMX519 : RED only")
        g_t = label_tile(g_t, "IMX500 mapped -> IMX519 : GREEN only")
        b_t = label_tile(b_t, "IMX500 mapped -> IMX519 : BLUE only")
        y_t = label_tile(y_t, "IMX519 : GRAYSCALE")
        t_t = label_tile(t_t, f"MLX : THERMAL (dx={t_dx}, dy={t_dy}, s={t_scale:.2f})")
        blank = label_tile(blank, "EMPTY")

        # Compose single window
        top = cv2.hconcat([r_t, g_t, b_t])
        bot = cv2.hconcat([y_t, t_t, blank])
        canvas = cv2.vconcat([top, bot])

        # FPS
        fps_inst = 1.0 / max(1e-6, (time.time() - t0))
        fps = 0.8 * fps + 0.2 * fps_inst
        cv2.putText(canvas, f"FPS {fps:.1f}", (10, 2 * tile_h - 10),
                    FONT, 0.7, (40, 255, 40), 2, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('h'), ord('H')):
            t_dx -= STEP
        elif key in (ord('k'), ord('K')):
            t_dx += STEP
        elif key in (ord('i'), ord('I')):
            t_dy -= STEP
        elif key in (ord('j'), ord('J')):
            t_dy += STEP
        elif key in (ord('+'), ord('=')):
            t_scale = min(SCALE_MAX, t_scale + ZOOM_STEP)
        elif key in (ord('-'), ord('_')):
            t_scale = max(SCALE_MIN, t_scale - ZOOM_STEP)
        elif key == ord('0'):
            t_dx, t_dy, t_scale = 0, 0, 1.0
            print("[INFO] MLX transform reset.")
        elif key in (ord('p'), ord('P')):
            save_mlx_alignment(t_dx, t_dy, t_scale)

finally:
    cam_master.stop()
    cam_secondary.stop()
    cam_master.close()
    cam_secondary.close()
    cv2.destroyAllWindows()
