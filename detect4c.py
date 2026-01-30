#!/usr/bin/env python3
import os
import time
import json
import glob
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
DATA_DIR = "data"

# Pick newest match if multiple exist (stereo*.yaml, mlx*.json)
YAML_GLOBS = [
    os.path.join(DATA_DIR, "stereo*.yaml"),
    os.path.join(DATA_DIR, "stereo*.yml"),
]
MLX_JSON_GLOBS = [
    os.path.join(DATA_DIR, "mlx*.json"),
]

MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

# MLX settings
THERM_MIN = 20.0
THERM_MAX = 45.0
THERM_COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

# Display scaling
SCALE_DISPLAY = 1.0

# Optional runtime controls for MLX mapping
STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5


def _pick_latest_file(globs_list):
    """Return latest modified file among glob patterns, else None."""
    matches = []
    for g in globs_list:
        matches.extend(glob.glob(g))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


# -----------------------------
# Load stereo YAML mapping
# -----------------------------
def load_stereo_yaml_optional(default_wh=(640, 480)):
    """
    Loads:
      - homography_secondary_to_master (3x3)
      - target_shape [W,H]
    Falls back to identity if not found.
    """
    yaml_path = _pick_latest_file(YAML_GLOBS)
    if not yaml_path:
        print(f"[WARN] Stereo YAML not found in {DATA_DIR} using patterns {YAML_GLOBS}. Using identity mapping.")
        return np.eye(3, dtype=np.float32), default_wh

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Be tolerant to slightly different key names
    H_key_candidates = [
        "homography_secondary_to_master",
        "H_secondary_to_master",
        "H_sec_to_master",
        "homography",
    ]
    H_list = None
    for k in H_key_candidates:
        if k in data:
            H_list = data[k]
            break

    if H_list is None:
        print(f"[WARN] YAML {yaml_path} missing homography key. Using identity mapping.")
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.array(H_list, dtype=np.float32).reshape(3, 3)

    target_shape = data.get("target_shape", [default_wh[0], default_wh[1]])  # [W,H]
    TW, TH = int(target_shape[0]), int(target_shape[1])

    print(f"[INFO] Loaded stereo YAML: {yaml_path} | target={TW}x{TH}")
    return H, (TW, TH)


# -----------------------------
# MLX alignment JSON mapping
# -----------------------------
def load_mlx_alignment_optional():
    """
    Loads dx/dy/scale from latest mlx*.json in data/.
    Falls back to dx=0,dy=0,scale=1.
    """
    dx, dy, sc = 0, 0, 1.0

    mlx_path = _pick_latest_file(MLX_JSON_GLOBS)
    if not mlx_path:
        print(f"[WARN] MLX alignment JSON not found in {DATA_DIR} using patterns {MLX_JSON_GLOBS}. Using defaults.")
        return dx, dy, sc, None

    try:
        with open(mlx_path, "r") as f:
            d = json.load(f) or {}
        dx = int(d.get("dx", 0))
        dy = int(d.get("dy", 0))
        sc = float(d.get("scale", 1.0))
        print("[INFO] Loaded MLX alignment:", {"file": mlx_path, "dx": dx, "dy": dy, "scale": sc})
    except Exception as e:
        print("[WARN] Failed reading MLX alignment JSON, using defaults:", e)
        mlx_path = None

    return dx, dy, sc, mlx_path


def save_mlx_alignment(dx, dy, sc, mlx_path=None):
    """Save to the detected mlx file if available, else default to data/mlx_manual_align.json."""
    if mlx_path is None:
        mlx_path = os.path.join(DATA_DIR, "mlx_manual_align.json")
    os.makedirs(os.path.dirname(mlx_path), exist_ok=True)
    with open(mlx_path, "w") as f:
        json.dump({"dx": dx, "dy": dy, "scale": sc}, f)
    print("[INFO] Saved MLX alignment:", {"file": mlx_path, "dx": dx, "dy": dy, "scale": sc})


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
    cfg = cam.create_preview_configuration(
        main={"size": size_wh, "format": "XBGR8888"},
        buffer_count=8
    )

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
    """WarpAffine using dx/dy/scale."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale, 0.0, (1 - scale) * cx + dx],
        [0.0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)
    border_val = 0 if img.ndim == 2 else (0, 0, 0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )


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
    """Show only one channel as a colored image. Input is BGR."""
    b, g, r = cv2.split(bgr_img)
    z = np.zeros_like(b)
    if which == 'r':
        return cv2.merge([z, z, r])
    if which == 'g':
        return cv2.merge([z, g, z])
    if which == 'b':
        return cv2.merge([b, z, z])
    return bgr_img


# -----------------------------
# MAIN
# -----------------------------
H_sec_to_master, (TW, TH) = load_stereo_yaml_optional(default_wh=(640, 480))
t_dx, t_dy, t_scale, mlx_path = load_mlx_alignment_optional()

info = list_cameras()
idx_master = pick_camera_index(info, MASTER_NAME)
idx_secondary = pick_camera_index(info, SECONDARY_NAME)

if idx_master is None or idx_secondary is None:
    print("[WARN] Could not auto-detect IMX519/IMX500 by name. Falling back to 0=IMX519, 1=IMX500")
    idx_master, idx_secondary = 0, 1

print(f"[INFO] IMX519 index={idx_master}")
print(f"[INFO] IMX500 index={idx_secondary}")

cam_master = open_camera(idx_master, (TW, TH))
cam_secondary = open_camera(idx_secondary, (TW, TH))

print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768  # MLX90640 is 32x24 = 768 pixels

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

        master = capture_bgr(cam_master)       # IMX519
        secondary = capture_bgr(cam_secondary) # IMX500

        warped_secondary = cv2.warpPerspective(
            secondary, H_sec_to_master, (TW, TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        red_only = channel_color_view(warped_secondary, 'r')
        green_only = channel_color_view(warped_secondary, 'g')
        blue_only = channel_color_view(warped_secondary, 'b')

        master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
        master_gray_bgr = ensure_bgr(master_gray)

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

        r_t = cv2.resize(red_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        g_t = cv2.resize(green_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        b_t = cv2.resize(blue_only, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        y_t = cv2.resize(master_gray_bgr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        t_t = cv2.resize(therm_color, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        r_t = label_tile(r_t, "IMX500 mapped -> IMX519 : RED only")
        g_t = label_tile(g_t, "IMX500 mapped -> IMX519 : GREEN only")
        b_t = label_tile(b_t, "IMX500 mapped -> IMX519 : BLUE only")
        y_t = label_tile(y_t, "IMX519 : GRAYSCALE")
        t_t = label_tile(t_t, f"MLX : THERMAL (dx={t_dx}, dy={t_dy}, s={t_scale:.2f})")
        blank = label_tile(blank, "EMPTY")

        top = cv2.hconcat([r_t, g_t, b_t])
        bot = cv2.hconcat([y_t, t_t, blank])
        canvas = cv2.vconcat([top, bot])

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
            save_mlx_alignment(t_dx, t_dy, t_scale, mlx_path)

finally:
    cam_master.stop()
    cam_secondary.stop()
    cam_master.close()
    cam_secondary.close()
    cv2.destroyAllWindows()
