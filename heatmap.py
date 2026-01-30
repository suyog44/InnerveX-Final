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
# PATHS (use ./data folder next to this script)
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

YAML_PATHS = [
    os.path.join(DATA_DIR, "stereo_alignment_matrix.yaml"),
    os.path.join(DATA_DIR, "stereo_alignment_matrix.yaml.bak"),
]
MLX_ALIGN_FILE = os.path.join(DATA_DIR, "mlx_manual_align.json")

# -----------------------------
# CONFIG
# -----------------------------
MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

THERM_MIN = 20.0
THERM_MAX = 45.0

OVERLAY_ALPHA = 0.45

FLIP_V = True
FLIP_H = False

t_dx, t_dy = 0, 0
t_scale = 1.0

STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

# Fusion weights (renormalized automatically)
W_IMX519 = 0.45
W_MLX    = 0.35
W_IMX500 = 0.20

# IMX500 "black frame" detection thresholds
BLACK_MEAN_THR = 0.05
BLACK_PX_THR   = 0.06
BLACK_FRAC_THR = 0.98
BLACK_BORDER_CROP = 0.08


# -----------------------------
# YAML load (OPTIONAL)
# -----------------------------
def load_stereo_yaml_optional(default_wh=(640, 640)):
    for p in YAML_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = yaml.safe_load(f)
            H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
            target_shape = data.get("target_shape", [default_wh[0], default_wh[1]])  # [W,H]
            W = int(target_shape[0]); Hh = int(target_shape[1])
            print(f"[INFO] Loaded stereo YAML: {p}  target={W}x{Hh}")
            return H, (W, Hh)

    print(f"[WARN] YAML not found in {YAML_PATHS}. Using identity homography (no warp).")
    H = np.eye(3, dtype=np.float32)
    return H, default_wh


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

    # best-effort raw disable
    try:
        cfg.enable_raw(False)
    except Exception:
        try:
            cfg["raw"] = None
        except Exception:
            pass

    cam.configure(cfg)
    cam.start()
    time.sleep(0.3)
    return cam

def capture_bgr(cam):
    frame = cam.capture_array()
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# -----------------------------
# Thermal helpers
# -----------------------------
def norm_thermal(t, lo, hi):
    t = np.clip(t, lo, hi)
    return ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)

def apply_thermal_transform(img, dx, dy, scale):
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
# Green (cold) -> Yellow -> Red (hot) LUT
# -----------------------------
def make_green_to_red_lut():
    """
    LUT in BGR for OpenCV (shape 256x1x3):
    0   -> GREEN  (0,255,0)
    128 -> YELLOW (0,255,255)
    255 -> RED    (0,0,255)
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for v in range(256):
        if v <= 128:
            r = int((v / 128.0) * 255.0)   # 0..255
            g = 255
            b = 0
        else:
            r = 255
            g = int((1.0 - (v - 128.0) / 127.0) * 255.0)  # 255..0
            b = 0
        lut[v, 0] = (b, g, r)
    return lut

GR_LUT = make_green_to_red_lut()

def heatmap_green_red(gray_u8):
    # OpenCV supports user colormap LUT via applyColorMap(userColor) [1](https://qualcomm-confluence.atlassian.net/wiki/spaces/LinAud/pages/573729844/SPv5.1+2+in+1+speaker+support+MIPS+optimizations)[2](https://qualcomm-confluence.atlassian.net/wiki/spaces/Nissan/pages/2400762548/2025-07-28+Camera+configuration+alignment+for+Wing+cameras)
    return cv2.applyColorMap(gray_u8, GR_LUT)


# -----------------------------
# MLX align persistence
# -----------------------------
def load_mlx_alignment():
    global t_dx, t_dy, t_scale
    try:
        with open(MLX_ALIGN_FILE, "r") as f:
            d = json.load(f)
        t_dx = int(d.get("dx", 0))
        t_dy = int(d.get("dy", 0))
        t_scale = float(d.get("scale", 1.0))
        print("[INFO] Loaded MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})
    except FileNotFoundError:
        print(f"[INFO] No saved MLX alignment found at {MLX_ALIGN_FILE}. Using defaults.")

def save_mlx_alignment():
    os.makedirs(os.path.dirname(MLX_ALIGN_FILE), exist_ok=True)
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump({"dx": t_dx, "dy": t_dy, "scale": t_scale}, f)
    print("[INFO] Saved MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})


# -----------------------------
# Fusion helpers
# -----------------------------
def to_gray01(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.clip(g, 0.0, 1.0)

def robust_norm01(x, p_lo=2.0, p_hi=98.0, eps=1e-6):
    x = x.astype(np.float32)
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)

def crop_border(x, frac):
    if frac <= 0:
        return x
    h, w = x.shape[:2]
    dh = int(h * frac)
    dw = int(w * frac)
    return x[dh:h-dh, dw:w-dw]

def imx500_is_black(sec_bgr):
    g = to_gray01(sec_bgr)
    g = crop_border(g, BLACK_BORDER_CROP)
    mean_val = float(g.mean())
    frac_dark = float((g < BLACK_PX_THR).mean())
    is_black = (mean_val < BLACK_MEAN_THR) and (frac_dark > BLACK_FRAC_THR)
    return is_black, mean_val, frac_dark

def overlay(base_bgr, heat_bgr, alpha):
    heat_bgr = cv2.resize(heat_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(heat_bgr, alpha, base_bgr, 1.0 - alpha, 0)

def put_status_bar(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


# -----------------------------
# MAIN
# -----------------------------
H_sec_to_master, TARGET_SIZE_WH = load_stereo_yaml_optional(default_wh=(640, 640))
TW, TH = TARGET_SIZE_WH

info = list_cameras()
idx_master = pick_camera_index(info, MASTER_NAME)
idx_secondary = pick_camera_index(info, SECONDARY_NAME)

if idx_master is None or idx_secondary is None:
    print("\n[WARN] Could not auto-detect IMX519/IMX500 by name. Falling back to 0=master, 1=secondary")
    idx_master = 0
    idx_secondary = 1

print(f"\n[INFO] MASTER (IMX519) index={idx_master}")
print(f"[INFO] SECONDARY (IMX500) index={idx_secondary}")

cam_master = open_camera(idx_master, TARGET_SIZE_WH)
cam_secondary = open_camera(idx_secondary, TARGET_SIZE_WH)

print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768

load_mlx_alignment()

print("\n[CONTROLS]")
print("  H/K/I/J : Move MLX left/right/up/down")
print("  + / -   : Zoom MLX in/out")
print("  0       : Reset MLX transform")
print("  p       : Save MLX transform")
print("  q       : Quit\n")

WIN = "FUSED ONLY (LOW=GREEN, HIGH=RED)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, TW, TH)

fps = 0.0

try:
    while True:
        t0 = time.time()

        master = capture_bgr(cam_master)
        secondary = capture_bgr(cam_secondary)

        master = cv2.resize(master, (TW, TH), interpolation=cv2.INTER_LINEAR)
        secondary = cv2.resize(secondary, (TW, TH), interpolation=cv2.INTER_LINEAR)

        # warp IMX500 -> IMX519 using stereo YAML (identity if missing)
        warped_secondary = cv2.warpPerspective(
            secondary, H_sec_to_master, (TW, TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # MLX frame
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

        # MLX -> uint8 map (0..255) based on THERM_MIN/MAX
        u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
        up = cv2.resize(u8, (TW, TH), interpolation=cv2.INTER_CUBIC)

        # apply MLX manual alignment on scalar map (mlx JSON)
        up_moved = apply_thermal_transform(up, t_dx, t_dy, t_scale)

        # IMX500 black check
        sec_black, sec_mean, sec_dark = imx500_is_black(secondary)
        use_imx500 = not sec_black

        # Build fusion maps (all normalized to 0..1)
        m01   = robust_norm01(to_gray01(master))
        mlx01 = robust_norm01(up_moved.astype(np.float32) / 255.0)
        s01   = robust_norm01(to_gray01(warped_secondary))

        # weights (drop IMX500 if black)
        w519, wmlx, w500 = W_IMX519, W_MLX, (W_IMX500 if use_imx500 else 0.0)
        wsum = max(1e-6, w519 + wmlx + w500)
        w519, wmlx, w500 = w519/wsum, wmlx/wsum, w500/wsum

        fused01 = (w519 * m01) + (wmlx * mlx01) + (w500 * s01)
        fused01 = robust_norm01(fused01, p_lo=1.0, p_hi=99.0)
        fused_u8 = (fused01 * 255.0).astype(np.uint8)

        # GREEN->RED heatmap and overlay on IMX519 master
        fused_color = heatmap_green_red(fused_u8)
        fused_overlay = overlay(master, fused_color, OVERLAY_ALPHA)

        mode_txt = "IMX519+MLX" if not use_imx500 else "IMX500+IMX519+MLX"

        # FPS
        fps_inst = 1.0 / max(1e-6, (time.time() - t0))
        fps = fps * 0.8 + fps_inst * 0.2

        status = (
            f"FUSED {mode_txt} | FPS {fps:.1f} | "
            f"IMX500 mean={sec_mean:.3f} dark={sec_dark:.3f} | "
            f"MLX dx={t_dx} dy={t_dy} s={t_scale:.2f} | "
            f"LOW=GREEN HIGH=RED"
        )
        fused_overlay = put_status_bar(fused_overlay, status)

        cv2.imshow(WIN, fused_overlay)

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
            print("[INFO] Thermal transform reset.")
        elif key in (ord('p'), ord('P')):
            save_mlx_alignment()

finally:
    cam_master.stop()
    cam_secondary.stop()
    cam_master.close()
    cam_secondary.close()
    cv2.destroyAllWindows()
