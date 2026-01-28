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
MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

THERM_MIN = 20.0
THERM_MAX = 45.0
COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

MLX_ALIGN_FILE = "mlx_manual_align.json"
t_dx, t_dy = 0, 0
t_scale = 1.0

STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5


# -----------------------------
# YAML load
# -----------------------------
def load_stereo_yaml():
    for p in YAML_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = yaml.safe_load(f)
            H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
            target_shape = data.get("target_shape", [640, 640])  # [W,H]
            W = int(target_shape[0]); Hh = int(target_shape[1])
            print(f"[INFO] Loaded stereo YAML: {p}")
            return H, (W, Hh)
    raise FileNotFoundError(f"None of these YAML files found: {YAML_PATHS}")


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
    """
    Key change:
    - Use XBGR8888 (known-good for IMX500 examples) instead of RGB888. [1](https://forums.raspberrypi.com/viewtopic.php?t=394713)
    - Disable raw stream explicitly where possible. [2](https://github.com/aswinzz/Image-Registration)
    """
    cam = Picamera2(camera_num=cam_index)

    # Use preview configuration (faster) with XBGR8888
    cfg = cam.create_preview_configuration(main={"size": size_wh, "format": "XBGR8888"}, buffer_count=8)

    # Try to disable raw stream (best effort)
    try:
        cfg.enable_raw(False)  # supported by CameraConfiguration [2](https://github.com/aswinzz/Image-Registration)
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
    # XBGR8888 usually gives 4 channels; convert to BGR
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

def apply_thermal_transform(img_bgr, dx, dy, scale):
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale, 0.0, (1 - scale) * cx + dx],
        [0.0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)
    return cv2.warpAffine(img_bgr, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


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
        print("[INFO] No saved MLX alignment found.")

def save_mlx_alignment():
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump({"dx": t_dx, "dy": t_dy, "scale": t_scale}, f)
    print("[INFO] Saved MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})


# -----------------------------
# MAIN
# -----------------------------
H_sec_to_master, TARGET_SIZE_WH = load_stereo_yaml()
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

# Open cameras (this is where you were failing)
cam_master = open_camera(idx_master, TARGET_SIZE_WH)
cam_secondary = open_camera(idx_secondary, TARGET_SIZE_WH)

print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768

load_mlx_alignment()

print("\n[CONTROLS]")
print("  H : Move MLX left")
print("  K : Move MLX right")
print("  I : Move MLX up")
print("  J : Move MLX down")
print("  + / = : Zoom IN MLX")
print("  - / _ : Zoom OUT MLX")
print("  0 : Reset MLX transform")
print("  p : Save MLX transform")
print("  q : Quit\n")

win = "3-View: IMX500->IMX519 | IMX519 | MLX->IMX519"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 3 * TW, TH)

fps = 0.0

try:
    while True:
        t0 = time.time()

        master = capture_bgr(cam_master)
        secondary = capture_bgr(cam_secondary)

        master = cv2.resize(master, (TW, TH), interpolation=cv2.INTER_LINEAR)
        secondary = cv2.resize(secondary, (TW, TH), interpolation=cv2.INTER_LINEAR)

        warped_secondary = cv2.warpPerspective(secondary, H_sec_to_master, (TW, TH),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))

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

        u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
        up = cv2.resize(u8, (TW, TH), interpolation=cv2.INTER_CUBIC)
        thermal = apply_colormap(up, COLORMAP)
        thermal_moved = apply_thermal_transform(thermal, t_dx, t_dy, t_scale)

        def label(img, text):
            out = img.copy()
            cv2.rectangle(out, (0, 0), (TW, 34), (0, 0, 0), -1)
            cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return out

        panel1 = label(warped_secondary, "IMX500 mapped -> IMX519")
        panel2 = label(master, "IMX519 (master)")
        panel3 = label(thermal_moved, f"MLX->IMX519 (dx={t_dx}, dy={t_dy}, s={t_scale:.2f})")

        combined = cv2.hconcat([panel1, panel2, panel3])

        fps_inst = 1.0 / max(1e-6, (time.time() - t0))
        fps = fps * 0.8 + fps_inst * 0.2
        cv2.putText(combined, f"FPS {fps:.1f}", (10, TH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)

        cv2.imshow(win, combined)

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
