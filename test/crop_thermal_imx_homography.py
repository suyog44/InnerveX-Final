import cv2
import json
import numpy as np
import time
import warnings
from picamera2 import Picamera2

# Thermal imports
warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board
import busio
import adafruit_mlx90640

# -------- Thermal Config --------
THERM_MIN = 20.0
THERM_MAX = 45.0
THERM_WIDTH = 640
THERM_HEIGHT = 480
COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

# -------- Blend Config --------
ALPHA_THERM = 0.5
ALPHA_RGB = 0.5

# -------- Manual adjust THERMAL relative to IMX519 --------
MLX_ALIGN_FILE = "mlx_manual_align.json"
t_dx, t_dy = 0, 0
t_scale = 1.0

STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

# -------- Thermal crop YAML (OpenCV FileStorage) --------
CROP_YAML = "crop_thermal.yml"
crop_coords_thermal = None
drawing_thermal = False
start_point = None

# For UI preview inside mouse callback
thermal_img_display = np.zeros((THERM_HEIGHT, THERM_WIDTH, 3), dtype=np.uint8)


def norm_thermal(t, lo, hi):
    t = np.clip(t, lo, hi)
    return ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)

def apply_colormap(gray_u8, cmap_name):
    cmap_id = getattr(cv2, cmap_name)
    return cv2.applyColorMap(gray_u8, cmap_id)

def apply_thermal_transform(img_bgr, dx, dy, scale):
    """
    Apply scale around image center + translation to THERMAL image.
    RGB stays fixed.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # x' = s*(x-cx) + cx + dx
    # y' = s*(y-cy) + cy + dy
    M = np.array([
        [scale, 0.0, (1 - scale) * cx + dx],
        [0.0, scale, (1 - scale) * cy + dy]
    ], dtype=np.float32)

    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

def save_crop_yaml(path, x1, y1, x2, y2):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("crop", np.array([x1, y1, x2, y2], dtype=np.int32))
    fs.release()

def load_crop_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    node = fs.getNode("crop")
    if node.empty():
        fs.release()
        return None
    arr = node.mat()
    fs.release()
    if arr is None:
        return None
    arr = arr.flatten().tolist()
    if len(arr) != 4:
        return None
    return [int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])]

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


# ----------------- Init IMX519 camera -----------------
picam1 = Picamera2(1)
picam1.configure(picam1.create_preview_configuration(main={"size": (640, 480)}))
picam1.start()

# ----------------- Init MLX90640 -----------------
print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768

# ----------------- Load Thermal crop from YAML -----------------
crop_coords_thermal = load_crop_yaml(CROP_YAML)
if crop_coords_thermal:
    print("[INFO] Loaded Thermal crop from YAML:", crop_coords_thermal)
else:
    print("[INFO] No previous Thermal crop YAML found.")

# ----------------- Mouse callback for Thermal crop -----------------
def draw_crop(event, x, y, flags, param):
    global start_point, crop_coords_thermal, drawing_thermal, thermal_img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_thermal = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing_thermal:
        temp_frame = thermal_img_display.copy()
        cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Thermal View (crop select)", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_thermal = False
        end_point = (x, y)
        x1, y1 = start_point
        x2, y2 = end_point

        if x2 > x1 and y2 > y1:
            crop_coords_thermal = [x1, y1, x2, y2]
            save_crop_yaml(CROP_YAML, x1, y1, x2, y2)
            print("[INFO] Thermal crop saved to YAML:", crop_coords_thermal)


# ----------------- Create windows -----------------
cv2.namedWindow("Thermal View (crop select)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thermal View (crop select)", 640, 480)
cv2.setMouseCallback("Thermal View (crop select)", draw_crop)

cv2.namedWindow("IMX519 (fixed)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IMX519 (fixed)", 640, 480)

cv2.namedWindow("Overlay (Thermal moves)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Overlay (Thermal moves)", 640, 480)

# Load MLX manual alignment
load_mlx_alignment()

fps = 0.0

print("\n[CONTROLS]")
print("  H : Move THERMAL left")
print("  K : Move THERMAL right")
print("  I : Move THERMAL up")
print("  J : Move THERMAL down")
print("  + / = : Zoom IN THERMAL")
print("  - / _ : Zoom OUT THERMAL")
print("  0 : Reset THERMAL transform")
print("  p : Save THERMAL transform")
print("  r : Reset Thermal crop (in memory)")
print("  q : Quit\n")

while True:
    t0 = time.time()

    # ---- Capture IMX519 (fixed) ----
    rgb = picam1.capture_array()
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
    rgb = cv2.resize(rgb, (THERM_WIDTH, THERM_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # ---- Read MLX frame ----
    try:
        mlx.getFrame(buf)
    except Exception as e:
        print("[WARN] Thermal frame read error:", e)
        continue

    raw = np.array(buf, dtype=np.float32).reshape(24, 32)
    if FLIP_V:
        raw = np.flip(raw, axis=0)
    if FLIP_H:
        raw = np.flip(raw, axis=1)

    u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
    up = cv2.resize(u8, (THERM_WIDTH, THERM_HEIGHT), interpolation=cv2.INTER_CUBIC)
    thermal_full = apply_colormap(up, COLORMAP)

    # ==========================
    # ✅ FIX: move/zoom BEFORE final resize (translation works)
    # ==========================
    if crop_coords_thermal:
        x1, y1, x2, y2 = crop_coords_thermal
        if x2 > x1 and y2 > y1:
            thermal_base = thermal_full[y1:y2, x1:x2]   # ROI (no resize yet)
        else:
            thermal_base = thermal_full
    else:
        thermal_base = thermal_full

    # Apply manual transform FIRST on thermal_base
    thermal_moved = apply_thermal_transform(thermal_base, t_dx, t_dy, t_scale)

    # Resize ONCE to match overlay size (640x480)
    thermal_moved = cv2.resize(thermal_moved, (THERM_WIDTH, THERM_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # ---- Overlay: RGB fixed + thermal moved ----
    overlay = cv2.addWeighted(rgb, ALPHA_RGB, thermal_moved, ALPHA_THERM, 0)

    # ---- FPS ----
    fps_inst = 1.0 / max(1e-6, (time.time() - t0))
    fps = fps * 0.8 + fps_inst * 0.2

    # ---- HUD ----
    cv2.putText(overlay, f"THERM dx={t_dx} dy={t_dy} scale={t_scale:.2f} | FPS {fps:.1f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Update display buffer used by crop UI
    thermal_img_display = thermal_moved.copy()

    # ---- Show windows ----
    cv2.imshow("Thermal View (crop select)", thermal_img_display)
    cv2.imshow("IMX519 (fixed)", rgb)
    cv2.imshow("Overlay (Thermal moves)", overlay)

    # ---- Key handling (letters) ----
    key = cv2.waitKey(1) & 0xFF  # letters are ASCII, no need waitKeyEx [1](https://qualcomm-confluence.atlassian.net/wiki/spaces/QDATeam/pages/2105443182/gf22fdx+Technology+Documents)

    if key == ord('q'):
        break

    elif key == ord('r'):
        crop_coords_thermal = None
        print("[INFO] Thermal crop reset (in memory).")

    # Move THERMAL using I/J/H/K
    elif key in (ord('h'), ord('H')):
        t_dx -= STEP
    elif key in (ord('k'), ord('K')):
        t_dx += STEP
    elif key in (ord('i'), ord('I')):
        t_dy -= STEP
    elif key in (ord('j'), ord('J')):
        t_dy += STEP

    # Zoom THERMAL
    elif key in (ord('+'), ord('=')):
        t_scale = min(SCALE_MAX, t_scale + ZOOM_STEP)
    elif key in (ord('-'), ord('_')):
        t_scale = max(SCALE_MIN, t_scale - ZOOM_STEP)

    # Reset THERMAL transform
    elif key == ord('0'):
        t_dx, t_dy, t_scale = 0, 0, 1.0
        print("[INFO] Thermal transform reset.")

    # Save THERMAL transform
    elif key in (ord('p'), ord('P')):
        save_mlx_alignment()

cv2.destroyAllWindows()
