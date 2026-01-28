
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

def norm_thermal(t, lo, hi):
    t = np.clip(t, lo, hi)
    return ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)

def apply_colormap(gray_u8, cmap_name):
    cmap_id = getattr(cv2, cmap_name)
    return cv2.applyColorMap(gray_u8, cmap_id)

# Initialize IMX519 camera
picam1 = Picamera2(1)
picam1.configure(picam1.create_preview_configuration(main={"size": (640, 480)}))
picam1.start()

# Initialize thermal camera
print("[INFO] Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
buf = [0.0] * 768

# Crop settings for Thermal
crop_file_thermal = "crop_thermal.json"
crop_coords_thermal = None
drawing_thermal = False
start_point = None

# Load previous crop settings
try:
    with open(crop_file_thermal, "r") as f:
        crop_coords_thermal = json.load(f)
        print("Loaded Thermal crop:", crop_coords_thermal)
except FileNotFoundError:
    print("No previous Thermal crop settings found.")

# Mouse callback for thermal crop
def draw_crop(event, x, y, flags, param):
    global start_point, crop_coords_thermal, drawing_thermal
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_thermal = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_thermal:
        temp_frame = thermal_img.copy()
        cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Thermal Crop", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_thermal = False
        end_point = (x, y)
        if end_point[0] > start_point[0] and end_point[1] > start_point[1]:
            crop_coords_thermal = [start_point[0], start_point[1], end_point[0], end_point[1]]
            with open(crop_file_thermal, "w") as f:
                json.dump(crop_coords_thermal, f)
            print("Thermal crop saved:", crop_coords_thermal)

# Create windows
cv2.namedWindow("Thermal Crop", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thermal Crop", 640, 480)
cv2.setMouseCallback("Thermal Crop", draw_crop)

cv2.namedWindow("Thermal + IMX519 Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thermal + IMX519 Overlay", 640, 480)

fps = 0.0
while True:
    t0 = time.time()

    # Capture IMX519 frame
    frame1 = picam1.capture_array()
    if frame1.shape[2] == 4:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)

    # Thermal frame
    try:
        mlx.getFrame(buf)
    except Exception as e:
        print("Frame read error:", e)
        continue

    raw = np.array(buf, dtype=np.float32).reshape(24, 32)
    if FLIP_V:
        raw = np.flip(raw, axis=0)
    if FLIP_H:
        raw = np.flip(raw, axis=1)

    u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
    up = cv2.resize(u8, (THERM_WIDTH, THERM_HEIGHT), interpolation=cv2.INTER_CUBIC)
    thermal_img = apply_colormap(up, COLORMAP)

    # Apply crop to thermal
    if crop_coords_thermal:
        tx1, ty1, tx2, ty2 = crop_coords_thermal
        if tx2 > tx1 and ty2 > ty1:
            thermal_img = thermal_img[ty1:ty2, tx1:tx2]
            thermal_img = cv2.resize(thermal_img, (THERM_WIDTH, THERM_HEIGHT))

    # FPS calculation
    fps_inst = 1.0 / max(1e-6, (time.time() - t0))
    fps = fps * 0.8 + fps_inst * 0.2
    cv2.putText(thermal_img, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)

    # Overlay IMX519 on cropped thermal
    imx519_resized = cv2.resize(frame1, (THERM_WIDTH, THERM_HEIGHT))
    overlay = cv2.addWeighted(thermal_img, 0.5, imx519_resized, 0.5, 0)

    # Show both windows
    cv2.imshow("Thermal Crop", thermal_img)
    cv2.imshow("Thermal + IMX519 Overlay", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        crop_coords_thermal = None
        print("Thermal crop reset.")

cv2.destroyAllWindows()
