#!/usr/bin/env python3
import time
import csv
import json
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2
import yaml
from PIL import Image, ImageTk

# Matplotlib for 3D heatmap
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Picamera2 for CSI sensors on Raspberry Pi (libcamera)
from picamera2 import Picamera2

# YOLO
from ultralytics import YOLO


# ============================================================
# Paths (ONLY mapping configs from data/)
# ============================================================
def repo_root_from_file(file_path: str) -> Path:
    p = Path(file_path).resolve()
    if p.parent.name == "apps":
        return p.parent.parent
    return Path.cwd().resolve()

REPO_ROOT = repo_root_from_file(__file__)
DATA_DIR = REPO_ROOT / "data"

STEREO_YAML = DATA_DIR / "stereo_alignment_matrix.yaml"
MLX_JSON    = DATA_DIR / "mlx_manual_align.json"
CSV_PATH    = DATA_DIR / "detections.csv"

# Keep 640x640 internal working resolution (square)
TW, TH = 640, 640


# ============================================================
# Optional: your existing ThermalMLX
# ============================================================
HAVE_THERMAL_MLX = False
ThermalMLX = None
try:
    from modules.thermal_mlx import ThermalMLX
    HAVE_THERMAL_MLX = True
except Exception:
    HAVE_THERMAL_MLX = False


# ============================================================
# Optional: Adafruit MLX90640 fallback
# ============================================================
HAVE_ADAFRUIT_MLX = False
try:
    import board
    import busio
    import adafruit_mlx90640
    HAVE_ADAFRUIT_MLX = True
except Exception:
    HAVE_ADAFRUIT_MLX = False


# ============================================================
# Helpers
# ============================================================
def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def ensure_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "label", "confidence", "threat_score", "camouflage", "x1", "y1", "x2", "y2"])

def clamp_homography(H):
    """Normalize H so H[2,2]=1 if possible."""
    H = np.array(H, dtype=np.float32)
    if H.shape != (3, 3):
        return np.eye(3, dtype=np.float32)
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H

def load_homography_yaml():
    if not STEREO_YAML.exists():
        print(f"[WARN] Missing {STEREO_YAML}, using identity H.")
        return np.eye(3, dtype=np.float32)

    with open(STEREO_YAML, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    H_list = y.get("homography_secondary_to_master", None)
    if H_list is None:
        print(f"[WARN] Key homography_secondary_to_master missing in {STEREO_YAML}, using identity H.")
        return np.eye(3, dtype=np.float32)

    H = clamp_homography(H_list)
    print("[INFO] Loaded stereo homography from data/stereo_alignment_matrix.yaml")
    return H

def find_homography_in_json(obj):
    """
    Scan JSON object for a 3x3 homography matrix.
    Returns matrix or None.
    """
    if isinstance(obj, dict):
        # direct hit
        for k, v in obj.items():
            key = str(k).lower()
            if "homography" in key and isinstance(v, list):
                arr = np.array(v, dtype=np.float32)
                if arr.shape == (3, 3):
                    return clamp_homography(arr)
        # recurse
        for v in obj.values():
            H = find_homography_in_json(v)
            if H is not None:
                return H
    elif isinstance(obj, list):
        for v in obj:
            H = find_homography_in_json(v)
            if H is not None:
                return H
    return None

def load_mlx_alignment_homography():
    """
    From data/mlx_manual_align.json, try to find thermal->master homography.
    If none found, return identity.
    """
    if not MLX_JSON.exists():
        print(f"[WARN] Missing {MLX_JSON}, thermal alignment will be identity.")
        return np.eye(3, dtype=np.float32)

    try:
        with open(MLX_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        H = find_homography_in_json(obj)
        if H is None:
            print("[WARN] No homography found in mlx_manual_align.json; using identity for thermal alignment.")
            return np.eye(3, dtype=np.float32)
        print("[INFO] Found thermal homography in mlx_manual_align.json")
        return H
    except Exception as e:
        print(f"[WARN] Failed to parse mlx_manual_align.json ({e}); using identity.")
        return np.eye(3, dtype=np.float32)

def letterbox_to_fit(rgb_img, out_w, out_h):
    """
    Keep aspect ratio. Add black borders to fit canvas.
    """
    h, w = rgb_img.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(rgb_img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def overlay_text(img, lines, color=(0, 255, 255)):
    y = 18
    for s in lines:
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 20
    return img


# ============================================================
# Picamera2 capture (IMX500 & IMX519)
# ============================================================
class PiCamStream:
    def __init__(self, cam_index: int, size=(640, 640)):
        self.index = cam_index
        self.picam = Picamera2(cam_index)
        w, h = size
        cfg = self.picam.create_video_configuration(main={"size": (w, h), "format": "RGB888"})
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.2)

    def read_bgr(self):
        rgb = self.picam.capture_array("main")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.picam.stop()
        except Exception:
            pass

def pick_picam_indices():
    """
    Auto-map by sensor string:
      RGB = IMX500
      IR  = IMX519
    Falls back to (0,1).
    """
    info = Picamera2.global_camera_info()

    def haystack(d):
        return " ".join([str(v).lower() for v in d.values()])

    rgb_idx = None
    ir_idx = None
    for i, d in enumerate(info):
        s = haystack(d)
        if rgb_idx is None and "imx500" in s:
            rgb_idx = i
        if ir_idx is None and "imx519" in s:
            ir_idx = i

    if rgb_idx is None:
        rgb_idx = 0
    if ir_idx is None:
        ir_idx = 1 if len(info) > 1 else 0
    if rgb_idx == ir_idx and len(info) > 1:
        ir_idx = 1 if rgb_idx == 0 else 0

    print("[INFO] Picamera2 camera_info:", info)
    print(f"[INFO] RGB(IMX500)->picam_index={rgb_idx}, IR(IMX519)->picam_index={ir_idx}")
    return rgb_idx, ir_idx


# ============================================================
# Thermal sources
# ============================================================
class ThermalSource:
    """
    Try ThermalMLX first (your module). If fails, try Adafruit MLX90640.
    Produces:
      thermal_gray: uint8 2D aligned/warped into (TH, TW)
      thermal_bgr : uint8 3-channel colorized image (TH, TW, 3)
      debug_lines : strings
    """
    def __init__(self):
        self.mode = "none"
        self.mlx = None
        self.ada = None
        self.i2c = None
        self.last_ok = False

        self.H_th = load_mlx_alignment_homography()

        # 1) Try ThermalMLX (repo module)
        if HAVE_THERMAL_MLX and MLX_JSON.exists():
            try:
                self.mlx = ThermalMLX(str(MLX_JSON), (TW, TH))
                self.mode = "thermalmlx"
                print("[INFO] ThermalSource: using modules.thermal_mlx.ThermalMLX")
                return
            except Exception as e:
                print("[WARN] ThermalMLX init failed:", e)

        # 2) Try Adafruit MLX90640 fallback
        if HAVE_ADAFRUIT_MLX:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)
                self.ada = adafruit_mlx90640.MLX90640(self.i2c)
                self.ada.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                self.mode = "adafruit_mlx90640"
                print("[INFO] ThermalSource: using adafruit_mlx90640 fallback")
                return
            except Exception as e:
                print("[WARN] Adafruit MLX90640 init failed:", e)

        self.mode = "none"
        print("[WARN] ThermalSource: thermal disabled (no ThermalMLX / no adafruit_mlx90640)")

    def _colorize(self, gray_u8):
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)

    def read(self):
        if self.mode == "thermalmlx":
            try:
                g = self.mlx.read_aligned_gray()
                if g is None or g.size == 0:
                    raise RuntimeError("ThermalMLX returned empty frame")
                # ensure uint8
                if g.dtype != np.uint8:
                    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                bgr = self.mlx.colorize(g) if hasattr(self.mlx, "colorize") else self._colorize(g)
                self.last_ok = True
                dbg = [f"Thermal: ThermalMLX OK  min={int(g.min())} max={int(g.max())}"]
                return g, bgr, dbg
            except Exception as e:
                self.last_ok = False
                dbg = [f"ThermalMLX FAIL: {e}"]
                # fallthrough to blank
                blank = np.zeros((TH, TW), dtype=np.uint8)
                return blank, self._colorize(blank), dbg

        if self.mode == "adafruit_mlx90640":
            # Read 32x24 temps and upscale to THxTW
            try:
                frame = np.zeros((24 * 32,), dtype=np.float32)
                self.ada.getFrame(frame)
                temp = frame.reshape((24, 32))
                # normalize temps to 0..255
                tmin, tmax = np.percentile(temp, 5), np.percentile(temp, 95)
                if abs(tmax - tmin) < 1e-6:
                    tmax = tmin + 1.0
                gray = np.clip((temp - tmin) * 255.0 / (tmax - tmin), 0, 255).astype(np.uint8)
                gray_up = cv2.resize(gray, (TW, TH), interpolation=cv2.INTER_CUBIC)

                # Warp thermal -> master coordinates using H_th
                warped = cv2.warpPerspective(gray_up, self.H_th, (TW, TH))
                bgr = self._colorize(warped)
                self.last_ok = True
                dbg = [f"Thermal: MLX90640 OK  tmin={tmin:.1f}C tmax={tmax:.1f}C"]
                return warped, bgr, dbg
            except Exception as e:
                self.last_ok = False
                dbg = [f"MLX90640 FAIL: {e}"]
                blank = np.zeros((TH, TW), dtype=np.uint8)
                return blank, self._colorize(blank), dbg

        # none
        blank = np.zeros((TH, TW), dtype=np.uint8)
        return blank, cv2.applyColorMap(blank, cv2.COLORMAP_INFERNO), ["Thermal: OFFLINE"]


# ============================================================
# YOLO detector
# ============================================================
class Detector:
    def __init__(self, weights="yolov8n.pt", conf=0.25):
        self.model = YOLO(weights)
        self.conf = conf
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def infer(self, bgr):
        dets = []
        results = self.model.predict(source=bgr, verbose=False, conf=self.conf)
        if not results:
            return dets
        r = results[0]
        if r.boxes is None:
            return dets
        for box in r.boxes:
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            cls = int(box.cls.item()) if box.cls is not None else -1
            xyxy = box.xyxy[0].tolist()
            label = self.names.get(cls, str(cls)) if isinstance(self.names, dict) else str(cls)
            dets.append({"label": str(label), "conf": conf, "xyxy": [float(x) for x in xyxy]})
        return dets


# ============================================================
# Homography auto-direction selection (H vs invH)
# ============================================================
class HomographyChooser:
    """
    Chooses between H and inv(H) automatically using ORB match score,
    so you don't have to guess direction.
    """
    def __init__(self, H):
        self.H = clamp_homography(H)
        try:
            self.Hinv = clamp_homography(np.linalg.inv(self.H))
        except Exception:
            self.Hinv = np.eye(3, dtype=np.float32)
        self.active = self.H
        self.last_check = 0
        self.orb = cv2.ORB_create(600)

    def _score(self, rgb_gray, ir_gray, Htry):
        try:
            warped = cv2.warpPerspective(ir_gray, Htry, (TW, TH))
            # ORB features
            k1, d1 = self.orb.detectAndCompute(rgb_gray, None)
            k2, d2 = self.orb.detectAndCompute(warped, None)
            if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
                return 0.0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(d1, d2)
            if not matches:
                return 0.0
            matches = sorted(matches, key=lambda m: m.distance)[:80]
            # higher is better: many matches, low distance
            avg_dist = np.mean([m.distance for m in matches])
            score = len(matches) / (avg_dist + 1e-6)
            return float(score)
        except Exception:
            return 0.0

    def maybe_update(self, rgb_bgr, ir_bgr):
        # run every 2 seconds max
        now = time.time()
        if now - self.last_check < 2.0:
            return self.active

        self.last_check = now
        rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        ir_gray  = cv2.cvtColor(ir_bgr,  cv2.COLOR_BGR2GRAY)

        s1 = self._score(rgb_gray, ir_gray, self.H)
        s2 = self._score(rgb_gray, ir_gray, self.Hinv)

        self.active = self.H if s1 >= s2 else self.Hinv
        print(f"[INFO] HomographyChooser score H={s1:.3f} invH={s2:.3f} -> using {'H' if s1>=s2 else 'invH'}")
        return self.active


# ============================================================
# Threat + Heatmap
# ============================================================
def compute_threat_and_camouflage(frame_shape, bbox_xyxy, conf, thermal_gray=None, camo_delta_c=1.5):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / float(w * h)

    base = 0.55 * float(conf) + 0.45 * min(1.0, area_ratio * 6.0)
    camouflage_flag = False

    if thermal_gray is not None and thermal_gray.shape[:2] == (h, w):
        roi = thermal_gray[y1:y2, x1:x2]
        if roi.size > 0:
            roi_mean = float(np.mean(roi))
            pad_x = int(0.2 * bw)
            pad_y = int(0.2 * bh)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(w, x2 + pad_x)
            ry2 = min(h, y2 + pad_y)

            ring = thermal_gray[ry1:ry2, rx1:rx2].copy()
            ring_mask = np.ones(ring.shape, dtype=bool)
            ix1 = x1 - rx1
            iy1 = y1 - ry1
            ix2 = ix1 + (x2 - x1)
            iy2 = iy1 + (y2 - y1)
            ring_mask[iy1:iy2, ix1:ix2] = False
            ring_vals = ring[ring_mask]

            if ring_vals.size > 0:
                ring_mean = float(np.mean(ring_vals))
                thermal_delta = roi_mean - ring_mean
                if thermal_delta < camo_delta_c:
                    camouflage_flag = True
                td_norm = max(0.0, min(1.0, (thermal_delta / 8.0)))
                base = 0.70 * base + 0.30 * td_norm

    threat_score = base + (0.12 if camouflage_flag else 0.0)
    threat_score = max(0.0, min(1.0, threat_score))
    return threat_score, camouflage_flag

class Heatmap3D:
    def __init__(self, grid_w=40, grid_h=40, decay=0.90):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.decay = decay
        self.Z = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.X, self.Y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))

    def step(self):
        self.Z *= self.decay

    def add_blob(self, cx_norm, cy_norm, intensity, sigma=2.2):
        gx = cx_norm * (self.grid_w - 1)
        gy = cy_norm * (self.grid_h - 1)
        dx = self.X - gx
        dy = self.Y - gy
        blob = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        self.Z += intensity * blob
        self.Z = np.clip(self.Z, 0.0, 3.0)


# ============================================================
# Tk video canvas (letterbox to keep square)
# ============================================================
class VideoCanvas:
    def __init__(self, parent, title):
        self.frame = ttk.LabelFrame(parent, text=title)
        self.canvas = tk.Canvas(self.frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._photo = None
        self._img_item = None

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def update_bgr(self, bgr):
        if bgr is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fitted = letterbox_to_fit(rgb, cw, ch)

        im = Image.fromarray(fitted)
        self._photo = ImageTk.PhotoImage(im)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._img_item, image=self._photo)


# ============================================================
# Main App (adds 3 extra rows -> squarer panels)
# ============================================================
class SpectraApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Bigger height + 3 extra rows (requested)
        self.title("SPECTRA : HYPERSPECTRAL SURVEILLANCE")
        self.geometry("1200x980")
        self.configure(bg="#1e1e1e")

        # Grid: add 3 more rows (23 total instead of 20)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        for i in range(23):
            self.grid_rowconfigure(i, weight=1)

        # Header
        header = tk.Label(
            self,
            text="SPECTRA : HYPERSPECTRAL SURVEILLANCE",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2b2b2b",
            pady=12
        )
        header.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

        # Streams (square-friendly row spans)
        self.rgb_panel = VideoCanvas(self, "RGB Stream (IMX500)")
        self.ir_panel  = VideoCanvas(self, "Infrared Stream (IMX519, warped)")
        self.th_panel  = VideoCanvas(self, "Thermal Stream (MLX)")

        # Use 6-row blocks to better match square 640x640 (was 5)
        self.rgb_panel.grid(row=1,  column=0, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.ir_panel.grid (row=7,  column=0, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.th_panel.grid (row=13, column=0, rowspan=6, sticky="nsew", padx=8, pady=6)

        # Heatmap (also 6 rows)
        heatmap_frame = ttk.LabelFrame(self, text="Heatmap (3D Threat/Camouflage)")
        heatmap_frame.grid(row=1, column=1, rowspan=6, sticky="nsew", padx=8, pady=6)

        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_title("Threat Intensity Surface")
        self.ax3d.set_xlabel("X grid")
        self.ax3d.set_ylabel("Y grid")
        self.ax3d.set_zlabel("Intensity")
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=heatmap_frame)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(fill="both", expand=True)

        # Detection console
        detect_frame = ttk.LabelFrame(self, text="Detection Console")
        detect_frame.grid(row=7, column=1, rowspan=5, sticky="nsew", padx=8, pady=6)

        self.detect_lines = {
            "Person": tk.Label(detect_frame, text="Person: --", font=("Segoe UI", 12), anchor="w"),
            "Vehicle": tk.Label(detect_frame, text="Vehicle: --", font=("Segoe UI", 12), anchor="w"),
            "Background": tk.Label(detect_frame, text="Background: --", font=("Segoe UI", 12), anchor="w"),
            "Alert": tk.Label(detect_frame, text="Alert: --", font=("Segoe UI", 12, "bold"), anchor="w", fg="orange"),
        }
        for k in ["Person", "Vehicle", "Background", "Alert"]:
            self.detect_lines[k].pack(fill="x", padx=10, pady=6)

        # Stats
        stats_frame = ttk.LabelFrame(self, text="Stats")
        stats_frame.grid(row=12, column=1, rowspan=7, sticky="nsew", padx=8, pady=6)

        self.stat_vars = {
            "Accuracy": tk.StringVar(value="--"),
            "Threat Score": tk.StringVar(value="--"),
            "Timestamp": tk.StringVar(value="--"),
            "FPS": tk.StringVar(value="--"),
        }
        for stat in ["Accuracy", "Threat Score", "Timestamp", "FPS"]:
            row = tk.Frame(stats_frame)
            row.pack(fill="x", padx=6, pady=4)
            tk.Label(row, text=stat, width=14, anchor="w").pack(side="left")
            tk.Label(row, textvariable=self.stat_vars[stat]).pack(side="left")

        # Runtime
        self.running = True
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load mappings
        H_stereo = load_homography_yaml()
        self.hchooser = HomographyChooser(H_stereo)

        self.thermal = ThermalSource()

        # Cameras
        rgb_idx, ir_idx = pick_picam_indices()
        self.cam_rgb = PiCamStream(rgb_idx, size=(TW, TH))
        self.cam_ir  = PiCamStream(ir_idx,  size=(TW, TH))

        # YOLO weights: prefer models/yolov8n.pt if exists
        weights = "yolov8n.pt"
        alt = REPO_ROOT / "models" / "yolov8n.pt"
        if alt.exists():
            weights = str(alt)

        self.detector = Detector(weights=weights, conf=0.25)

        # Heatmap engine
        self.hm = Heatmap3D(grid_w=40, grid_h=40, decay=0.90)
        self.frame_count = 0

        # Logging
        ensure_csv()

        # FPS + last detection
        self._last_t = time.perf_counter()
        self._fps_ema = 0.0
        self._last_threat = None
        self._last_detect_time = None
        self._last_camo = False

        # Start loop
        self.after(20, self.loop)

    def update_heatmap_plot(self):
        self.ax3d.cla()
        self.ax3d.set_title("Threat Intensity Surface")
        self.ax3d.set_xlabel("X grid")
        self.ax3d.set_ylabel("Y grid")
        self.ax3d.set_zlabel("Intensity")
        self.ax3d.plot_surface(self.hm.X, self.hm.Y, self.hm.Z, cmap="inferno", linewidth=0, antialiased=True)
        self.ax3d.set_zlim(0, max(1.0, float(np.max(self.hm.Z)) + 0.5))
        self.canvas_fig.draw()

    def log_detection(self, label, conf, threat_score, camouflage, bbox):
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            x1, y1, x2, y2 = bbox
            w.writerow([now_iso(), label, f"{conf:.4f}", f"{threat_score:.4f}", int(camouflage), x1, y1, x2, y2])

    def loop(self):
        if not self.running:
            return

        t0 = time.perf_counter()

        rgb = self.cam_rgb.read_bgr()
        ir  = self.cam_ir.read_bgr()

        # Auto-select correct direction H vs invH
        H_active = self.hchooser.maybe_update(rgb, ir)

        # Warp IR -> RGB
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        ir_warp = cv2.warpPerspective(ir, H_active, (TW, TH))

        # Thermal
        thermal_gray, thermal_bgr, tdbg = self.thermal.read()

        # YOLO on RGB
        dets = self.detector.infer(rgb)

        # Visuals
        vis_rgb = rgb.copy()
        vis_ir  = ir_warp.copy()
        vis_th  = thermal_bgr.copy()

        # Draw boxes on all 3 streams (so you see rectangles everywhere)
        for d in dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(vis_ir,  (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(vis_th,  (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Debug overlays to verify alignment + thermal
        vis_ir = overlay_text(vis_ir, [f"H active: {'H' if np.allclose(H_active, self.hchooser.H) else 'invH'}"])
        vis_th = overlay_text(vis_th, tdbg, color=(255, 255, 0))

        # Heatmap + stats
        self.hm.step()
        person_count, vehicle_count, bg_count = 0, 0, 0
        best_threat, best_conf, best_camo = None, 0.0, False

        for d in dets:
            label = d["label"].lower()
            conf = float(d["conf"])
            xyxy = d["xyxy"]

            if "person" in label:
                person_count += 1
                threat_score, camo_flag = compute_threat_and_camouflage(
                    rgb.shape, xyxy, conf, thermal_gray, camo_delta_c=1.5
                )

                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2.0) / TW
                cy = ((y1 + y2) / 2.0) / TH
                self.hm.add_blob(cx, cy, intensity=2.0 * threat_score, sigma=2.2)

                if best_threat is None or threat_score > best_threat:
                    best_threat = threat_score
                    best_conf = conf
                    best_camo = camo_flag

                if threat_score >= 0.30:
                    self.log_detection("person", conf, threat_score, camo_flag, xyxy)

            elif any(k in label for k in ["car", "truck", "bus", "vehicle", "motorbike"]):
                vehicle_count += 1
            else:
                bg_count += 1

        # Stats panel values
        acc_pct = (best_conf * 100.0) if person_count > 0 else 0.0
        if best_threat is not None:
            self._last_threat = best_threat
            self._last_detect_time = now_iso()
            self._last_camo = best_camo

        self.stat_vars["Accuracy"].set(f"{acc_pct:.1f}%")
        self.stat_vars["Threat Score"].set("--" if self._last_threat is None else f"{self._last_threat:.3f}" + (" (CAMO)" if self._last_camo else ""))
        self.stat_vars["Timestamp"].set("--" if self._last_detect_time is None else self._last_detect_time)

        dt = max(1e-6, t0 - self._last_t)
        fps = 1.0 / dt
        self._fps_ema = (0.90 * self._fps_ema + 0.10 * fps) if self._fps_ema > 0 else fps
        self._last_t = t0
        self.stat_vars["FPS"].set(f"{self._fps_ema:.1f}")

        self.detect_lines["Person"].config(text=f"Person: {person_count}")
        self.detect_lines["Vehicle"].config(text=f"Vehicle: {vehicle_count}")
        self.detect_lines["Background"].config(text=f"Background: {bg_count}")

        if best_threat is not None and best_threat >= 0.65:
            self.detect_lines["Alert"].config(text=f"Alert: THREAT ({best_threat:.2f})" + (" + CAMO" if best_camo else ""), fg="red")
        elif best_threat is not None and best_camo:
            self.detect_lines["Alert"].config(text="Alert: CAMOUFLAGE SUSPECTED", fg="orange")
        else:
            self.detect_lines["Alert"].config(text="Alert: --", fg="orange")

        # Update GUI (letterboxed to keep square)
        self.rgb_panel.update_bgr(vis_rgb)
        self.ir_panel.update_bgr(vis_ir)
        self.th_panel.update_bgr(vis_th)

        # Heatmap update every 2 frames
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            self.update_heatmap_plot()

        self.after(20, self.loop)

    def on_close(self):
        self.running = False
        try:
            self.cam_rgb.close()
            self.cam_ir.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = SpectraApp()
    app.mainloop()
