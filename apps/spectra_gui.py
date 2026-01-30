#!/usr/bin/env python3
import os
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

# Picamera2 for CSI sensors
from picamera2 import Picamera2

# YOLO (Ultralytics)
from ultralytics import YOLO

# MLX90640 (Adafruit)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board
import busio
import adafruit_mlx90640


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

# Square working resolution
TW, TH = 640, 640

# Thermal display range
THERM_MIN = 20.0
THERM_MAX = 45.0
COLORMAP = cv2.COLORMAP_TURBO

FLIP_V = True
FLIP_H = False

# Manual thermal alignment controls (saved into mlx_manual_align.json)
STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

# Detection thresholds
YOLO_CONF = 0.25
LOG_THREAT_THRESHOLD = 0.30
ALERT_THREAT_THRESHOLD = 0.65

# Heatmap settings
HM_GRID_W = 40
HM_GRID_H = 40
HM_DECAY = 0.90
HM_SIGMA = 2.2
HM_RENDER_EVERY = 2  # frames


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
    print("[INFO] Loaded stereo homography")
    return clamp_homography(H_list)

def letterbox_to_fit(rgb_img, out_w, out_h):
    """Keep aspect ratio; add black borders to fit the canvas."""
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


# ============================================================
# Camera capture (Picamera2) - use XBGR8888 (stable)
# ============================================================
class PiCamStream:
    def __init__(self, cam_index: int, size=(640, 640)):
        self.index = cam_index
        self.picam = Picamera2(camera_num=cam_index)
        w, h = size

        cfg = self.picam.create_preview_configuration(
            main={"size": (w, h), "format": "XBGR8888"},
            buffer_count=8
        )

        # best-effort disable raw
        try:
            cfg.enable_raw(False)
        except Exception:
            try:
                cfg["raw"] = None
            except Exception:
                pass

        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.2)

    def read_bgr(self):
        frame = self.picam.capture_array()
        # XBGR8888 comes as 4 channels; convert to BGR
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

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
# Thermal (MLX90640) - working logic + dx/dy/scale persistence
# ============================================================
class ThermalMLXManual:
    def __init__(self):
        self.dx = 0
        self.dy = 0
        self.scale = 1.0

        self._load_manual_align()

        # Stable I2C settings
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

        self.buf = [0.0] * 768

    def _load_manual_align(self):
        if not MLX_JSON.exists():
            return
        try:
            with open(MLX_JSON, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            self.dx = int(d.get("dx", 0))
            self.dy = int(d.get("dy", 0))
            self.scale = float(d.get("scale", 1.0))
            print("[INFO] Loaded MLX manual align:", {"dx": self.dx, "dy": self.dy, "scale": self.scale})
        except Exception as e:
            print("[WARN] Failed to load mlx_manual_align.json:", e)

    def save_manual_align(self):
        d = {}
        if MLX_JSON.exists():
            try:
                with open(MLX_JSON, "r", encoding="utf-8") as f:
                    d = json.load(f) or {}
            except Exception:
                d = {}
        d["dx"] = int(self.dx)
        d["dy"] = int(self.dy)
        d["scale"] = float(self.scale)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(MLX_JSON, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print("[INFO] Saved MLX manual align:", {"dx": self.dx, "dy": self.dy, "scale": self.scale})

    def _norm_thermal(self, t):
        t = np.clip(t, THERM_MIN, THERM_MAX)
        return ((t - THERM_MIN) / (THERM_MAX - THERM_MIN) * 255.0).astype(np.uint8)

    def _apply_thermal_transform(self, img_bgr):
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        s = float(self.scale)
        M = np.array([
            [s, 0.0, (1 - s) * cx + self.dx],
            [0.0, s, (1 - s) * cy + self.dy]
        ], dtype=np.float32)
        return cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    def read(self):
        """Returns (thermal_gray_u8, thermal_bgr, debug_lines)."""
        try:
            self.mlx.getFrame(self.buf)
        except Exception as e:
            blank = np.zeros((TH, TW), dtype=np.uint8)
            bgr = cv2.applyColorMap(blank, cv2.COLORMAP_INFERNO)
            return blank, bgr, [f"MLX read error: {e}"]

        raw = np.array(self.buf, dtype=np.float32).reshape(24, 32)

        if FLIP_V:
            raw = np.flip(raw, axis=0)
        if FLIP_H:
            raw = np.flip(raw, axis=1)

        u8 = self._norm_thermal(raw)
        up = cv2.resize(u8, (TW, TH), interpolation=cv2.INTER_CUBIC)
        bgr = cv2.applyColorMap(up, COLORMAP)
        moved = self._apply_thermal_transform(bgr)

        dbg = [f"MLX OK dx={self.dx} dy={self.dy} s={self.scale:.2f}"]
        return up, moved, dbg


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
# Tk video canvas
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
        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fitted = letterbox_to_fit(rgb, cw, ch)

        im = Image.fromarray(fitted)
        self._photo = ImageTk.PhotoImage(im)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._img_item, image=self._photo)


# ============================================================
# Drawing helper for boxes + confidence
# ============================================================
VEHICLE_KEYS = ("car", "truck", "bus", "motorbike", "motorcycle", "vehicle")

def draw_detections(img, dets, color=(0, 255, 0)):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        label = d["label"]
        conf = float(d["conf"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.2f}"
        cv2.putText(out, txt, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


# ============================================================
# Main App (UI you requested + heatmap + stats + MLX controls)
# ============================================================
class SpectraApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("SPECTRA : HYPERSPECTRAL SURVEILLANCE")
        self.geometry("1200x980")
        self.configure(bg="#1e1e1e")

        # Grid
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        for i in range(23):
            self.grid_rowconfigure(i, weight=1)

        header = tk.Label(
            self,
            text="SPECTRA : HYPERSPECTRAL SURVEILLANCE",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2b2b2b",
            pady=12
        )
        header.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

        # Streams
        self.rgb_panel = VideoCanvas(self, "RGB Stream (IMX500) + YOLO")
        self.ir_panel  = VideoCanvas(self, "Infrared Stream (IMX519 warped) + YOLO")
        self.th_panel  = VideoCanvas(self, "Thermal Stream (MLX90640) + YOLO")

        self.rgb_panel.grid(row=1,  column=0, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.ir_panel.grid (row=7,  column=0, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.th_panel.grid (row=13, column=0, rowspan=6, sticky="nsew", padx=8, pady=6)

        # Heatmap
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
            "Alert": tk.Label(detect_frame, text="Alert: --", font=("Segoe UI", 12, "bold"),
                              anchor="w", fg="orange"),
            "Controls": tk.Label(detect_frame,
                                 text="MLX Controls: H/K/I/J move, +/- zoom, 0 reset, P save",
                                 font=("Segoe UI", 10), anchor="w", fg="#bbbbbb")
        }
        for k in ["Person", "Vehicle", "Background", "Alert", "Controls"]:
            self.detect_lines[k].pack(fill="x", padx=10, pady=4)

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
        self.H = load_homography_yaml()

        # Cameras
        rgb_idx, ir_idx = pick_picam_indices()
        self.cam_rgb = PiCamStream(rgb_idx, size=(TW, TH))
        self.cam_ir  = PiCamStream(ir_idx,  size=(TW, TH))

        # Thermal (working MLX)
        self.thermal = ThermalMLXManual()

        # YOLO weights preference
        weights = "yolov8n.pt"
        if (REPO_ROOT / "yolov8n.pt").exists():
            weights = str(REPO_ROOT / "yolov8n.pt")
        elif (REPO_ROOT / "models" / "yolov8n.pt").exists():
            weights = str(REPO_ROOT / "models" / "yolov8n.pt")

        self.detector = Detector(weights=weights, conf=YOLO_CONF)

        # Heatmap
        self.hm = Heatmap3D(grid_w=HM_GRID_W, grid_h=HM_GRID_H, decay=HM_DECAY)
        self.frame_count = 0

        # Logging
        ensure_csv()

        # FPS & last detection
        self._last_t = time.perf_counter()
        self._fps_ema = 0.0
        self._last_threat = None
        self._last_detect_time = None
        self._last_camo = False

        # Key bindings for MLX control
        self.bind("<KeyPress>", self.on_key)

        # Start loop
        self.after(20, self.loop)

    # --- MLX key controls ---
    def on_key(self, event):
        k = event.keysym.lower()
        if k == "h":
            self.thermal.dx -= STEP
        elif k == "k":
            self.thermal.dx += STEP
        elif k == "i":
            self.thermal.dy -= STEP
        elif k == "j":
            self.thermal.dy += STEP
        elif k in ("plus", "equal"):
            self.thermal.scale = min(SCALE_MAX, self.thermal.scale + ZOOM_STEP)
        elif k in ("minus", "underscore"):
            self.thermal.scale = max(SCALE_MIN, self.thermal.scale - ZOOM_STEP)
        elif k == "0":
            self.thermal.dx, self.thermal.dy, self.thermal.scale = 0, 0, 1.0
        elif k == "p":
            self.thermal.save_manual_align()

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

        rgb = cv2.resize(rgb, (TW, TH), interpolation=cv2.INTER_AREA)
        ir  = cv2.resize(ir,  (TW, TH), interpolation=cv2.INTER_AREA)

        # Warp IR -> RGB
        ir_warp = cv2.warpPerspective(ir, self.H, (TW, TH),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        # Thermal (manual aligned)
        thermal_gray, thermal_bgr, tdbg = self.thermal.read()

        # YOLO on RGB frame
        dets = self.detector.infer(rgb)

        # Draw boxes on all views
        vis_rgb = draw_detections(rgb, dets, color=(0, 255, 0))
        vis_ir  = draw_detections(ir_warp, dets, color=(0, 255, 0))
        vis_th  = draw_detections(thermal_bgr, dets, color=(0, 255, 0))

        # Heatmap + stats
        self.hm.step()

        person_count, vehicle_count, bg_count = 0, 0, 0
        best_threat, best_conf, best_camo = None, 0.0, False

        # Accumulate avg confidence for persons
        person_confs = []

        for d in dets:
            label = d["label"].lower()
            conf = float(d["conf"])
            xyxy = d["xyxy"]

            if "person" in label:
                person_count += 1
                person_confs.append(conf)

                threat_score, camo_flag = compute_threat_and_camouflage(
                    rgb.shape, xyxy, conf, thermal_gray, camo_delta_c=1.5
                )

                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2.0) / TW
                cy = ((y1 + y2) / 2.0) / TH
                self.hm.add_blob(cx, cy, intensity=2.0 * threat_score, sigma=HM_SIGMA)

                if best_threat is None or threat_score > best_threat:
                    best_threat = threat_score
                    best_conf = conf
                    best_camo = camo_flag

                if threat_score >= LOG_THREAT_THRESHOLD:
                    self.log_detection("person", conf, threat_score, camo_flag, xyxy)

            elif any(k in label for k in VEHICLE_KEYS):
                vehicle_count += 1
            else:
                bg_count += 1

        # Accuracy = avg person confidence
        acc_pct = (np.mean(person_confs) * 100.0) if person_confs else 0.0

        if best_threat is not None:
            self._last_threat = best_threat
            self._last_detect_time = now_iso()
            self._last_camo = best_camo

        self.stat_vars["Accuracy"].set(f"{acc_pct:.1f}%")
        self.stat_vars["Threat Score"].set(
            "--" if self._last_threat is None else f"{self._last_threat:.3f}" + (" (CAMO)" if self._last_camo else "")
        )
        self.stat_vars["Timestamp"].set("--" if self._last_detect_time is None else self._last_detect_time)

        # FPS
        dt = max(1e-6, t0 - self._last_t)
        fps = 1.0 / dt
        self._fps_ema = (0.90 * self._fps_ema + 0.10 * fps) if self._fps_ema > 0 else fps
        self._last_t = t0
        self.stat_vars["FPS"].set(f"{self._fps_ema:.1f}")

        # Console
        self.detect_lines["Person"].config(text=f"Person: {person_count}")
        self.detect_lines["Vehicle"].config(text=f"Vehicle: {vehicle_count}")
        self.detect_lines["Background"].config(text=f"Background: {bg_count}")

        if best_threat is not None and best_threat >= ALERT_THREAT_THRESHOLD:
            self.detect_lines["Alert"].config(
                text=f"Alert: THREAT ({best_threat:.2f})" + (" + CAMO" if best_camo else ""),
                fg="red"
            )
        elif best_threat is not None and best_camo:
            self.detect_lines["Alert"].config(text="Alert: CAMOUFLAGE SUSPECTED", fg="orange")
        else:
            self.detect_lines["Alert"].config(text="Alert: --", fg="orange")

        # Update GUI
        self.rgb_panel.update_bgr(vis_rgb)
        self.ir_panel.update_bgr(vis_ir)
        self.th_panel.update_bgr(vis_th)

        # Heatmap redraw throttle
        self.frame_count += 1
        if self.frame_count % HM_RENDER_EVERY == 0:
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
