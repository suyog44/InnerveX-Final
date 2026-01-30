#!/usr/bin/env python3
import os
import time
import csv
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

# Picamera2 (libcamera)
from picamera2 import Picamera2

# YOLO fallback (robust)
from ultralytics import YOLO


# ============================================================
# Repo paths + config bootstrap
# ============================================================

def repo_root_from_file(file_path: str) -> Path:
    p = Path(file_path).resolve()
    if p.parent.name == "apps":
        return p.parent.parent
    return Path.cwd().resolve()

REPO_ROOT = repo_root_from_file(__file__)
DATA_DIR = REPO_ROOT / "data"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def safe_write_yaml(path: Path, data: dict):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_config_path(user_path: str | None) -> Path:
    if not user_path:
        return DATA_DIR / "spectra_config.yaml"
    p = Path(user_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()

def bootstrap_default_config(cfg_path: Path):
    ensure_dir(DATA_DIR)
    default_cfg = {
        "app": {
            "title": "SPECTRA : Hyperspectral Surveillance",
            "header_text": "SPECTRA : HYPERSPECTRAL SURVEILLANCE",
            "geometry": "1200x800",
            "update_ms": 20
        },
        "video": {
            "target_width": 640,
            "target_height": 640
        },
        # Mapping you requested:
        "cameras": {
            "rgb": {"sensor": "imx500", "picam_index": 0},  # Picamera2 index 0
            "ir":  {"sensor": "imx519", "picam_index": 1}   # Picamera2 index 1
        },
        # alignment assumes homography_secondary_to_master = IR -> RGB
        "alignment": {
            "stereo_homography_yaml": str(DATA_DIR / "stereo_alignment_matrix.yaml"),
            "thermal_align_json": str(DATA_DIR / "mlx_manual_align.json")
        },
        "yolo": {
            "weights": "yolov8n.pt",
            "conf_threshold": 0.25
        },
        "threat": {
            "score_threshold": 0.65,
            "camouflage_thermal_delta_c": 1.5
        },
        "heatmap": {
            "grid_w": 40,
            "grid_h": 40,
            "decay": 0.90,
            "sigma": 2.2,
            "render_every_n_frames": 2
        },
        "logging": {
            "csv_path": str(DATA_DIR / "detections.csv"),
            "log_threat_threshold": 0.30
        }
    }
    safe_write_yaml(cfg_path, default_cfg)
    print(f"[INFO] Created missing config: {cfg_path}")


# ============================================================
# Video capture using Picamera2 (fixes your error)
# ============================================================

class PiCamStream:
    """
    Captures frames via Picamera2 (libcamera) and returns BGR images.
    """
    def __init__(self, cam_index: int, size=(640, 640), framerate=30):
        self.picam = Picamera2(cam_index)
        w, h = size

        config = self.picam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"}
        )
        self.picam.configure(config)
        self.picam.start()
        # small warm-up
        time.sleep(0.2)

    def read_bgr(self):
        # capture_array returns RGB
        rgb = self.picam.capture_array("main")
        # convert to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def close(self):
        try:
            self.picam.stop()
        except Exception:
            pass


# ============================================================
# Detector wrapper
# ============================================================

class Detector:
    def __init__(self, weights: str, conf: float):
        self.model = YOLO(weights)
        self.conf = conf
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def infer(self, bgr):
        out = []
        results = self.model.predict(source=bgr, verbose=False, conf=self.conf)
        if not results:
            return out
        r = results[0]
        if r.boxes is None:
            return out
        for box in r.boxes:
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            cls = int(box.cls.item()) if box.cls is not None else -1
            xyxy = box.xyxy[0].tolist()
            label = self.names.get(cls, str(cls)) if isinstance(self.names, dict) else str(cls)
            out.append({"label": str(label), "conf": conf, "xyxy": [float(x) for x in xyxy]})
        return out


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
# Tk video panel helper
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
        resized = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(resized)
        self._photo = ImageTk.PhotoImage(im)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._img_item, image=self._photo)


# ============================================================
# Main App
# ============================================================

class SpectraApp(tk.Tk):
    def __init__(self, config_path: str | None = None):
        super().__init__()

        cfg_path = resolve_config_path(config_path)
        if not cfg_path.exists():
            bootstrap_default_config(cfg_path)
        self.cfg = load_yaml(cfg_path)

        app_cfg = self.cfg.get("app", {})
        self.title(app_cfg.get("title", "SPECTRA"))
        self.geometry(app_cfg.get("geometry", "1200x800"))
        self.configure(bg="#1e1e1e")

        # Grid
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        for i in range(20):
            self.grid_rowconfigure(i, weight=1)

        header = tk.Label(
            self,
            text=app_cfg.get("header_text", "SPECTRA : HYPERSPECTRAL SURVEILLANCE"),
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2b2b2b",
            pady=12
        )
        header.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

        # Panels (your requested mapping)
        self.rgb_panel = VideoCanvas(self, "RGB Stream (IMX500)")
        self.ir_panel  = VideoCanvas(self, "Infrared Stream (IMX519)")
        self.th_panel  = VideoCanvas(self, "Thermal Stream (MLX)")

        self.rgb_panel.grid(row=1, column=0, rowspan=5, sticky="nsew", padx=8, pady=6)
        self.ir_panel.grid (row=6, column=0, rowspan=5, sticky="nsew", padx=8, pady=6)
        self.th_panel.grid (row=11, column=0, rowspan=5, sticky="nsew", padx=8, pady=6)

        # Heatmap
        heatmap_frame = ttk.LabelFrame(self, text="Heatmap (3D Threat/Camouflage)")
        heatmap_frame.grid(row=1, column=1, rowspan=5, sticky="nsew", padx=8, pady=6)

        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_title("Threat Intensity Surface")
        self.ax3d.set_xlabel("X grid")
        self.ax3d.set_ylabel("Y grid")
        self.ax3d.set_zlabel("Intensity")
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=heatmap_frame)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(fill="both", expand=True)

        # Detection Console
        self.detect_frame = ttk.LabelFrame(self, text="Detection Console")
        self.detect_frame.grid(row=6, column=1, rowspan=4, sticky="nsew", padx=8, pady=6)

        self.detect_lines = {
            "Person": tk.Label(self.detect_frame, text="Person: --", font=("Segoe UI", 12), anchor="w"),
            "Vehicle": tk.Label(self.detect_frame, text="Vehicle: --", font=("Segoe UI", 12), anchor="w"),
            "Background": tk.Label(self.detect_frame, text="Background: --", font=("Segoe UI", 12), anchor="w"),
            "Alert": tk.Label(self.detect_frame, text="Alert: --", font=("Segoe UI", 12, "bold"), anchor="w", fg="orange"),
        }
        for k in ["Person", "Vehicle", "Background", "Alert"]:
            self.detect_lines[k].pack(fill="x", padx=10, pady=6)

        # Stats
        stats_frame = ttk.LabelFrame(self, text="Stats")
        stats_frame.grid(row=10, column=1, rowspan=6, sticky="nsew", padx=8, pady=6)

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

        self.TW = int(self.cfg.get("video", {}).get("target_width", 640))
        self.TH = int(self.cfg.get("video", {}).get("target_height", 640))

        # Load homography (IR -> RGB)
        self.H = np.eye(3, dtype=np.float32)
        stereo_yaml = Path(self.cfg.get("alignment", {}).get("stereo_homography_yaml", str(DATA_DIR / "stereo_alignment_matrix.yaml")))
        if not stereo_yaml.is_absolute():
            stereo_yaml = (REPO_ROOT / stereo_yaml).resolve()
        if stereo_yaml.exists():
            with open(stereo_yaml, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            h_list = loaded.get("homography_secondary_to_master", None)
            if h_list is not None:
                self.H = np.array(h_list, dtype=np.float32)

        # Thermal (placeholder: keep your ThermalMLX if you want; here we keep blank unless you wire it)
        self.thermal = None
        self.thermal_gray = None

        # YOLO
        ycfg = self.cfg.get("yolo", {})
        self.detector = Detector(ycfg.get("weights", "yolov8n.pt"), float(ycfg.get("conf_threshold", 0.25)))

        # Heatmap
        hm_cfg = self.cfg.get("heatmap", {})
        self.hm = Heatmap3D(int(hm_cfg.get("grid_w", 40)), int(hm_cfg.get("grid_h", 40)), float(hm_cfg.get("decay", 0.9)))
        self._frame_count = 0

        # Logging
        self.csv_path = Path(self.cfg.get("logging", {}).get("csv_path", str(DATA_DIR / "detections.csv")))
        if not self.csv_path.is_absolute():
            self.csv_path = (REPO_ROOT / self.csv_path).resolve()
        self._csv_initialized = False

        # Cameras (Picamera2 indices)
        cams = self.cfg.get("cameras", {})
        rgb_idx = int(cams.get("rgb", {}).get("picam_index", 0))
        ir_idx  = int(cams.get("ir", {}).get("picam_index", 1))

        self.cam_rgb = PiCamStream(rgb_idx, size=(self.TW, self.TH))
        self.cam_ir  = PiCamStream(ir_idx,  size=(self.TW, self.TH))

        # FPS
        self._last_t = time.perf_counter()
        self._fps_ema = 0.0
        self._last_threat = None
        self._last_detect_time = None
        self._last_camo = False

        self.after(int(app_cfg.get("update_ms", 20)), self.loop)

    def ensure_csv(self):
        if self._csv_initialized:
            return
        ensure_dir(self.csv_path.parent)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "label", "confidence", "threat_score", "camouflage", "x1", "y1", "x2", "y2"])
        self._csv_initialized = True

    def log_detection(self, label, conf, threat_score, camouflage, bbox):
        self.ensure_csv()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            x1, y1, x2, y2 = bbox
            w.writerow([now_iso(), label, f"{conf:.4f}", f"{threat_score:.4f}", int(camouflage), x1, y1, x2, y2])

    def update_3d_heatmap_plot(self):
        self.ax3d.cla()
        self.ax3d.set_title("Threat Intensity Surface")
        self.ax3d.set_xlabel("X grid")
        self.ax3d.set_ylabel("Y grid")
        self.ax3d.set_zlabel("Intensity")
        self.ax3d.plot_surface(self.hm.X, self.hm.Y, self.hm.Z, cmap="inferno", linewidth=0, antialiased=True)
        self.ax3d.set_zlim(0, max(1.0, float(np.max(self.hm.Z)) + 0.5))
        self.canvas_fig.draw()

    def loop(self):
        if not self.running:
            return

        t0 = time.perf_counter()

        rgb = self.cam_rgb.read_bgr()
        ir  = self.cam_ir.read_bgr()

        # Warp IR -> RGB
        ir_warp = cv2.warpPerspective(ir, self.H, (self.TW, self.TH))

        # Thermal placeholder (kept black unless you integrate ThermalMLX)
        thermal_img = np.zeros_like(rgb)
        thermal_gray = None

        # YOLO on RGB
        dets = self.detector.infer(rgb)

        # Draw visuals
        vis_rgb = rgb.copy()
        vis_ir  = ir_warp.copy()
        vis_th  = thermal_img.copy()

        for d in dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(vis_ir,  (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(vis_th,  (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Heatmap + stats
        self.hm.step()
        person_count, vehicle_count, bg_count = 0, 0, 0

        best_threat = None
        best_conf = 0.0
        best_camo = False

        camo_delta = float(self.cfg.get("threat", {}).get("camouflage_thermal_delta_c", 1.5))
        log_thr = float(self.cfg.get("logging", {}).get("log_threat_threshold", 0.30))
        sigma = float(self.cfg.get("heatmap", {}).get("sigma", 2.2))

        for d in dets:
            label = d["label"].lower()
            conf = float(d["conf"])
            xyxy = d["xyxy"]

            if "person" in label:
                person_count += 1
                threat_score, camo_flag = compute_threat_and_camouflage(
                    rgb.shape, xyxy, conf, thermal_gray, camo_delta_c=camo_delta
                )

                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2.0) / self.TW
                cy = ((y1 + y2) / 2.0) / self.TH
                self.hm.add_blob(cx, cy, intensity=2.0 * threat_score, sigma=sigma)

                if best_threat is None or threat_score > best_threat:
                    best_threat = threat_score
                    best_conf = conf
                    best_camo = camo_flag

                if threat_score >= log_thr:
                    self.log_detection("person", conf, threat_score, camo_flag, xyxy)

            elif any(k in label for k in ["car", "truck", "bus", "vehicle", "motorbike"]):
                vehicle_count += 1
            else:
                bg_count += 1

        # Stats fields
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

        # Console
        self.detect_lines["Person"].config(text=f"Person: {person_count}")
        self.detect_lines["Vehicle"].config(text=f"Vehicle: {vehicle_count}")
        self.detect_lines["Background"].config(text=f"Background: {bg_count}")

        threat_alarm = float(self.cfg.get("threat", {}).get("score_threshold", 0.65))
        if best_threat is not None and best_threat >= threat_alarm:
            self.detect_lines["Alert"].config(text=f"Alert: THREAT ({best_threat:.2f})" + (" + CAMO" if best_camo else ""), fg="red")
        elif best_threat is not None and best_camo:
            self.detect_lines["Alert"].config(text="Alert: CAMOUFLAGE SUSPECTED", fg="orange")
        else:
            self.detect_lines["Alert"].config(text="Alert: --", fg="orange")

        # Update GUI panes (RGB=IMX500, IR=IMX519, Thermal=MLX/placeholder)
        self.rgb_panel.update_bgr(vis_rgb)
        self.ir_panel.update_bgr(vis_ir)
        self.th_panel.update_bgr(vis_th)

        # Heatmap update throttled
        self._frame_count += 1
        every = int(self.cfg.get("heatmap", {}).get("render_every_n_frames", 2))
        if self._frame_count % max(1, every) == 0:
            self.update_3d_heatmap_plot()

        self.after(int(self.cfg.get("app", {}).get("update_ms", 20)), self.loop)

    def on_close(self):
        self.running = False
        try:
            if self.cam_rgb:
                self.cam_rgb.close()
            if self.cam_ir:
                self.cam_ir.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = SpectraApp("data/spectra_config.yaml")
    app.mainloop()
