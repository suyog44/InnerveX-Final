import os
import time
import json
import cv2
import numpy as np
import yaml
import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime

from picamera2 import Picamera2

warnings.filterwarnings("ignore", category=RuntimeWarning, module="adafruit_blinka")
import board
import busio
import adafruit_mlx90640

# YOLO + drawer from your modules (same as your reference logic)
from modules.vision_yolo import YoloDetector
from modules.draw_utils import draw_boxes

# Matplotlib for 3D heatmap (embedded in Tk)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# CONFIG
# -----------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../InnerveX-Final
DATA_DIR = os.path.join(REPO_ROOT, "data")

YAML_PATHS = [
    os.path.join(DATA_DIR, "stereo_alignment_matrix.yaml"),
    os.path.join(DATA_DIR, "stereo_alignment_matrix.yaml.bak")
]

MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

THERM_MIN = 20.0
THERM_MAX = 45.0
COLORMAP = "COLORMAP_TURBO"
FLIP_V = True
FLIP_H = False

MLX_ALIGN_FILE = os.path.join(DATA_DIR, "mlx_manual_align.json")
t_dx, t_dy = 0, 0
t_scale = 1.0

STEP = 3
ZOOM_STEP = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.5

# YOLO weights priority:
# 1) models/yolo/yolov8n.pt
# 2) repo_root/yolov8n.pt
# 3) repo_root/models/yolov8n.pt
YOLO_WEIGHTS = os.path.join(REPO_ROOT, "models", "yolo", "yolov8n.pt")
if not os.path.exists(YOLO_WEIGHTS):
    alt1 = os.path.join(REPO_ROOT, "yolov8n.pt")
    alt2 = os.path.join(REPO_ROOT, "models", "yolov8n.pt")
    YOLO_WEIGHTS = alt1 if os.path.exists(alt1) else alt2

VEHICLE_KEYS = ("car", "truck", "bus", "motorbike", "motorcycle", "vehicle")

# Heatmap settings
HM_GRID_W = 40
HM_GRID_H = 40
HM_DECAY = 0.90
HM_SIGMA = 2.2
HM_RENDER_EVERY = 2  # frames


# -----------------------------
# YAML load
# -----------------------------
def load_stereo_yaml():
    for p in YAML_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = yaml.safe_load(f) or {}
            H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
            if abs(H[2, 2]) > 1e-9:
                H = H / H[2, 2]
            target_shape = data.get("target_shape", [640, 640])  # [W,H]
            W = int(target_shape[0]); Hh = int(target_shape[1])
            print(f"[INFO] Loaded stereo YAML: {p} target={W}x{Hh}")
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
    ✅ FIXED:
    Picamera2 create_preview_configuration may return a dict in some versions,
    so cfg.enable_raw(False) may not exist. We handle both.
    """
    cam = Picamera2(camera_num=cam_index)
    cfg = cam.create_preview_configuration(
        main={"size": size_wh, "format": "XBGR8888"},
        buffer_count=8
    )

    # best-effort disable raw (works for both object and dict configs)
    try:
        if hasattr(cfg, "enable_raw"):
            cfg.enable_raw(False)
        else:
            # dict-style config
            cfg["raw"] = None
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
# MLX align persistence (preserve JSON)
# -----------------------------
def load_mlx_alignment():
    global t_dx, t_dy, t_scale
    try:
        with open(MLX_ALIGN_FILE, "r") as f:
            d = json.load(f) or {}
        t_dx = int(d.get("dx", 0))
        t_dy = int(d.get("dy", 0))
        t_scale = float(d.get("scale", 1.0))
        print("[INFO] Loaded MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})
    except FileNotFoundError:
        print("[INFO] No saved MLX alignment found.")
    except Exception as e:
        print("[WARN] MLX alignment load failed:", e)

def save_mlx_alignment():
    global t_dx, t_dy, t_scale
    d = {}
    if os.path.exists(MLX_ALIGN_FILE):
        try:
            with open(MLX_ALIGN_FILE, "r") as f:
                d = json.load(f) or {}
        except Exception:
            d = {}
    d["dx"] = int(t_dx)
    d["dy"] = int(t_dy)
    d["scale"] = float(t_scale)

    os.makedirs(os.path.dirname(MLX_ALIGN_FILE), exist_ok=True)
    with open(MLX_ALIGN_FILE, "w") as f:
        json.dump(d, f, indent=2)

    print("[INFO] Saved MLX alignment:", {"dx": t_dx, "dy": t_dy, "scale": t_scale})


# -----------------------------
# Tk canvas helper (square panes with letterbox)
# -----------------------------
def letterbox_to_fit(bgr, out_w, out_h):
    h, w = bgr.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

class VideoPanel:
    def __init__(self, parent, title):
        self.frame = ttk.LabelFrame(parent, text=title)
        self.canvas = tk.Canvas(self.frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.photo = None
        self.item = None

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def update(self, bgr):
        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())
        fitted = letterbox_to_fit(bgr, cw, ch)
        rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(img)
        if self.item is None:
            self.item = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self.item, image=self.photo)


# ============================================================
# Heatmap + Threat map
# ============================================================
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
        blob = np.exp(-(dx*dx + dy*dy) / (2.0*sigma*sigma))
        self.Z += intensity * blob
        self.Z = np.clip(self.Z, 0.0, 3.0)


# -----------------------------
# NEW DASHBOARD WINDOW
# -----------------------------
class DashboardWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("SPECTRA Dashboard")
        self.geometry("980x720")
        self.configure(bg="#1e1e1e")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hm_frame = ttk.LabelFrame(self, text="3D Heatmap (Threat Intensity)")
        hm_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=hm_frame)
        self.canvas_fig.get_tk_widget().pack(fill="both", expand=True)

        conf_frame = ttk.LabelFrame(self, text="Detection Confidence (Distribution %)")
        conf_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.var_person = tk.StringVar(value="--")
        self.var_vehicle = tk.StringVar(value="--")
        self.var_background = tk.StringVar(value="--")

        def row(parent, title, var):
            r = tk.Frame(parent, bg="#1e1e1e")
            r.pack(fill="x", padx=10, pady=8)
            tk.Label(r, text=title, width=14, anchor="w",
                     fg="white", bg="#1e1e1e", font=("Segoe UI", 12)).pack(side="left")
            tk.Label(r, textvariable=var, anchor="w",
                     fg="#00ff99", bg="#1e1e1e", font=("Consolas", 14, "bold")).pack(side="left")

        row(conf_frame, "Person %", self.var_person)
        row(conf_frame, "Vehicle %", self.var_vehicle)
        row(conf_frame, "Background %", self.var_background)

        stats_frame = ttk.LabelFrame(self, text="Stats Table")
        stats_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.var_acc = tk.StringVar(value="--")
        self.var_threat = tk.StringVar(value="--")
        self.var_time = tk.StringVar(value="--")
        self.var_fps = tk.StringVar(value="--")

        row(stats_frame, "Accuracy %", self.var_acc)
        row(stats_frame, "Threat %", self.var_threat)
        row(stats_frame, "Timestamp", self.var_time)
        row(stats_frame, "FPS", self.var_fps)

        self.hm = Heatmap3D(grid_w=HM_GRID_W, grid_h=HM_GRID_H, decay=HM_DECAY)
        self._count = 0

    def update_dashboard(self, person_pct, vehicle_pct, bg_pct, acc_pct, threat_pct, timestamp, fps):
        self.var_person.set(f"{person_pct:.1f}%")
        self.var_vehicle.set(f"{vehicle_pct:.1f}%")
        self.var_background.set(f"{bg_pct:.1f}%")

        self.var_acc.set(f"{acc_pct:.1f}%")
        self.var_threat.set(f"{threat_pct:.1f}%")
        self.var_time.set(timestamp)
        self.var_fps.set(f"{fps:.1f}")

        self._count += 1
        if self._count % HM_RENDER_EVERY == 0:
            self.ax3d.cla()
            self.ax3d.set_title("Threat Surface")
            self.ax3d.plot_surface(self.hm.X, self.hm.Y, self.hm.Z, cmap="inferno",
                                   linewidth=0, antialiased=True)
            self.ax3d.set_zlim(0, max(1.0, float(np.max(self.hm.Z)) + 0.5))
            self.canvas_fig.draw()


# -----------------------------
# MAIN TK APP
# -----------------------------
class SpectraMLXYoloApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPECTRA : IMX500/IMX519 + MLX90640 + YOLO (Tk)")
        self.geometry("1400x980")

        for c in range(3):
            self.grid_columnconfigure(c, weight=1)
        for r in range(10):
            self.grid_rowconfigure(r, weight=1)

        header = tk.Label(self, text="3-View: IMX500->IMX519 | IMX519 | MLX->IMX519  + YOLO",
                          font=("Segoe UI", 16, "bold"), bg="#2b2b2b", fg="white", pady=10)
        header.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)

        self.p1 = VideoPanel(self, "IMX500 mapped -> IMX519 (YOLO)")
        self.p2 = VideoPanel(self, "IMX519 (master YOLO)")
        self.p3 = VideoPanel(self, "MLX -> IMX519 (YOLO + dx/dy/scale)")

        self.p1.grid(row=1, column=0, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.p2.grid(row=1, column=1, rowspan=6, sticky="nsew", padx=8, pady=6)
        self.p3.grid(row=1, column=2, rowspan=6, sticky="nsew", padx=8, pady=6)

        self.status = tk.StringVar(value="Initializing...")
        st = tk.Label(self, textvariable=self.status, anchor="w", bg="#2b2b2b", fg="white")
        st.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)

        self.controls = tk.Label(
            self,
            text=("Controls: H/K=left/right, I/J=up/down, +/-=zoom, 0=reset, P=save, Q=quit"),
            bg="#1e1e1e", fg="#cccccc", anchor="w"
        )
        self.controls.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)

        # Load mappings
        self.H_sec_to_master, self.TARGET_SIZE_WH = load_stereo_yaml()
        self.TW, self.TH = self.TARGET_SIZE_WH

        # Open cameras
        info = list_cameras()
        idx_master = pick_camera_index(info, MASTER_NAME)
        idx_secondary = pick_camera_index(info, SECONDARY_NAME)
        if idx_master is None or idx_secondary is None:
            print("\n[WARN] Could not auto-detect IMX519/IMX500 by name. Falling back to 0=master, 1=secondary")
            idx_master, idx_secondary = 0, 1

        self.cam_master = open_camera(idx_master, self.TARGET_SIZE_WH)
        self.cam_secondary = open_camera(idx_secondary, self.TARGET_SIZE_WH)

        # MLX init
        self.status.set("Initializing MLX90640...")
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

        load_mlx_alignment()

        # YOLO init
        self.status.set(f"Loading YOLO weights: {YOLO_WEIGHTS}")
        self.detector = YoloDetector(YOLO_WEIGHTS)

        # Dashboard window
        self.dashboard = DashboardWindow(self)

        self.bind("<KeyPress>", self.on_key)
        self._fps_ema = 0.0
        self.running = True
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.after(10, self.loop)

    def on_key(self, event):
        global t_dx, t_dy, t_scale
        k = event.keysym.lower()

        if k == "q":
            self.on_close()
            return

        if k == "h":
            t_dx -= STEP
        elif k == "k":
            t_dx += STEP
        elif k == "i":
            t_dy -= STEP
        elif k == "j":
            t_dy += STEP
        elif k in ("plus", "equal"):
            t_scale = min(SCALE_MAX, t_scale + ZOOM_STEP)
        elif k in ("minus", "underscore"):
            t_scale = max(SCALE_MIN, t_scale - ZOOM_STEP)
        elif k == "0":
            t_dx, t_dy, t_scale = 0, 0, 1.0
        elif k == "p":
            save_mlx_alignment()

    def loop(self):
        if not self.running:
            return

        t0 = time.time()

        master = capture_bgr(self.cam_master)        # IMX519
        secondary = capture_bgr(self.cam_secondary)  # IMX500

        master = cv2.resize(master, (self.TW, self.TH), interpolation=cv2.INTER_LINEAR)
        secondary = cv2.resize(secondary, (self.TW, self.TH), interpolation=cv2.INTER_LINEAR)

        warped_secondary = cv2.warpPerspective(
            secondary, self.H_sec_to_master, (self.TW, self.TH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # MLX frame
        try:
            self.mlx.getFrame(self.buf)
        except Exception as e:
            self.status.set(f"MLX read error: {e}")
            self.after(30, self.loop)
            return

        raw = np.array(self.buf, dtype=np.float32).reshape(24, 32)
        if FLIP_V:
            raw = np.flip(raw, axis=0)
        if FLIP_H:
            raw = np.flip(raw, axis=1)

        u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
        up = cv2.resize(u8, (self.TW, self.TH), interpolation=cv2.INTER_CUBIC)
        thermal = apply_colormap(up, COLORMAP)
        thermal_moved = apply_thermal_transform(thermal, t_dx, t_dy, t_scale)

        # YOLO detection on master (your existing behavior)
        boxes = self.detector.infer(master)

        vis_master = draw_boxes(master.copy(), boxes)
        vis_warped = draw_boxes(warped_secondary.copy(), boxes)
        vis_thermal = draw_boxes(thermal_moved.copy(), boxes)

        # FPS
        fps_inst = 1.0 / max(1e-6, (time.time() - t0))
        self._fps_ema = self._fps_ema * 0.8 + fps_inst * 0.2

        # Categorize detections into person/vehicle/background
        total = len(boxes) if boxes else 0
        person_n = vehicle_n = bg_n = 0
        person_confs = []
        best_threat = 0.0

        # update dashboard heatmap
        self.dashboard.hm.step()

        if boxes:
            for b in boxes:
                # try dict-based format first
                if isinstance(b, dict):
                    lbl = str(b.get("label", b.get("class", ""))).lower()
                    conf = float(b.get("conf", b.get("confidence", 0.0)))
                    x1, y1, x2, y2 = b.get("xyxy", b.get("bbox", [0, 0, 1, 1]))
                else:
                    # fallback list/tuple
                    try:
                        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                        conf = float(b[4]) if len(b) > 4 else 0.0
                        lbl = str(b[5]).lower() if len(b) > 5 else ""
                    except Exception:
                        continue

                if "person" in lbl:
                    person_n += 1
                    person_confs.append(conf)
                    intensity = 2.0 * conf
                elif any(k in lbl for k in VEHICLE_KEYS):
                    vehicle_n += 1
                    intensity = 1.4 * conf
                else:
                    bg_n += 1
                    intensity = 0.9 * conf

                # best threat = max confidence (simple threat proxy)
                best_threat = max(best_threat, conf)

                # heatmap add blob
                cx = ((float(x1) + float(x2)) * 0.5) / self.TW
                cy = ((float(y1) + float(y2)) * 0.5) / self.TH
                self.dashboard.hm.add_blob(cx, cy, intensity=intensity, sigma=HM_SIGMA)

        if total > 0:
            person_pct = 100.0 * person_n / total
            vehicle_pct = 100.0 * vehicle_n / total
            bg_pct = 100.0 * bg_n / total
        else:
            person_pct = vehicle_pct = bg_pct = 0.0

        acc_pct = (float(np.mean(person_confs)) * 100.0) if person_confs else 0.0
        threat_pct = best_threat * 100.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.status.set(
            f"FPS {self._fps_ema:.1f} | total={total} person={person_n} vehicle={vehicle_n} bg={bg_n} | "
            f"dx={t_dx} dy={t_dy} scale={t_scale:.2f}"
        )

        # update dashboard
        if self.dashboard.winfo_exists():
            self.dashboard.update_dashboard(
                person_pct, vehicle_pct, bg_pct,
                acc_pct, threat_pct, ts, self._fps_ema
            )

        # labels on top
        def label(img, text):
            out = img.copy()
            cv2.rectangle(out, (0, 0), (self.TW, 34), (0, 0, 0), -1)
            cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return out

        panel1 = label(vis_warped, "IMX500 mapped -> IMX519")
        panel2 = label(vis_master, "IMX519 (master) YOLO")
        panel3 = label(vis_thermal, f"MLX -> IMX519 (dx={t_dx}, dy={t_dy}, s={t_scale:.2f})")

        self.p1.update(panel1)
        self.p2.update(panel2)
        self.p3.update(panel3)

        self.after(10, self.loop)

    def on_close(self):
        self.running = False
        try:
            self.cam_master.stop()
            self.cam_secondary.stop()
            self.cam_master.close()
            self.cam_secondary.close()
        except Exception:
            pass
        try:
            if self.dashboard and self.dashboard.winfo_exists():
                self.dashboard.destroy()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = SpectraMLXYoloApp()
    app.mainloop()
