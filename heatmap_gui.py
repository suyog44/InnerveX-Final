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

# Spectrum source: "heatmap" or "overlay"
SPECTRUM_SOURCE = "heatmap"   # change to "overlay" if you want spectrum of final displayed image


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
    cfg = cam.create_preview_configuration(
        main={"size": size_wh, "format": "XBGR8888"},
        buffer_count=8
    )

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
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )


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


# -----------------------------
# Spectrum (RGB + Gray) helpers
# -----------------------------
def compute_histograms_rgb_gray(bgr_img):
    """
    Returns normalized histograms (0..1) for R, G, B, Gray (each length 256).
    bgr_img: uint8 BGR
    """
    b = bgr_img[:, :, 0]
    g = bgr_img[:, :, 1]
    r = bgr_img[:, :, 2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    hb = np.bincount(b.ravel(), minlength=256).astype(np.float32)
    hg = np.bincount(g.ravel(), minlength=256).astype(np.float32)
    hr = np.bincount(r.ravel(), minlength=256).astype(np.float32)
    hy = np.bincount(gray.ravel(), minlength=256).astype(np.float32)

    def norm(h):
        m = float(h.max()) + 1e-9
        return h / m

    return norm(hr), norm(hg), norm(hb), norm(hy)

def draw_spectrum_on_canvas(canvas, w, h, hr, hg, hb, hy):
    """
    Draw spectrum curves on Tk canvas.
    """
    canvas.delete("all")

    # Grid
    grid_color = "#222222"
    for i in range(1, 4):
        y = int(i * h / 4)
        canvas.create_line(0, y, w, y, fill=grid_color)
    for i in range(1, 8):
        x = int(i * w / 8)
        canvas.create_line(x, 0, x, h, fill=grid_color)

    def poly_points(hist):
        pts = []
        for x in range(256):
            xx = int(x * (w - 1) / 255)
            yy = int((1.0 - float(hist[x])) * (h - 1))
            pts.extend([xx, yy])
        return pts

    # Curves: R, G, B, Gray
    canvas.create_line(poly_points(hr), fill="red",   width=2)
    canvas.create_line(poly_points(hg), fill="green", width=2)
    canvas.create_line(poly_points(hb), fill="blue",  width=2)
    canvas.create_line(poly_points(hy), fill="gray",  width=2)


# -----------------------------
# GUI APP
# -----------------------------
class FusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fused Heatmap GUI + Spectrometer")

        # load calibration
        self.H_sec_to_master, self.TARGET_SIZE_WH = load_stereo_yaml_optional(default_wh=(640, 640))
        self.TW, self.TH = self.TARGET_SIZE_WH

        # open cameras
        info = list_cameras()
        idx_master = pick_camera_index(info, MASTER_NAME)
        idx_secondary = pick_camera_index(info, SECONDARY_NAME)

        if idx_master is None or idx_secondary is None:
            print("\n[WARN] Could not auto-detect IMX519/IMX500 by name. Falling back to 0=master, 1=secondary")
            idx_master = 0
            idx_secondary = 1

        print(f"\n[INFO] MASTER (IMX519) index={idx_master}")
        print(f"[INFO] SECONDARY (IMX500) index={idx_secondary}")

        self.cam_master = open_camera(idx_master, self.TARGET_SIZE_WH)
        self.cam_secondary = open_camera(idx_secondary, self.TARGET_SIZE_WH)

        # init MLX
        print("[INFO] Initializing MLX90640...")
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

        load_mlx_alignment()

        # state
        self.fps = 0.0
        self.running = True

        # ---- Layout (3 columns x 2 rows) ----
        # Col0 spans both rows: image
        self.img_label = ttk.Label(root)
        self.img_label.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=6, pady=6)

        # Col1 Row0: RGB ratio
        rgb_frame = ttk.LabelFrame(root, text="RGB Ratio (%)")
        rgb_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self.rgb_var = tk.StringVar(value="R: -- %   G: -- %   B: -- %")
        self.rgb_label = ttk.Label(rgb_frame, textvariable=self.rgb_var, font=("Segoe UI", 14))
        self.rgb_label.pack(anchor="center", pady=16, padx=10)

        self.align_var = tk.StringVar(value="MLX Align: dx=0 dy=0 s=1.00")
        ttk.Label(rgb_frame, textvariable=self.align_var).pack(anchor="center", pady=6)

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(rgb_frame, textvariable=self.fps_var).pack(anchor="center", pady=6)

        # Col1 Row1: Min/Max temp
        temp_frame = ttk.LabelFrame(root, text="MLX Temperature (°C)")
        temp_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        self.temp_var = tk.StringVar(value="Min: -- °C   Max: -- °C")
        self.temp_label = ttk.Label(temp_frame, textvariable=self.temp_var, font=("Segoe UI", 14))
        self.temp_label.pack(anchor="center", pady=16, padx=10)

        btns = ttk.Frame(temp_frame)
        btns.pack(pady=10)

        ttk.Button(btns, text="Reset (0)", command=self.reset_align).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Save (P)", command=self.save_align).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Quit (Q)", command=self.on_close).grid(row=0, column=2, padx=5)

        hint = "Keys: H/K/I/J move  |  +/- zoom  |  0 reset  |  P save  |  Q quit"
        ttk.Label(temp_frame, text=hint).pack(anchor="center", pady=6)

        # Col2 Row1: Spectrum
        spec_frame = ttk.LabelFrame(root, text="Spectrometer (R/G/B/Gray)")
        spec_frame.grid(row=1, column=2, sticky="nsew", padx=6, pady=6)

        self.spec_w = 360
        self.spec_h = 220
        self.spec_canvas = tk.Canvas(spec_frame, width=self.spec_w, height=self.spec_h, bg="black", highlightthickness=0)
        self.spec_canvas.pack(fill="both", expand=True)

        legend = ttk.Frame(spec_frame)
        legend.pack(fill="x", pady=4)
        ttk.Label(legend, text="R", foreground="red").pack(side="left", padx=8)
        ttk.Label(legend, text="G", foreground="green").pack(side="left", padx=8)
        ttk.Label(legend, text="B", foreground="blue").pack(side="left", padx=8)
        ttk.Label(legend, text="Gray", foreground="gray").pack(side="left", padx=8)

        # grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=4)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)

        # key bindings
        root.bind("<KeyPress>", self.on_key)

        # close handler
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # start loop
        self.update_loop()


    # ---- UI actions ----
    def reset_align(self):
        global t_dx, t_dy, t_scale
        t_dx, t_dy, t_scale = 0, 0, 1.0

    def save_align(self):
        save_mlx_alignment()

    def on_close(self):
        self.running = False
        try:
            self.cam_master.stop()
            self.cam_secondary.stop()
            self.cam_master.close()
            self.cam_secondary.close()
        except Exception:
            pass
        self.root.destroy()

    def on_key(self, event):
        global t_dx, t_dy, t_scale
        k = event.keysym.lower()

        if k == "q":
            self.on_close()
        elif k == "h":
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
            self.reset_align()
        elif k == "p":
            self.save_align()

    # ---- compute + render loop ----
    def update_loop(self):
        if not self.running:
            return

        t0 = time.time()

        try:
            master = capture_bgr(self.cam_master)
            secondary = capture_bgr(self.cam_secondary)

            master = cv2.resize(master, (self.TW, self.TH), interpolation=cv2.INTER_LINEAR)
            secondary = cv2.resize(secondary, (self.TW, self.TH), interpolation=cv2.INTER_LINEAR)

            # warp IMX500 -> IMX519 using stereo YAML (identity if missing)
            warped_secondary = cv2.warpPerspective(
                secondary, self.H_sec_to_master, (self.TW, self.TH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # MLX frame
            self.mlx.getFrame(self.buf)
            raw = np.array(self.buf, dtype=np.float32).reshape(24, 32)

            if FLIP_V:
                raw = np.flip(raw, axis=0)
            if FLIP_H:
                raw = np.flip(raw, axis=1)

            # min/max temperature (°C)
            tmin = float(np.min(raw))
            tmax = float(np.max(raw))

            # MLX -> uint8 map based on THERM_MIN/MAX
            u8 = norm_thermal(raw, THERM_MIN, THERM_MAX)
            up = cv2.resize(u8, (self.TW, self.TH), interpolation=cv2.INTER_CUBIC)

            # apply MLX manual alignment
            global t_dx, t_dy, t_scale
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

            # GREEN->RED heatmap and overlay on master
            fused_color = heatmap_green_red(fused_u8)          # BGR
            fused_overlay = overlay(master, fused_color, OVERLAY_ALPHA)  # BGR

            # ---- RGB ratio (%) from heatmap image ----
            rgb = cv2.cvtColor(fused_color, cv2.COLOR_BGR2RGB).astype(np.float32)
            sums = rgb.reshape(-1, 3).sum(axis=0)  # [Rsum, Gsum, Bsum]
            total = float(np.sum(sums)) + 1e-9
            r_pct = 100.0 * float(sums[0]) / total
            g_pct = 100.0 * float(sums[1]) / total
            b_pct = 100.0 * float(sums[2]) / total

            # ---- FPS ----
            fps_inst = 1.0 / max(1e-6, (time.time() - t0))
            self.fps = self.fps * 0.8 + fps_inst * 0.2

            # ---- Update text panels ----
            self.rgb_var.set(f"R: {r_pct:5.1f} %   G: {g_pct:5.1f} %   B: {b_pct:5.1f} %")
            self.temp_var.set(f"Min: {tmin:5.1f} °C   Max: {tmax:5.1f} °C")
            self.align_var.set(f"MLX Align: dx={t_dx}  dy={t_dy}  s={t_scale:.2f}")
            self.fps_var.set(f"FPS: {self.fps:.1f} | IMX500 mean={sec_mean:.3f} dark={sec_dark:.3f}")

            # ---- Spectrometer ----
            if SPECTRUM_SOURCE.lower() == "overlay":
                spectrum_src = fused_overlay
            else:
                spectrum_src = fused_color  # default = heatmap

            hr, hg, hb, hy = compute_histograms_rgb_gray(spectrum_src)
            draw_spectrum_on_canvas(self.spec_canvas, self.spec_w, self.spec_h, hr, hg, hb, hy)

            # ---- Show image in Tkinter (left big panel) ----
            disp_rgb = cv2.cvtColor(fused_overlay, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(disp_rgb)
            imgtk = ImageTk.PhotoImage(image=pil)

            # keep reference to avoid GC
            self.img_label.imgtk = imgtk
            self.img_label.configure(image=imgtk)

        except Exception as e:
            self.fps_var.set(f"Error: {e}")

        # schedule next update (1ms is aggressive; increase if CPU usage is high)
        self.root.after(1, self.update_loop)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FusionApp(root)
    root.mainloop()
