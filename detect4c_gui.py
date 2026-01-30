#!/usr/bin/env python3
import os
import time
import json
import cv2
import yaml
import warnings
import tkinter as tk
from tkinter import ttk
import numpy as np

from PIL import Image, ImageTk
from picamera2 import Picamera2

warnings.filterwarnings("ignore", category=RuntimeWarning)

import board
import busio
import adafruit_mlx90640

# ================= FILE PATHS =================
STEREO_YAML = "/home/pi/InnerveX-Final/data/stereo_alignment_matrix.yaml"
MLX_JSON    = "/home/pi/InnerveX-Final/data/mlx_manual_align.json"

MASTER_NAME = "imx519"
SECONDARY_NAME = "imx500"

# ================= FIXED DISPLAY SIZES =================
COL1_W, COL1_H = 320, 240      # Column 1 fixed
COL2_W, COL2_H = 160, 120      # Column 2 fixed (tight stack, no gaps)
COL4_W, COL4_H = 520, 420      # Column 4 fixed

MODEL_TEXT = "X-factor Trained YOLO RGBT Model"

THERM_MIN, THERM_MAX = 20.0, 45.0
FLIP_V, FLIP_H = True, False

# Fusion weights
ALPHA_THERMAL = 0.40
ALPHA_IMX500  = 0.25


# ================= HELPERS =================
def load_yaml():
    """Load homography + target_shape; fallback identity."""
    if not os.path.exists(STEREO_YAML):
        print(f"[WARN] Missing YAML: {STEREO_YAML} -> identity")
        return np.eye(3, dtype=np.float32), (640, 480)

    y = yaml.safe_load(open(STEREO_YAML)) or {}
    H_list = (y.get("homography_secondary_to_master")
              or y.get("H_secondary_to_master")
              or y.get("H_sec_to_master")
              or y.get("homography"))

    if H_list is None:
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.array(H_list, dtype=np.float32).reshape(3, 3)

    W, Hh = y.get("target_shape", [640, 480])
    return H, (int(W), int(Hh))


def load_mlx():
    """Load dx/dy/scale; fallback defaults."""
    if not os.path.exists(MLX_JSON):
        print(f"[WARN] Missing MLX JSON: {MLX_JSON} -> defaults")
        return 0, 0, 1.0
    j = json.load(open(MLX_JSON))
    return int(j.get("dx", 0)), int(j.get("dy", 0)), float(j.get("scale", 1.0))


def ensure_bgr(frame):
    """
    Picamera2 XBGR8888 -> typically BGRA array.
    Convert to 3-channel BGR.
    """
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape[2] == 3:
            return frame
    return frame[:, :, :3].copy()


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


def open_cam(idx, size):
    cam = Picamera2(camera_num=idx)
    cfg = cam.create_preview_configuration(main={"size": size, "format": "XBGR8888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)
    return cam


def normalize_thermal(t):
    t = np.clip(t, THERM_MIN, THERM_MAX)
    return ((t - THERM_MIN) / (THERM_MAX - THERM_MIN) * 255).astype(np.uint8)


def warp_affine(img, dx, dy, s):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = np.array([[s, 0, (1 - s) * cx + dx],
                  [0, s, (1 - s) * cy + dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def put_title(img, text):
    img = ensure_bgr(img)
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


def to_tk(img_bgr, w, h):
    """Fixed-size conversion (no auto-fit)."""
    img_bgr = ensure_bgr(img_bgr)
    if img_bgr is None:
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ================= APP =================
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("InnerveX Multispectral Dashboard")
        self.configure(bg="#f5f7fa")

        # ---- NO RESIZE WINDOW PART ----
        self.geometry("1900x950")
        self.resizable(False, False)

        self.H, (self.TW, self.TH) = load_yaml()
        self.dx, self.dy, self.sc = load_mlx()

        # Pick cameras by name if possible; fallback to 0/1
        info = list_cameras()
        idx_master = pick_camera_index(info, MASTER_NAME)
        idx_secondary = pick_camera_index(info, SECONDARY_NAME)
        if idx_master is None or idx_secondary is None:
            print("[WARN] Could not match imx519/imx500 by name. Using indices 0/1.")
            idx_master, idx_secondary = 0, 1

        self.camM = open_cam(idx_master, (self.TW, self.TH))
        self.camS = open_cam(idx_secondary, (self.TW, self.TH))

        # MLX init
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

        self.fps = 0.0
        self.build_ui()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(1, self.update_loop)

    def build_ui(self):
        # Main layout grid
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        for c in range(4):
            self.grid_columnconfigure(c, weight=0)

        hdr = dict(font=("Segoe UI", 12, "bold"), bg="#1f3a5f", fg="white", pady=8)

        tk.Label(self, text="Data Feed", **hdr).grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        tk.Label(self, text="Preprocessing", **hdr).grid(row=0, column=1, sticky="ew", padx=6, pady=0)
        tk.Label(self, text=MODEL_TEXT, **hdr).grid(row=0, column=2, sticky="ew", padx=6, pady=6)
        tk.Label(self, text="Fused Output", **hdr).grid(row=0, column=3, sticky="ew", padx=6, pady=6)

        # ===== Column 1 (Data Feed) =====
        col1 = tk.Frame(self, bg="#f5f7fa")
        col1.grid(row=1, column=0, sticky="n", padx=6, pady=6)

        self.lf_imx500 = tk.LabelFrame(col1, text="IMX500 (Raw Feed)", bg="white", fg="#1f3a5f",
                                       font=("Segoe UI", 10, "bold"))
        self.lf_imx500.pack(pady=6)
        self.l_imx500 = tk.Label(self.lf_imx500, bg="#eaeaea")
        self.l_imx500.pack()

        self.lf_imx519 = tk.LabelFrame(col1, text="IMX519 (Raw Feed)", bg="white", fg="#1f3a5f",
                                       font=("Segoe UI", 10, "bold"))
        self.lf_imx519.pack(pady=6)
        self.l_imx519 = tk.Label(self.lf_imx519, bg="#eaeaea")
        self.l_imx519.pack()

        self.lf_mlxraw = tk.LabelFrame(col1, text="MLX90640 (Thermal Feed)", bg="white", fg="#1f3a5f",
                                       font=("Segoe UI", 10, "bold"))
        self.lf_mlxraw.pack(pady=6)
        self.l_mlxraw = tk.Label(self.lf_mlxraw, bg="#eaeaea")
        self.l_mlxraw.pack()

        # ===== Column 2 (Preprocessing) — NO GAPS =====
        # Put all tiles in one vertical pack stack => no row expansion gaps possible.
        col2 = tk.Frame(self, bg="#f5f7fa")
        col2.grid(row=1, column=1, sticky="n", padx=6, pady=0)

        def compact_tile(title):
            lf = tk.LabelFrame(col2, text=title, bg="white", fg="#1f3a5f",
                               font=("Segoe UI", 9, "bold"), padx=0, pady=0)
            lbl = tk.Label(lf, bg="#eaeaea")
            lbl.pack()
            return lf, lbl

        self.lf_red, self.l_red = compact_tile("IMX500 Red (mapped)")
        self.lf_green, self.l_green = compact_tile("IMX500 Green (mapped)")
        self.lf_blue, self.l_blue = compact_tile("IMX500 Blue (mapped)")
        self.lf_gray, self.l_gray = compact_tile("IMX519 Gray")
        self.lf_therm, self.l_therm = compact_tile("MLX Thermal (aligned)")

        # Pack with pady=0 => ZERO vertical gaps
        for lf in [self.lf_red, self.lf_green, self.lf_blue, self.lf_gray, self.lf_therm]:
            lf.pack(side="top", pady=0, padx=0, anchor="n")

        # ===== Column 3 (Model text only) =====
        col3 = tk.LabelFrame(self, text="Model", bg="white", fg="#1f3a5f",
                             font=("Segoe UI", 10, "bold"), padx=10, pady=10)
        col3.grid(row=1, column=2, sticky="n", padx=6, pady=6)

        model_label = tk.Label(col3, text=MODEL_TEXT, bg="white", fg="#1f3a5f",
                               font=("Segoe UI", 18, "bold"),
                               wraplength=380, justify="center")
        model_label.pack(expand=True, fill="both")

        # ===== Column 4 (Fused output) =====
        col4 = tk.LabelFrame(self, text="Fused Stream", bg="white", fg="#1f3a5f",
                             font=("Segoe UI", 10, "bold"))
        col4.grid(row=1, column=3, sticky="n", padx=6, pady=6)

        self.l_fused = tk.Label(col4, bg="#eaeaea")
        self.l_fused.pack()

        # Status bar
        self.status = ttk.Label(self, text="FPS: --", anchor="w")
        self.status.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 8))

    def update_loop(self):
        t0 = time.time()
        try:
            imx519_raw = ensure_bgr(self.camM.capture_array())
            imx500_raw = ensure_bgr(self.camS.capture_array())
            if imx519_raw is None or imx500_raw is None:
                self.after(30, self.update_loop)
                return

            warped_500 = cv2.warpPerspective(
                imx500_raw, self.H, (self.TW, self.TH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            warped_500 = ensure_bgr(warped_500)

            b, g, r = cv2.split(warped_500)
            red_img   = cv2.merge([np.zeros_like(b), np.zeros_like(b), r])
            green_img = cv2.merge([np.zeros_like(b), g, np.zeros_like(b)])
            blue_img  = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])

            gray = cv2.cvtColor(imx519_raw, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            self.mlx.getFrame(self.buf)
            raw = np.array(self.buf, dtype=np.float32).reshape(24, 32)
            if FLIP_V:
                raw = np.flip(raw, axis=0)
            if FLIP_H:
                raw = np.flip(raw, axis=1)

            therm_u8 = normalize_thermal(raw)
            therm_u8 = cv2.resize(therm_u8, (self.TW, self.TH), interpolation=cv2.INTER_CUBIC)
            therm_u8 = warp_affine(therm_u8, self.dx, self.dy, self.sc)
            therm_color = cv2.applyColorMap(therm_u8, cv2.COLORMAP_TURBO)

            fused = cv2.addWeighted(imx519_raw, 1.0 - ALPHA_THERMAL, therm_color, ALPHA_THERMAL, 0)
            fused = cv2.addWeighted(fused, 1.0 - ALPHA_IMX500, warped_500, ALPHA_IMX500, 0)

            # Titles
            imx500_show = put_title(imx500_raw, "IMX500 RAW")
            imx519_show = put_title(imx519_raw, "IMX519 RAW")
            therm_show  = put_title(therm_color, "MLX THERMAL (aligned)")
            fused_show  = put_title(fused, "FUSED: IMX519 + IMX500(mapped) + MLX")

            red_show   = put_title(red_img, "RED (mapped)")
            green_show = put_title(green_img, "GREEN (mapped)")
            blue_show  = put_title(blue_img, "BLUE (mapped)")
            gray_show  = put_title(gray_bgr, "IMX519 GRAY")

            # Column 1 (fixed)
            img1 = to_tk(imx500_show, COL1_W, COL1_H)
            self.l_imx500.configure(image=img1); self.l_imx500.image = img1

            img2 = to_tk(imx519_show, COL1_W, COL1_H)
            self.l_imx519.configure(image=img2); self.l_imx519.image = img2

            img3 = to_tk(therm_show, COL1_W, COL1_H)
            self.l_mlxraw.configure(image=img3); self.l_mlxraw.image = img3

            # Column 2 (fixed, tight stack)
            imgR = to_tk(red_show, COL2_W, COL2_H)
            self.l_red.configure(image=imgR); self.l_red.image = imgR

            imgG = to_tk(green_show, COL2_W, COL2_H)
            self.l_green.configure(image=imgG); self.l_green.image = imgG

            imgB = to_tk(blue_show, COL2_W, COL2_H)
            self.l_blue.configure(image=imgB); self.l_blue.image = imgB

            imgY = to_tk(gray_show, COL2_W, COL2_H)
            self.l_gray.configure(image=imgY); self.l_gray.image = imgY

            imgT = to_tk(therm_show, COL2_W, COL2_H)
            self.l_therm.configure(image=imgT); self.l_therm.image = imgT

            # Column 4 (fixed)
            imgF = to_tk(fused_show, COL4_W, COL4_H)
            self.l_fused.configure(image=imgF); self.l_fused.image = imgF

        except Exception as e:
            print("[WARN] update_loop error:", repr(e))

        dt = max(1e-6, time.time() - t0)
        fps_inst = 1.0 / dt
        self.fps = 0.85 * self.fps + 0.15 * fps_inst
        self.status.configure(
            text=f"FPS: {self.fps:.1f} | Stereo: {os.path.basename(STEREO_YAML)} | MLX: {os.path.basename(MLX_JSON)}"
        )

        self.after(30, self.update_loop)

    def on_close(self):
        try:
            self.camM.stop(); self.camM.close()
            self.camS.stop(); self.camS.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
