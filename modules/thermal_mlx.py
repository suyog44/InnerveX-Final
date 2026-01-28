import json
import numpy as np
import board
import busio
import adafruit_mlx90640
from .geometry import affine_warp
import cv2

class ThermalMLX:
    def __init__(self, data_file, size, tmin=20.0, tmax=45.0):
        self.size = size
        self.tmin = tmin
        self.tmax = tmax
        self.dx, self.dy, self.scale = 0, 0, 1.0

        if data_file:
            self.load_alignment(data_file)

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.buf = [0.0] * 768

    def load_alignment(self, path):
        try:
            with open(path) as f:
                d = json.load(f)
                self.dx = d.get("dx", 0)
                self.dy = d.get("dy", 0)
                self.scale = d.get("scale", 1.0)
        except Exception:
            pass

    def read_aligned_gray(self):
        self.mlx.getFrame(self.buf)
        raw = np.array(self.buf, np.float32).reshape(24, 32)
        raw = np.flip(raw, axis=0)
        raw = np.flip(raw, axis=1)
        gray = np.clip((raw - self.tmin) / (self.tmax - self.tmin), 0, 1)
        gray = (gray * 255).astype(np.uint8)
        gray_up = cv2.resize(gray, self.size, interpolation=cv2.INTER_CUBIC)
        return affine_warp(gray_up, self.dx, self.dy, self.scale)

    def colorize(self, gray):
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
