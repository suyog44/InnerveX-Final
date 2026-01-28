import numpy as np
import cv2

def warp_perspective(img, H, size):
    return cv2.warpPerspective(img, H, size)

def affine_warp(gray, dx, dy, scale):
    h, w = gray.shape[:2]
    cx, cy = w / 2, h / 2
    M = np.array([
        [scale, 0, (1 - scale) * cx + dx],
        [0, scale, (1 - scale) * cy + dy]
    ], np.float32)
    return cv2.warpAffine(gray, M, (w, h))
