import cv2
import yaml
import numpy as np

from modules.camera_manager import (
    list_cameras,
    pick_camera_num,
    open_dual_cameras,
    capture_bgr
)
from modules.geometry import warp_perspective
from modules.thermal_mlx import ThermalMLX
from modules.vision_yolo import YoloDetector
from modules.draw_utils import draw_boxes

with open("data/stereo_alignment_matrix.yaml") as f:
    H = yaml.safe_load(f)["homography_secondary_to_master"]
H = np.array(H, dtype=np.float32)

TW, TH = 640, 640

info = list_cameras()
idx_master = pick_camera_num(info, "imx519")
idx_secondary = pick_camera_num(info, "imx500")

cam_master, cam_secondary = open_dual_cameras(idx_master, idx_secondary, (TW, TH))

thermal = ThermalMLX("data/mlx_manual_align.json", (TW, TH))
detector = YoloDetector("yolov8n.pt")

while True:
    master = capture_bgr(cam_master)
    secondary = capture_bgr(cam_secondary)

    warped_secondary = warp_perspective(secondary, H, (TW, TH))
    thermal_gray = thermal.read_aligned_gray()
    thermal_img = thermal.colorize(thermal_gray)

    boxes = detector.infer(master)

    vis1 = draw_boxes(master.copy(), boxes)
    vis2 = draw_boxes(warped_secondary.copy(), boxes)
    vis3 = draw_boxes(thermal_img.copy(), boxes)

    out = cv2.hconcat([vis2, vis1, vis3])
    cv2.imshow("Multispectral Fusion App", out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
