import cv2
import yaml
import numpy as np
import time
from datetime import datetime
import cvzone  # pip install cvzone

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

# Load homography matrix
with open("data/stereo_alignment_matrix.yaml") as f:
    H = yaml.safe_load(f)["homography_secondary_to_master"]
H = np.array(H, dtype=np.float32)

TW, TH = 640, 640

# Camera setup
info = list_cameras()
idx_master = pick_camera_num(info, "imx519")
idx_secondary = pick_camera_num(info, "imx500")

cam_master, cam_secondary = open_dual_cameras(idx_master, idx_secondary, (TW, TH))

# Thermal and detector setup
thermal = ThermalMLX("data/mlx_manual_align.json", (TW, TH))
detector = YoloDetector("yolov8n.pt")

# FPS calculation
prev_time = time.time()

while True:
    master = capture_bgr(cam_master)
    secondary = capture_bgr(cam_secondary)

    warped_secondary = warp_perspective(secondary, H, (TW, TH))
    thermal_gray = thermal.read_aligned_gray()
    thermal_img = thermal.colorize(thermal_gray)

    boxes = detector.infer(master)

    # Robust filtering
    filtered_boxes = []
    for box in boxes:
        label = box.get("label") or box.get("class") or box.get("name")
        if label is None:
            filtered_boxes.append(box)
            continue
        if label.lower() not in ["cat", "toilet"]:
            filtered_boxes.append(box)

    vis1 = draw_boxes(master.copy(), filtered_boxes)
    vis2 = draw_boxes(warped_secondary.copy(), filtered_boxes)
    vis3 = draw_boxes(thermal_img.copy(), filtered_boxes)

    out = cv2.hconcat([vis2, vis1, vis3])

    # ---------------- Overlay Info ----------------
    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Example threat score & accuracy (replace with your logic)
    threat_score = np.random.uniform(0, 1)  # dummy score
    accuracy = np.random.uniform(0.8, 1.0)  # dummy accuracy

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use cvzone for nice display
    cvzone.putTextRect(out, f"Threat Score: {threat_score:.2f}", (20, 40), scale=1.2, thickness=2)
    cvzone.putTextRect(out, f"Accuracy: {accuracy:.2f}", (20, 80), scale=1.2, thickness=2)
    cvzone.putTextRect(out, f"FPS: {fps:.1f}", (20, 120), scale=1.2, thickness=2)
    cvzone.putTextRect(out, f"Time: {timestamp}", (20, 160), scale=1.2, thickness=2)

    cv2.imshow("Multispectral Fusion App", out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()