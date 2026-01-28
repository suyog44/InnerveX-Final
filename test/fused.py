import cv2
import numpy as np
import yaml
import json

# -----------------------------
# Load YAML + JSON
# -----------------------------
def load_stereo_yaml(path="stereo_alignment_matrix.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    H = np.array(data["homography_secondary_to_master"], dtype=np.float32)
    target_shape = data.get("target_shape", [640, 640])  # [W,H]
    return H, target_shape

def load_mlx_alignment(path="mlx_manual_align.json"):
    with open(path, "r") as f:
        d = json.load(f)
    return d.get("dx", 0), d.get("dy", 0), d.get("scale", 1.0)

# -----------------------------
# Projection functions
# -----------------------------
def project_boxes_imx500(boxes, H):
    projected = []
    for (x1, y1, x2, y2) in boxes:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, np.linalg.inv(H))  # master?secondary
        projected.append(warped.reshape(-1, 2))
    return projected

def project_boxes_mlx(boxes, dx, dy, scale):
    projected = []
    for (x1, y1, x2, y2) in boxes:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        # affine transform
        M = np.array([[scale, 0, dx], [0, scale, dy]], dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)
        warped = cv2.transform(pts, M)
        projected.append(warped.reshape(-1, 2))
    return projected

# -----------------------------
# Example usage
# -----------------------------
H, target_shape = load_stereo_yaml()
dx, dy, scale = load_mlx_alignment()

# Example detection boxes from IMX519
detections = [[100, 120, 200, 220], [300, 350, 400, 420]]

# Project to IMX500 + MLX90640
boxes_imx500 = project_boxes_imx500(detections, H)
boxes_mlx = project_boxes_mlx(detections, dx, dy, scale)

print("IMX519 detections:", detections)
print("Projected IMX500:", boxes_imx500)
print("Projected MLX90640:", boxes_mlx)