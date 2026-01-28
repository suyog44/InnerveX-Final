import cv2
import numpy as np

# -----------------------------
# Draw boxes utility
# -----------------------------
def draw_boxes(img, boxes, color=(0,255,0), label=""):
    out = img.copy()
    for box in boxes:
        # box is either [x1,y1,x2,y2] or 4 corner points
        pts = np.array(box, dtype=np.int32).reshape(-1,2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    if label:
        cv2.putText(out, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out

# -----------------------------
# Example fusion
# -----------------------------
# Assume you already have frames:
# master_frame (IMX519), secondary_frame (IMX500 warped), thermal_frame (MLX90640 aligned)

detections_master = [[100,120,200,220],[300,350,400,420]]
boxes_imx500 = [np.array([[139.8,316.3],[181.8,317.4],[180.4,357.8],[138.8,356.8]])]
boxes_mlx = [np.array([[344,236.4],[496,236.4],[496,388.4],[344,388.4]])]

# Draw boxes on each view
panel_master   = draw_boxes(master_frame, detections_master, color=(0,255,0), label="IMX519 (master)")
panel_secondary= draw_boxes(secondary_frame, boxes_imx500, color=(255,0,0), label="IMX500 mapped")
panel_thermal  = draw_boxes(thermal_frame, boxes_mlx, color=(0,0,255), label="MLX90640 aligned")

# Fuse horizontally
combined = cv2.hconcat([panel_master, panel_secondary, panel_thermal])

cv2.imshow("Fused Views", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()