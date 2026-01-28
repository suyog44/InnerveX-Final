def fuse_boxes_with_thermal(boxes, thermal_gray, ambient, delta=3.0):
    fused = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        roi = thermal_gray[y1:y2, x1:x2]
        mean_temp = ambient
        if roi.size > 0:
            mean_temp = roi.mean()

        boost = 0.25 if (b["cls"] == 0 and mean_temp > ambient + delta) else 0.0
        b["conf"] = min(1.0, b["conf"] + boost)
        fused.append(b)
    return fused
