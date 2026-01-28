import cv2

def draw_boxes(img, boxes, color=(0, 255, 0)):
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        label = f'{b["name"]} {b["conf"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img
