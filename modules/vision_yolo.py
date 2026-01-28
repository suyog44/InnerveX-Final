from ultralytics import YOLO
import os

class YoloDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35):
        model_path = os.path.join("models", "yolo", model_name)
        self.model = YOLO(model_path)
        self.conf = conf

    def infer(self, img):
        result = self.model(img, conf=self.conf, verbose=False)[0]
        boxes = []
        for b in result.boxes:
            boxes.append({
                "cls": int(b.cls[0]),
                "name": result.names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "box": list(map(int, b.xyxy[0]))
            })
        return boxes
