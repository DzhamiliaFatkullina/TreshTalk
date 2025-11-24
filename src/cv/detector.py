# src/detector.py
from ultralytics import YOLO
import torch
from typing import List, Dict

# Classes to ignore (we were asked to ignore "столы" = "tables")
IGNORED_CLASSES = {"table", "desk"}

class YoloDetector:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ultralytics YOLO accepts path or model name
        self.model = YOLO(model_path)
        # model.to() is handled internally by ultralytics normally, but ensure device
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def detect(self, image_path: str):
        """
        Always returns list of dicts in format:

        {
            "bbox": [x1, y1, x2, y2],
            "yolo_class": "bottle",
            "confidence": 0.95
        }
        """
        results = self.model(image_path)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls.cpu())
            conf = float(box.conf.cpu())
            label = results.names[cls_id]  # e.g. "bottle"

            detections.append({
                "bbox": box.xyxy.cpu().tolist()[0],
                "yolo_class": label,       # ← гарантируем правильный ключ
                "confidence": conf
            })

        return detections
