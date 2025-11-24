from ultralytics import YOLO
import torch

SUPPORTED_WASTE_CLASSES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "trash", "paper", "plastic"
]

class YoloDetector:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image_path: str):
        """
        Returns list:
        [
            {
                "bbox": [x1,y1,x2,y2],
                "yolo_class": "plastic_bottle",
                "confidence": 0.92
            },
            ...
        ]
        """
        results = self.model(image_path)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.cpu())
            conf = float(box.conf.cpu())
            label = results.names[cls_id]

            detections.append({
                "bbox": box.xyxy.cpu().tolist()[0],
                "yolo_class": label,
                "confidence": conf
            })

        return detections
