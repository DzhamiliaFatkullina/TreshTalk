# src/pipeline.py
import os
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict

from src.cv.detector import YoloDetector
from src.cv.classifier import WasteClassifier
from src.cv.utils.draw import draw_annotations

# Initialize models once (best practice)
# Paths: adapt to your layout
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "src/cv/yolo/yolov8l.pt")
BASELINE_WEIGHTS = os.getenv("BASELINE_WEIGHTS", "src/models/baseline.pth")

# create singletons
YOLO = YoloDetector(YOLO_WEIGHTS)
BASE = WasteClassifier(BASELINE_WEIGHTS, model_name="resnet18")

def detect_and_classify(image_path: str) -> Tuple[str, List[Dict]]:
    # 1) YOLO detection
    detections = YOLO.detect(image_path)
    if not detections:
        # nothing detected -> return copy
        out = image_path + ".annotated.jpg"
        Image.open(image_path).save(out)
        return out, []

    classifier_results = []
    aggregated = []

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        # safe crop bounds
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        crop = img.crop((x1, y1, x2, y2))

        # 2) baseline classifier on crop
        base_pred = BASE.predict(crop)  # sorted list
        classifier_results.append(base_pred)

        top_base = base_pred[0]
        agg = {
            "bbox": [x1, y1, x2, y2],
            "yolo": {"label": det["yolo_class"], "confidence": det["confidence"]},
            "baseline": {"label": top_base["label"], "confidence": top_base["confidence"]},
        }
        aggregated.append(agg)

    annotated_path = draw_annotations(image_path, detections, classifier_results)

    return annotated_path, aggregated
