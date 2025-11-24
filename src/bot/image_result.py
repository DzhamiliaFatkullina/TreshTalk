import os
from pathlib import Path
from PIL import Image

from src.cv.detector import YoloDetector, SUPPORTED_WASTE_CLASSES
from src.cv.classifier import WasteClassifier
from src.cv.utils.draw import draw_boxes


# --------------- GLOBAL LOAD (best practice) -------------------
YOLO_MODEL = YoloDetector("src/cv/yolo/yolov8l.pt")
CLASSIFIER = WasteClassifier("src/models/baseline.pth", model_name="resnet18")
# ---------------------------------------------------------------


def detect_and_classify(image_path: str):
    """
    INPUT:
        image_path: str
    RETURNS:
        annotated_path: str
        classifications: list of dicts -> [{"label": "...", "confidence": ...}]
    """

    # ---------------- YOLO DETECTION ----------------
    detections = YOLO_MODEL.detect(image_path)

    if not detections:
        # если ничего не нашли — просто вернуть копию
        annotated = image_path + ".annotated.jpg"
        Image.open(image_path).save(annotated)
        return annotated, []

    classifier_results = []

    # ---------------- CROP & CLASSIFY ----------------
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # аккуратный crop
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        crop = img.crop((x1, y1, x2, y2))

        # классификация только если подходит под категории мусора
        clf = CLASSIFIER.predict(crop)
        classifier_results.append(clf)

    # ---------------- DRAW FINAL IMAGE ----------------
    annotated_image = draw_boxes(image_path, detections, classifier_results)

    annotated_path = image_path + ".annotated.jpg"
    annotated_image.save(annotated_path, quality=95)

    # финальная агрегация
    # берем лучший класс по первому объекту (или все)
    final_classifications = classifier_results[0]

    return annotated_path, final_classifications
