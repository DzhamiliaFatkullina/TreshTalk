# src/utils/draw.py
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import os

def _get_font(size=22):
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    try:
        return ImageFont.load_default()
    except:
        return None

from PIL import Image, ImageDraw, ImageFont

def draw_annotations(image_path, detections, classifier_results):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)   
    except:
        font = ImageFont.load_default()

    bbox = font.getbbox("Ay")
    line_height = (bbox[3] - bbox[1]) + 6

    for det, base_clf in zip(detections, classifier_results):

        x1, y1, x2, y2 = det["bbox"]

        # получаем top-1
        baseline_label = f"{base_clf[0]['label']} ({base_clf[0]['confidence']:.2f})"
        yolo_label     = f"{det['yolo_class']} ({det['confidence']:.2f})"

        # Рисуем рамку
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        # ТЕКСТ НАД РАМКОЙ (крупный)
        text_lines = [
            f"object: {yolo_label}",
            f"class: {baseline_label}"
        ]

        for i, line in enumerate(text_lines):
            draw.text(
                (x1, y1 - (len(text_lines) - i) * line_height),
                line,
                fill="red",
                font=font,
            )

    annotated_path = image_path + ".annotated.jpg"
    img.save(annotated_path, quality=95)
    return annotated_path

