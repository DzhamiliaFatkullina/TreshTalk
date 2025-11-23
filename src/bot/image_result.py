"""
image_result.py
Responsible only for: 
- loading model
- running inference
- returning: (annotated_image_path, classifications_dict)
"""

import os
import json
from shutil import copyfile


def detect_and_classify(image_path: str):
    """
    INPUT: path to input image
    RETURNS:
        annotated_path: path to image with detections drawn
        classifications: dict like {"plastic": 0.88}
    """

    # ------------------- PLACEHOLDER -------------------
    # TODO: Load model.pth once globally, not inside the function.
    # TODO: Run inference, draw boxes, save annotated image.

    annotated_path = image_path + ".annotated.jpg"
    copyfile(image_path, annotated_path)

    classifications = {"plastic": 0.88}

    return annotated_path, classifications
