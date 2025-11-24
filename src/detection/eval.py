import torch
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from .detector import GroundingDINODetector


def evaluate_detector(image_path: str, output_dir: str = None, device: str = None):
    detector = GroundingDINODetector(device=device)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    detections, annotated_image, labels = detector.detect(image)
    
    print(f"🔍 Detection Results for {image_path}:")
    print(f"📦 Detected {len(detections.xyxy)} objects:")
    for i, (bbox, label) in enumerate(zip(detections.xyxy, labels)):
        x1, y1, x2, y2 = bbox.astype(int)
        print(f"  Object {i+1}: {label} at [{x1}, {y1}, {x2}, {y2}]")
    
    # Extract and count crops
    crops = detector.extract_object_crops(image, detections)
    print(f"🌱 Extracted {len(crops)} object crops")
    
    # Save or display results
    # if output_dir:
    #     output_path = Path(output_dir) / f"detection_{Path(image_path).stem}.jpg"
    #     cv2.imwrite(str(output_path), annotated_image)
    #     print(f"💾 Detection visualization saved to: {output_path}")
    
    # Display
    cv2.imshow("Detection Results", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'detections': detections,
        'annotated_image': annotated_image,
        'labels': labels,
        'crops': crops
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Grounding DINO detector')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    # parser.add_argument('--output', type=str, default='./detection_results', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    # Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluate_detector(args.image, args.device)


if __name__ == "__main__":
    main()