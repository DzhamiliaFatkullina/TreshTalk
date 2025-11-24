import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Tuple
import supervision as sv

class CVDetector:
    """
    Traditional Computer Vision detector for waste objects
    Uses classical CV techniques: edge detection, contour analysis, morphological operations
    """
    
    def __init__(self, min_area: int = 1000, device: str = "cpu"):
        self.min_area = min_area
        self.device = device
        
        # Image preprocessing transform for classifier
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("Traditional CV Detector initialized")
    
    def detect(self, image: np.ndarray) -> Tuple[object, np.ndarray, List[str]]:
        """
        Detect waste objects using traditional CV methods
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            detections: Detection results in supervision format
            annotated_image: Image with bounding boxes
            labels: Detected object labels (all 'waste' for now)
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 3. Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # 4. Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 5. Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare detections in supervision format
        bboxes = []
        confidences = []
        labels_list = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append([x, y, x + w, y + h])
                confidence = min(area / 10000, 1.0)  # Confidence based on size
                confidences.append(confidence)
                labels_list.append("waste")
        
        if len(bboxes) == 0:
            # Return empty detections in supervision format
            empty_detections = sv.Detections(
                xyxy=np.array([], dtype=np.float32).reshape(0, 4),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=np.int32)
            )
            return empty_detections, image.copy(), []
        
        # Convert to numpy arrays
        bboxes_np = np.array(bboxes, dtype=np.float32)
        confidences_np = np.array(confidences, dtype=np.float32)
        class_ids = np.zeros(len(bboxes), dtype=np.int32)  # All class 0 for waste
        
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=bboxes_np,
            confidence=confidences_np,
            class_id=class_ids
        )
        
        # Annotate image with bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=detections,
            labels=labels_list
        )
        
        return detections, annotated_image, labels_list
    
    def extract_object_crops(self, image: np.ndarray, detections) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Extract cropped images of detected objects
        
        Args:
            image: Original image
            detections: Detection results from detect() method
            
        Returns:
            List of tuples (cropped_image, bbox_coordinates)
        """
        crops = []
        
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract crop (keep in BGR for OpenCV)
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:  # Ensure crop is not empty
                crops.append((crop, [x1, y1, x2, y2]))
        
        return crops
    
    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess crop for classification
        
        Args:
            crop: Cropped image as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB for PyTorch models
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(crop_rgb).unsqueeze(0).to(self.device)
    
    def set_detection_parameters(self, min_area: float = None):
        """
        Update detection parameters
        """
        if min_area is not None:
            self.min_area = min_area