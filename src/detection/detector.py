import torch
import cv2
import numpy as np
from PIL import Image
import supervision as sv
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import os


class GroundingDINODetector:
    """
    Object detector using Grounding DINO for waste detection
    """
    
    def __init__(self, model_config: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                 model_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Grounding DINO detector
        
        Args:
            model_config: Path to model config file
            model_checkpoint: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = self.load_model(model_config, model_checkpoint)
        self.text_prompt = "waste . trash . garbage . rubbish . litter"
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
        # Image preprocessing transform (RL processing later)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def set_detection_parameters(self, box_threshold: float = None, text_threshold: float = None):
        """
        Update detection parameters
        """
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold
    
    def load_model(self, model_config: str, model_checkpoint: str):
        """
        Load Grounding DINO model
        """
        try:
            from groundingdino.util.inference import Model
            
            model = Model(
                model_config_path=model_config,
                model_checkpoint_path=model_checkpoint,
                device=self.device
            )
            print("✅ Grounding DINO model loaded successfully")
            return model
        except ImportError:
            raise ImportError("Grounding DINO not installed. Please install it from: "
                            "https://github.com/IDEA-Research/GroundingDINO")
        except Exception as e:
            raise Exception(f"Failed to load Grounding DINO model: {e}")
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Detect waste objects in image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            detections: Detection results
            annotated_image: Image with bounding boxes
            labels: Detected object labels
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Perform detection
        detections, labels = self.model.predict_with_caption(
            image=pil_image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Annotate image with bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(
            scene=image_rgb.copy(),
            detections=detections,
            labels=labels
        )
        
        # Convert back to BGR for OpenCV
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        return detections, annotated_image, labels
    
    def extract_object_crops(self, image: np.ndarray, detections) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Extract cropped images of detected objects
        
        Args:
            image: Original image
            detections: Detection results
            
        Returns:
            List of tuples (cropped_image, bbox_coordinates)
        """
        crops = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract crop
            crop = image_rgb[y1:y2, x1:x2]
            
            if crop.size > 0:  # Ensure crop is not empty
                crops.append((crop, [x1, y1, x2, y2]))
        
        return crops
    
    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess crop for classification
        
        Args:
            crop: Cropped image as numpy array
            
        Returns:
            Preprocessed tensor
        """
        pil_image = Image.fromarray(crop)
        return self.transform(pil_image).unsqueeze(0).to(self.device)