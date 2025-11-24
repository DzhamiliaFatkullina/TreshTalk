# src/classifier.py
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict
from pathlib import Path

# MODEL_BUILDERS minimal mapping using torchvision
from torchvision import models

MODEL_BUILDERS = {
    "resnet18": lambda num_classes, pretrained=False: _build_resnet18(num_classes, pretrained)
}

def _build_resnet18(num_classes, pretrained=False):
    m = models.resnet18(pretrained=pretrained)
    # Replace final fc
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

# Your waste classes order must match training
WASTE_CLASSES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "trash", "paper", "plastic"
]

class WasteClassifier:
    def __init__(self, model_path: str, model_name: str = "resnet18", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODEL_BUILDERS[model_name](num_classes=len(WASTE_CLASSES), pretrained=False)
        state = torch.load(model_path, map_location=self.device)
        # If state dict wrapped, attempt to find best key
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

    def predict(self, pil_crop: Image.Image) -> List[Dict]:
        """
        Return sorted list: [{"label": str, "confidence": float}, ...]
        """
        x = self.transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        results = [{"label": WASTE_CLASSES[i], "confidence": float(probs[i])} for i in range(len(WASTE_CLASSES))]
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results
