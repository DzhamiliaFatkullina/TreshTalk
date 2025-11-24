import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from src.models.baselines import build_resnet18

# Ваши классы мусора
WASTE_CLASSES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "trash", "paper", "plastic"
]

class WasteClassifier:
    def __init__(self, model_path: str, model_name: str = "resnet18", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_resnet18(num_classes=9, pretrained=True).to(device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict(self, crop_img: Image.Image):
        x = self.transform(crop_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        result = []
        for idx, p in enumerate(probs):
            result.append({
                "label": WASTE_CLASSES[idx],
                "confidence": float(p.cpu())
            })

        result.sort(key=lambda x: x["confidence"], reverse=True)
        return result
