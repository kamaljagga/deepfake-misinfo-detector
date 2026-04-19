import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class DeepfakeClassifier:
    def __init__(self, model_path=None, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # EfficientNet-B0 with binary classification head
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, 2)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"Loaded deepfake model from {model_path}")
        else:
            print("No model path given — using untrained model for now")

        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, face_tensor) -> dict:
        """
        face_tensor: [3, H, W] tensor from MTCNN
        Returns dict with label and confidence
        """
        if face_tensor is None:
            return {
                "label":      "NO_FACE",
                "confidence": 0.0
            }

        try:
            # Convert tensor to PIL then apply transforms
            to_pil = transforms.ToPILImage()
            img    = to_pil(face_tensor.byte()
                    if face_tensor.dtype == torch.uint8
                    else face_tensor)
            x = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1)[0]

            label = "FAKE" if probs[1] > 0.5 else "REAL"
            return {
                "label":      label,
                "confidence": round(probs[1].item(), 4)
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return {"label": "ERROR", "confidence": 0.0}