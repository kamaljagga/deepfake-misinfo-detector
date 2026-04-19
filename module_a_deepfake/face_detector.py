import torch
from PIL import Image
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, image_size=224, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=20,
            keep_all=False,
            device=self.device,
            post_process=False
        )

    def extract_face(self, image_path: str):
        """
        Returns cropped face tensor [3, H, W] or None if no face found.
        """
        try:
            img  = Image.open(image_path).convert('RGB')
            face = self.mtcnn(img)
            return face
        except Exception as e:
            print(f"Face detection error on {image_path}: {e}")
            return None

    def extract_face_from_pil(self, pil_image):
        """
        Takes a PIL image directly and returns face tensor.
        Used for Streamlit uploads.
        """
        try:
            img  = pil_image.convert('RGB')
            face = self.mtcnn(img)
            return face
        except Exception as e:
            print(f"Face detection error: {e}")
            return None