import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageEmbedder:
    def __init__(self):
        self.use_torch = False
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            resnet = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.use_torch = True
            logger.info("Initialized ResNet50 for image embeddings.")
        except ImportError:
            self.use_torch = False
            logger.warning("Torch/torchvision not available. Falling back to OpenCV.")
            
    def embed(self, image_path: str) -> np.ndarray:
        if self.use_torch:
            try:
                from PIL import Image
                import torch
                img = Image.open(image_path).convert('RGB')
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model(tensor).squeeze().cpu().numpy()
                return embedding
            except Exception as e:
                logger.error(f"Torch embedding failed: {e}")
                return np.zeros(2048)
        else:
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    return np.zeros(2048)
                img = cv2.resize(img, (224, 224))
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                padded = np.zeros(2048)
                padded[:len(hist)] = hist
                return padded
            except Exception as e:
                logger.error(f"CV2 embedding failed: {e}")
                return np.zeros(2048)

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def batch_embed(self, image_paths: List[str]) -> List[np.ndarray]:
        return [self.embed(path) for path in image_paths]
