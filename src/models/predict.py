from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import build_transforms
from src.models.factory import create_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import get_device


class SkinCancerPredictor:
    """Inference wrapper for trained classifiers."""

    def __init__(self, checkpoint_path: str | Path, model_key: str, config: dict) -> None:
        self.config = config
        self.device = get_device(config["project"].get("device", "auto"))
        self.class_names = config["data"]["class_names"]
        self.idx_to_class = dict(enumerate(self.class_names))
        self.model_key = model_key
        self.model = create_model(model_key, len(self.class_names), config)
        load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms(config["data"]["image_size"], train=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, index = torch.max(probabilities, dim=0)
        return {
            "class_index": int(index.item()),
            "class_name": self.idx_to_class[int(index.item())],
            "confidence": float(confidence.item()),
            "probabilities": probabilities.detach().cpu().numpy(),
        }
