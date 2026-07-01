from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_transforms
from src.models.factory import get_gradcam_target_layer


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    overlay: Image.Image


class GradCAM:
    """Small Grad-CAM implementation for CNNs and ViT token activations."""

    def __init__(self, model: torch.nn.Module, model_key: str) -> None:
        self.model = model
        self.model_key = model_key
        self.target_layer = get_gradcam_target_layer(model, model_key)
        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: Image.Image,
        class_index: int | None,
        image_size: int,
        device: torch.device,
    ) -> GradCAMResult:
        self.model.eval()
        transform = build_transforms(image_size=image_size, train=False)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

        logits = self.model(tensor)
        target_index = int(torch.argmax(logits, dim=1).item()) if class_index is None else class_index
        score = logits[:, target_index].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        activations, gradients = self._spatialize(self.activations, self.gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return GradCAMResult(heatmap=cam, overlay=_overlay_heatmap(image, cam, image_size))

    def _spatialize(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if activations.ndim == 4:
            return activations, gradients

        if self.model_key == "vit" and activations.ndim == 3:
            activations = activations[:, 1:, :]
            gradients = gradients[:, 1:, :]
            grid_size = int(activations.shape[1] ** 0.5)
            activations = activations.transpose(1, 2).reshape(
                activations.shape[0],
                activations.shape[2],
                grid_size,
                grid_size,
            )
            gradients = gradients.transpose(1, 2).reshape(
                gradients.shape[0],
                gradients.shape[2],
                grid_size,
                grid_size,
            )
            return activations, gradients

        raise ValueError(f"Unsupported activation shape for Grad-CAM: {activations.shape}")


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, image_size: int) -> Image.Image:
    base = image.convert("RGB").resize((image_size, image_size))
    base_array = np.array(base)
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * base_array + 0.45 * color_map)
    return Image.fromarray(overlay)
