import timm
import torch


def create_model(
    model_key: str,
    num_classes: int,
    config: dict,
) -> torch.nn.Module:
    """Create a configured classifier model."""
    if model_key not in config["models"]:
        supported = ", ".join(config["models"].keys())
        raise ValueError(f"Unknown model '{model_key}'. Supported: {supported}")

    model_config = config["models"][model_key]
    return timm.create_model(
        model_config["name"],
        pretrained=model_config.get("pretrained", True),
        num_classes=num_classes,
    )


def get_gradcam_target_layer(model: torch.nn.Module, model_key: str) -> torch.nn.Module:
    """Pick a useful final feature layer for Grad-CAM-like visualization."""
    if model_key == "efficientnet":
        return model.conv_head
    if model_key == "vit":
        # The final transformer block norm provides token-level activations.
        return model.blocks[-1].norm1
    raise ValueError(f"No Grad-CAM target layer configured for {model_key}")
