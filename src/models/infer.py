import argparse
import json
from pathlib import Path

from PIL import Image

from src.models.predict import SkinCancerPredictor
from src.utils.config import load_config
from src.utils.explain import generate_prediction_explanation
from src.utils.gradcam import GradCAM


def predict_image(
    config_path: str,
    model_key: str,
    checkpoint_path: str,
    image_path: str,
    output_heatmap: str | None,
) -> dict:
    config = load_config(config_path)
    predictor = SkinCancerPredictor(checkpoint_path, model_key, config)
    image = Image.open(image_path).convert("RGB")
    prediction = predictor.predict(image)
    prediction["explanation"] = generate_prediction_explanation(
        prediction["class_name"],
        prediction["confidence"],
        config["data"]["class_descriptions"],
    )

    if output_heatmap:
        gradcam = GradCAM(predictor.model, model_key)
        try:
            result = gradcam.generate(
                image=image,
                class_index=prediction["class_index"],
                image_size=config["data"]["image_size"],
                device=predictor.device,
            )
            Path(output_heatmap).parent.mkdir(parents=True, exist_ok=True)
            result.overlay.save(output_heatmap)
        finally:
            gradcam.close()

    prediction["probabilities"] = prediction["probabilities"].tolist()
    return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["efficientnet", "vit"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-heatmap")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(
        args.config,
        args.model,
        args.checkpoint,
        args.image,
        args.output_heatmap,
    )
    print(json.dumps(result, indent=2))
