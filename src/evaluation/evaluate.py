import argparse
import json
from pathlib import Path

from src.data.datamodule import create_dataloaders
from src.evaluation.metrics import (
    collect_predictions,
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
)
from src.models.factory import create_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ensure_dir, load_config
from src.utils.logger import get_logger
from src.utils.seed import get_device, seed_everything


def evaluate(config_path: str, model_key: str, checkpoint_path: str) -> dict[str, float]:
    config = load_config(config_path)
    seed_everything(config["project"]["seed"])
    logger = get_logger("evaluate", "logs/evaluate.log")
    device = get_device(config["project"].get("device", "auto"))
    dataloaders = create_dataloaders(config)

    model = create_model(model_key, len(dataloaders.class_to_idx), config).to(device)
    load_checkpoint(checkpoint_path, model, device)

    y_true, y_pred, y_prob = collect_predictions(model, dataloaders.test, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, config["data"]["class_names"])

    output_dir = ensure_dir(config["evaluation"]["output_dir"])
    metrics_path = output_dir / f"{model_key}_metrics.json"
    report_path = output_dir / f"{model_key}_classification_report.csv"
    cm_path = output_dir / "figures" / f"{model_key}_confusion_matrix.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_classification_report(y_true, y_pred, config["data"]["class_names"], report_path)
    save_confusion_matrix(y_true, y_pred, config["data"]["class_names"], cm_path)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved confusion matrix to %s", cm_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["efficientnet", "vit"], required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.config, args.model, args.checkpoint)
    print(json.dumps(result, indent=2))
