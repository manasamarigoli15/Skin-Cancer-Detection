import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.datamodule import create_dataloaders
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.models.factory import create_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import ensure_dir, load_config
from src.utils.logger import get_logger
from src.utils.seed import get_device, seed_everything


def train(
    config_path: str,
    model_key: str,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
) -> Path:
    config = load_config(config_path)
    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override
    if batch_size_override is not None:
        config["data"]["batch_size"] = batch_size_override
    seed_everything(config["project"]["seed"])
    ensure_dir("logs")
    logger = get_logger(f"train.{model_key}", f"logs/{model_key}_train.log")
    device = get_device(config["project"].get("device", "auto"))
    logger.info("Using device: %s", device)

    dataloaders = create_dataloaders(config)
    model = create_model(model_key, len(dataloaders.class_to_idx), config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = _build_criterion(config, dataloaders.train_df, dataloaders.class_to_idx, device)
    scaler = torch.cuda.amp.GradScaler(
        enabled=config["training"]["mixed_precision"] and device.type == "cuda"
    )

    best_f1 = -1.0
    patience_counter = 0
    checkpoint_dir = ensure_dir(config["training"]["checkpoint_dir"])
    best_path = checkpoint_dir / f"best_{model_key}.pt"

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=dataloaders.train,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        y_true, y_pred, y_prob = collect_predictions(model, dataloaders.val, device)
        val_metrics = compute_metrics(
            y_true,
            y_pred,
            y_prob,
            config["data"]["class_names"],
        )
        logger.info(
            "Epoch %d | train_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_auc=%.4f",
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["roc_auc"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                class_to_idx=dataloaders.class_to_idx,
            )
            logger.info("Saved new best checkpoint: %s", best_path)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    return best_path


def _train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())

    return running_loss / len(dataloader.dataset)


def _build_criterion(
    config: dict,
    train_df,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> nn.CrossEntropyLoss:
    if not config["training"]["use_weighted_loss"]:
        return nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])

    labels = train_df["dx"].map(class_to_idx).to_numpy()
    counts = np.bincount(labels, minlength=len(class_to_idx))
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=config["training"]["label_smoothing"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a skin cancer classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["efficientnet", "vit"], required=True)
    parser.add_argument("--epochs", type=int, help="Override training.epochs from config.")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size from config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = train(args.config, args.model, args.epochs, args.batch_size)
    print(f"Best checkpoint: {checkpoint}")


