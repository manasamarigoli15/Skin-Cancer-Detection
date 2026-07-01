from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import HAM10000Dataset, load_metadata


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def create_dataloaders(config: dict) -> DataLoaders:
    data_config = config["data"]
    class_names = data_config["class_names"]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    metadata = load_metadata(
        metadata_csv=data_config["metadata_csv"],
        image_dirs=data_config["image_dirs"],
    )
    metadata = metadata[metadata["dx"].isin(class_names)].copy()

    train_val_df, test_df = train_test_split(
        metadata,
        test_size=data_config["test_size"],
        stratify=metadata["dx"],
        random_state=config["project"]["seed"],
    )
    adjusted_val_size = data_config["val_size"] / (1.0 - data_config["test_size"])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["dx"],
        random_state=config["project"]["seed"],
    )

    train_dataset = HAM10000Dataset(
        train_df,
        class_to_idx=class_to_idx,
        image_size=data_config["image_size"],
        train=True,
    )
    val_dataset = HAM10000Dataset(
        val_df,
        class_to_idx=class_to_idx,
        image_size=data_config["image_size"],
        train=False,
    )
    test_dataset = HAM10000Dataset(
        test_df,
        class_to_idx=class_to_idx,
        image_size=data_config["image_size"],
        train=False,
    )

    sampler = _build_weighted_sampler(train_df, class_to_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        sampler=sampler,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def _build_weighted_sampler(
    dataframe: pd.DataFrame,
    class_to_idx: dict[str, int],
) -> WeightedRandomSampler:
    labels = dataframe["dx"].map(class_to_idx).to_numpy()
    class_counts = np.bincount(labels, minlength=len(class_to_idx))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
