from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000Dataset(Dataset):
    """PyTorch dataset for HAM10000 dermoscopic images."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int,
        train: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.class_to_idx[row["dx"]]
        return self.transform(image), label


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_metadata(metadata_csv: str | Path, image_dirs: list[str]) -> pd.DataFrame:
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {metadata_path}. "
            "Download the dataset and update configs/config.yaml."
        )

    dataframe = pd.read_csv(metadata_path)
    if "image_id" not in dataframe.columns or "dx" not in dataframe.columns:
        raise ValueError("Metadata CSV must include image_id and dx columns.")

    path_lookup = {}
    for image_dir in image_dirs:
        directory = Path(image_dir)
        if not directory.exists():
            continue
        for path in directory.glob("*.jpg"):
            path_lookup[path.stem] = str(path)

    dataframe["image_path"] = dataframe["image_id"].map(path_lookup)
    missing = dataframe["image_path"].isna().sum()
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {missing} image paths. Check data.image_dirs in config."
        )

    return dataframe
