"""
dataset_loader.py
-----------------
Handles loading and preprocessing of Caltech-101 images for the selected 8 classes.

Provides:
    - ImageNet-normalised transforms (224×224)
    - A PyTorch Dataset that scans the flat class-folder layout
    - A DataLoader factory with configurable batch size / workers
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ──────────────────────────────────────────────
# ImageNet statistics for normalisation
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Standard transform pipeline
# ──────────────────────────────────────────────

def get_transform(image_size: int = 224) -> transforms.Compose:
    """
    Return the standard preprocessing transform:
    Resize → CenterCrop → ToTensor → Normalise (ImageNet stats).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

class Caltech101Subset(Dataset):
    """
    A PyTorch Dataset that loads images from a flat directory layout::

        dataset_root/
            accordion/
                image_0001.jpg
                ...
            airplane/
                ...

    Parameters
    ----------
    dataset_root : str or Path
        Root directory containing one sub-folder per class.
    selected_classes : list[str]
        Class folder names to include.
    transform : callable, optional
        Torchvision transform applied to every image.
    max_per_class : int, optional
        Cap the number of images loaded per class (useful for balancing).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    def __init__(
        self,
        dataset_root: str,
        selected_classes: List[str],
        transform: Optional[transforms.Compose] = None,
        max_per_class: int = 300,
    ):
        self.dataset_root = Path(dataset_root)
        self.selected_classes = sorted(selected_classes)
        self.transform = transform or get_transform()
        self.max_per_class = max_per_class

        # Build class-to-index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Scan filesystem
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self._scan_dataset()

    # ── private helpers ──

    def _scan_dataset(self) -> None:
        """Walk each class directory and collect image paths + labels."""
        for class_name in self.selected_classes:
            class_dir = self.dataset_root / class_name
            if not class_dir.is_dir():
                print(f"[WARNING] Class directory not found: {class_dir}")
                continue

            files = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ])

            # Limit per class
            files = files[: self.max_per_class]

            for fpath in files:
                self.image_paths.append(str(fpath))
                self.labels.append(self.class_to_idx[class_name])

        print(
            f"[DatasetLoader] Loaded {len(self.image_paths)} images "
            f"across {len(self.selected_classes)} classes."
        )

    # ── Dataset API ──

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """Return (transformed_image, label, image_path)."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_path


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloader(
    dataset_root: str,
    selected_classes: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    max_per_class: int = 300,
) -> Tuple[DataLoader, Caltech101Subset]:
    """
    Convenience function that returns a DataLoader and the underlying Dataset.

    Parameters
    ----------
    dataset_root : str
        Path to the root dataset directory.
    selected_classes : list[str]
        Which Caltech-101 class folders to include.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        Number of parallel data-loading workers.
    max_per_class : int
        Maximum images to load per class.

    Returns
    -------
    tuple[DataLoader, Caltech101Subset]
    """
    dataset = Caltech101Subset(
        dataset_root=dataset_root,
        selected_classes=selected_classes,
        transform=get_transform(),
        max_per_class=max_per_class,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset
