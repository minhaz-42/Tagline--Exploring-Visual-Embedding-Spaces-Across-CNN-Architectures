#!/usr/bin/env python
"""
run_cifar100_pipeline.py
========================
Offline pipeline for CIFAR-100 dataset — downloads CIFAR-100, extracts
embeddings using ResNet-101, ZFNet, and GoogLeNet, computes evaluation
metrics, and generates t-SNE visualisations.

CIFAR-100 has 100 classes with 600 images each (500 train + 100 test).
Using the training split (50,000 images) or a configurable subset.

Usage::

    cd feature_understanding_project
    python run_cifar100_pipeline.py

Options::

    python run_cifar100_pipeline.py --max_per_class 100 --batch_size 64
    python run_cifar100_pipeline.py --use_test  # use test split instead

The script auto-detects GPU/MPS and falls back to CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.feature_pipeline.feature_extractor import (
    get_model,
    _get_device,
)
from retrieval.feature_pipeline.evaluation import evaluate_all_models


# ──────────────────────────────────────────────
# ImageNet normalisation (same as Caltech pipeline)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# CIFAR-100 class names (coarse + fine)
# ──────────────────────────────────────────────
CIFAR100_FINE_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

MODEL_KEYS = ["resnet", "zfnet", "googlenet"]


# ──────────────────────────────────────────────
# Dataset wrapper for CIFAR-100 that returns
# (image, label, path_string) like the Caltech loader
# ──────────────────────────────────────────────
class CIFAR100WithPaths(Dataset):
    """
    Wraps torchvision CIFAR-100 and returns (image_tensor, label, fake_path).
    The 'path' is synthesised as 'cifar100/{class_name}/{index}.png' so it can
    be stored in image_paths.json consistently.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        max_per_class: int = 0,
    ):
        self.cifar = datasets.CIFAR100(
            root=root, train=train, download=True, transform=None
        )
        self.transform = transform or self._default_transform()
        self.class_names = CIFAR100_FINE_LABELS

        # Build per-class indices
        class_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(self.cifar):
            class_indices.setdefault(label, []).append(idx)

        # Optional: limit per class
        self.indices: List[int] = []
        for label in sorted(class_indices.keys()):
            idxs = class_indices[label]
            if max_per_class > 0:
                idxs = idxs[:max_per_class]
            self.indices.extend(idxs)

        print(
            f"[CIFAR-100] Loaded {len(self.indices)} images "
            f"across {len(class_indices)} classes "
            f"(max_per_class={max_per_class or 'all'})"
        )

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        real_idx = self.indices[idx]
        image, label = self.cifar[real_idx]
        image = image.convert("RGB")

        # Synthesise a path string
        class_name = self.class_names[label]
        fake_path = f"cifar100/{class_name}/{real_idx}.png"

        if self.transform:
            image = self.transform(image)

        return image, label, fake_path


# ──────────────────────────────────────────────
# Embedding extraction (identical to original)
# ──────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract embeddings for every sample in dataloader."""
    all_embeddings = []
    all_labels = []
    all_paths = []

    model.eval()
    for images, labels, paths in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)
        all_embeddings.append(features.cpu().numpy())
        all_labels.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else labels)
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)
    return embeddings, labels_arr, all_paths


# ──────────────────────────────────────────────
# t-SNE generation (inline to avoid import issues)
# ──────────────────────────────────────────────
def generate_cifar100_tsne(embeddings_dir, output_dir, model_keys):
    """Generate t-SNE plots for CIFAR-100 embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emb_dir = Path(embeddings_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = np.load(emb_dir / "labels.npy")
        unique_labels = np.unique(labels)

        # Use a subset for t-SNE if too many points
        max_tsne_points = 5000
        if len(labels) > max_tsne_points:
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(len(labels), max_tsne_points, replace=False)
        else:
            subset_idx = np.arange(len(labels))

        labels_sub = labels[subset_idx]

        for key in model_keys:
            emb_path = emb_dir / f"{key}_embeddings.npy"
            if not emb_path.exists():
                continue

            embeddings = np.load(emb_path)
            emb_sub = embeddings[subset_idx]

            print(f"  [t-SNE] Computing for {key} ({len(emb_sub)} points)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            coords = tsne.fit_transform(emb_sub)

            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=labels_sub, cmap="tab20", alpha=0.6, s=8,
            )
            ax.set_title(f"t-SNE — {key.upper()} on CIFAR-100", fontsize=14)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            # Add colorbar with some class names
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Class index")

            plt.tight_layout()
            plot_path = out_dir / f"tsne_{key}.png"
            fig.savefig(str(plot_path), dpi=150)
            plt.close(fig)
            print(f"    Saved {plot_path}")

    except Exception as e:
        print(f"[WARNING] t-SNE generation failed: {e}")


# ──────────────────────────────────────────────
# Save CIFAR-100 images to disk for web display
# ──────────────────────────────────────────────
def save_cifar100_images(dataset_obj, output_dir, max_per_class=50):
    """
    Save a subset of CIFAR-100 images as actual files so the web app
    can display them in retrieval results.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Access underlying cifar dataset
    cifar = dataset_obj.cifar
    class_names = dataset_obj.class_names
    class_counts: Dict[int, int] = {}

    print(f"[Save Images] Saving CIFAR-100 images to {out_dir}...")

    for idx in tqdm(dataset_obj.indices, desc="Saving images"):
        image, label = cifar[idx]
        class_name = class_names[label]

        cnt = class_counts.get(label, 0)
        if cnt >= max_per_class:
            continue
        class_counts[label] = cnt + 1

        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        img_path = class_dir / f"{idx}.png"
        if not img_path.exists():
            image.save(str(img_path))

    total = sum(class_counts.values())
    print(f"[Save Images] Saved {total} images across {len(class_counts)} classes")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    cifar_root = Path(args.cifar_root).resolve()
    embeddings_dir = Path(args.embeddings_dir).resolve()
    tsne_dir = Path(args.tsne_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"\n{'='*60}")
    print(f"  Tagline — CIFAR-100 Embedding Pipeline")
    print(f"  Device      : {device}")
    print(f"  CIFAR root  : {cifar_root}")
    print(f"  Embeddings  : {embeddings_dir}")
    print(f"  Dataset imgs: {dataset_dir}")
    print(f"  Max/class   : {args.max_per_class or 'all'}")
    print(f"{'='*60}\n")

    # ── 1. Load CIFAR-100 ──
    print("[Step 1/6] Downloading & loading CIFAR-100 ...")
    cifar_dataset = CIFAR100WithPaths(
        root=str(cifar_root),
        train=not args.use_test,
        max_per_class=args.max_per_class,
    )

    dataloader = DataLoader(
        cifar_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 2. Save images to disk for web display ──
    print("\n[Step 2/6] Saving CIFAR-100 images to disk ...")
    save_cifar100_images(cifar_dataset, dataset_dir, max_per_class=args.save_per_class)

    # ── 3. Extract embeddings ──
    print("\n[Step 3/6] Extracting embeddings ...")
    all_labels = None
    all_paths = None

    for model_key in MODEL_KEYS:
        print(f"\n  → {model_key.upper()}")
        t0 = time.time()

        model = get_model(model_key)
        embeddings, labels, paths = extract_embeddings(model, dataloader, device)

        emb_file = embeddings_dir / f"{model_key}_embeddings.npy"
        np.save(str(emb_file), embeddings)
        print(f"    Saved {emb_file.name}  shape={embeddings.shape}  ({time.time()-t0:.1f}s)")

        if all_labels is None:
            all_labels = labels
            all_paths = paths

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 4. Save labels & paths ──
    print("\n[Step 4/6] Saving labels and image paths ...")
    np.save(str(embeddings_dir / "labels.npy"), all_labels)

    with open(embeddings_dir / "image_paths.json", "w") as f:
        json.dump(all_paths, f)

    print(f"    labels.npy        ({len(all_labels)} entries)")
    print(f"    image_paths.json  ({len(all_paths)} entries)")

    # ── 5. Evaluate ──
    print("\n[Step 5/6] Computing evaluation metrics ...")
    evaluate_all_models(
        embeddings_dir=str(embeddings_dir),
        model_keys=MODEL_KEYS,
        k=10,
    )

    # ── 6. t-SNE plots ──
    print("\n[Step 6/6] Generating t-SNE plots ...")
    generate_cifar100_tsne(
        embeddings_dir=str(embeddings_dir),
        output_dir=str(tsne_dir),
        model_keys=MODEL_KEYS,
    )

    # ── 7. Update Django settings with new class names ──
    class_names_path = embeddings_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(CIFAR100_FINE_LABELS, f)
    print(f"\n    class_names.json saved ({len(CIFAR100_FINE_LABELS)} classes)")

    print(f"\n{'='*60}")
    print("  CIFAR-100 Pipeline complete!")
    print(f"  Embeddings   → {embeddings_dir}")
    print(f"  Dataset imgs → {dataset_dir}")
    print(f"  t-SNE        → {tsne_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-100 embedding extraction & evaluation pipeline."
    )
    parser.add_argument(
        "--cifar_root", type=str,
        default=str(PROJECT_ROOT / "_cifar100_download"),
        help="Where to download/cache CIFAR-100 (default: ./_cifar100_download).",
    )
    parser.add_argument(
        "--embeddings_dir", type=str,
        default=str(PROJECT_ROOT / "embeddings"),
        help="Where to save .npy embedding files (default: ./embeddings).",
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="Where to save CIFAR-100 images for web display (default: ./dataset).",
    )
    parser.add_argument(
        "--tsne_dir", type=str,
        default=str(PROJECT_ROOT / "retrieval" / "static" / "tsne_plots"),
        help="Where to save t-SNE PNG plots.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for embedding extraction (default: 64).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=100,
        help="Max images per class for embeddings (0 = all, default: 100).",
    )
    parser.add_argument(
        "--save_per_class", type=int, default=50,
        help="Max images per class to save as files for web display (default: 50).",
    )
    parser.add_argument(
        "--use_test", action="store_true",
        help="Use test split instead of train split.",
    )

    args = parser.parse_args()
    main(args)
