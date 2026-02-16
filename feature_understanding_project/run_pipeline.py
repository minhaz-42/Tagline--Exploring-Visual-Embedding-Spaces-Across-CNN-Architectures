#!/usr/bin/env python
"""
run_pipeline.py
===============
Offline pipeline script — run ONCE before starting the Django server.

Steps executed:
    1. Load Caltech-101 images for the 8 selected classes.
    2. Extract embeddings using ResNet-101, ZFNet, and GoogLeNet.
    3. Save embeddings, labels, and image paths to disk.
    4. Compute evaluation metrics (Top-10 accuracy, mAP).
    5. Generate t-SNE visualisation plots.

Usage::

    cd feature_understanding_project
    python run_pipeline.py --dataset_dir ./dataset

The script auto-detects GPU/MPS and falls back to CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path so we can import the pipeline ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.feature_pipeline.dataset_loader import get_dataloader
from retrieval.feature_pipeline.feature_extractor import (
    get_model,
    extract_embeddings,
    _get_device,
)
from retrieval.feature_pipeline.evaluation import evaluate_all_models
from retrieval.feature_pipeline.visualize_tsne import generate_all_tsne_plots


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SELECTED_CLASSES = [
    "accordion",
    "airplane",
    "camera",
    "elephant",
    "laptop",
    "motorbike",
    "watch",
    "wheelchair",
]

MODEL_KEYS = ["resnet", "zfnet", "googlenet"]


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    embeddings_dir = Path(args.embeddings_dir).resolve()
    tsne_dir = Path(args.tsne_dir).resolve()

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"\n{'='*60}")
    print(f"  Tagline — Offline Embedding Pipeline")
    print(f"  Device : {device}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output : {embeddings_dir}")
    print(f"{'='*60}\n")

    # ── 1. Load dataset ──
    print("[Step 1/5] Loading dataset …")
    dataloader, dataset = get_dataloader(
        dataset_root=str(dataset_dir),
        selected_classes=SELECTED_CLASSES,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_per_class=args.max_per_class,
    )

    # ── 2. Extract embeddings for each model ──
    print("\n[Step 2/5] Extracting embeddings …")
    all_labels = None
    all_paths = None

    for model_key in MODEL_KEYS:
        print(f"\n  → {model_key.upper()}")
        t0 = time.time()

        model = get_model(model_key)
        embeddings, labels, paths = extract_embeddings(model, dataloader, device)

        # Save embeddings
        emb_file = embeddings_dir / f"{model_key}_embeddings.npy"
        np.save(str(emb_file), embeddings)
        print(f"    Saved {emb_file.name}  shape={embeddings.shape}  "
              f"({time.time()-t0:.1f}s)")

        # Labels and paths are the same for all models (same dataset order)
        if all_labels is None:
            all_labels = labels
            all_paths = paths

    # ── 3. Save labels & paths ──
    print("\n[Step 3/5] Saving labels and image paths …")
    np.save(str(embeddings_dir / "labels.npy"), all_labels)

    with open(embeddings_dir / "image_paths.json", "w") as f:
        json.dump(all_paths, f)

    print(f"    labels.npy        ({len(all_labels)} entries)")
    print(f"    image_paths.json  ({len(all_paths)} entries)")

    # ── 4. Evaluate ──
    print("\n[Step 4/5] Computing evaluation metrics …")
    evaluate_all_models(
        embeddings_dir=str(embeddings_dir),
        model_keys=MODEL_KEYS,
        k=10,
    )

    # ── 5. t-SNE plots ──
    print("\n[Step 5/5] Generating t-SNE plots …")
    generate_all_tsne_plots(
        embeddings_dir=str(embeddings_dir),
        output_dir=str(tsne_dir),
        class_names=SELECTED_CLASSES,
        model_keys=MODEL_KEYS,
    )

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Embeddings → {embeddings_dir}")
    print(f"  t-SNE      → {tsne_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the offline embedding extraction & evaluation pipeline."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="Path to the Caltech-101 subset directory (default: ./dataset).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(PROJECT_ROOT / "embeddings"),
        help="Where to save .npy embedding files (default: ./embeddings).",
    )
    parser.add_argument(
        "--tsne_dir",
        type=str,
        default=str(PROJECT_ROOT / "retrieval" / "static" / "tsne_plots"),
        help="Where to save t-SNE PNG plots (default: ./retrieval/static/tsne_plots/).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=300,
        help="Max images per class to load (default: 300).",
    )

    args = parser.parse_args()
    main(args)
