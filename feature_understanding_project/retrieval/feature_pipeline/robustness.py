"""
robustness.py
-------------
Robustness benchmarking module — measures how each CNN backbone degrades
under common image perturbations.

Perturbation types
~~~~~~~~~~~~~~~~~~
* Rotation (15°, 45°, 90°)
* Gaussian noise (σ = 0.01, 0.05, 0.1)
* Gaussian blur (kernel 3, 5, 7)
* Brightness shift (factor 0.5, 1.5, 2.0)
* Contrast shift (factor 0.5, 1.5, 2.0)

For each (model, perturbation, level) triple the module:
  1. Applies the transform to every image in the dataset
  2. Extracts embeddings from the perturbed images
  3. Queries the *original* clean index
  4. Measures Top-K accuracy drop
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════
# Image perturbation functions
# ═══════════════════════════════════════════════

def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def apply_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


PERTURBATION_FUNCTIONS = {
    "rotation": rotate_image,
    "noise": add_gaussian_noise,
    "blur": apply_blur,
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
}

DEFAULT_PERTURBATIONS = {
    "rotation": [15, 45, 90],
    "noise": [0.01, 0.05, 0.1],
    "blur": [3, 5, 7],
    "brightness": [0.5, 1.5, 2.0],
    "contrast": [0.5, 1.5, 2.0],
}


def apply_perturbation(img: Image.Image, perturbation_type: str, level) -> Image.Image:
    """Apply a named perturbation at a given level to a PIL image."""
    fn = PERTURBATION_FUNCTIONS.get(perturbation_type)
    if fn is None:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")
    return fn(img, level)


# ═══════════════════════════════════════════════
# Batch perturbation + embedding extraction
# ═══════════════════════════════════════════════

def _extract_perturbed_embeddings(
    model,
    image_paths: List[str],
    perturbation_type: str,
    level,
    transform,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract embeddings for all images after applying a perturbation."""
    import torch

    all_embeddings = []
    model.eval()

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = apply_perturbation(img, perturbation_type, level)
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
        all_embeddings.append(features)

    return np.concatenate(all_embeddings, axis=0)


def _topk_accuracy_against_clean(
    perturbed_embeddings: np.ndarray,
    clean_embeddings: np.ndarray,
    perturbed_labels: np.ndarray,
    clean_labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Measure how well perturbed queries retrieve correct results from the
    *clean* database index.
    """
    sim = cosine_similarity(perturbed_embeddings, clean_embeddings)
    n = len(perturbed_labels)
    correct = 0
    total = 0
    for i in range(n):
        top_idx = np.argsort(sim[i])[::-1][:k]
        matches = np.sum(clean_labels[top_idx] == perturbed_labels[i])
        correct += matches
        total += k
    return float(correct / total) if total > 0 else 0.0


# ═══════════════════════════════════════════════
# Full robustness evaluation
# ═══════════════════════════════════════════════

def evaluate_model_robustness(
    model_key: str,
    embeddings_dir: str | Path,
    perturbations: Optional[Dict] = None,
    k: int = 10,
    max_images: int = 500,
) -> Dict:
    """
    Evaluate a single model's robustness against all perturbation types.

    Uses a subsample of *max_images* for speed (perturbation extraction
    requires forward passes).
    """
    import torch
    from .feature_extractor import get_model
    from .dataset_loader import get_transform

    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    emb_dir = Path(embeddings_dir)
    clean_embeddings = np.load(emb_dir / f"{model_key}_embeddings.npy")
    labels = np.load(emb_dir / "labels.npy")
    with open(emb_dir / "image_paths.json") as f:
        image_paths = json.load(f)

    # Subsample for speed
    n = len(labels)
    if n > max_images:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_images, replace=False)
    else:
        idx = np.arange(n)

    sub_paths = [image_paths[i] for i in idx]
    sub_labels = labels[idx]
    sub_clean_emb = clean_embeddings[idx]

    # Baseline (clean-vs-clean)
    baseline_acc = _topk_accuracy_against_clean(
        sub_clean_emb, clean_embeddings, sub_labels, labels, k
    )

    model = get_model(model_key)
    device = next(model.parameters()).device
    transform = get_transform()

    results: Dict = {
        "model_key": model_key,
        "baseline_accuracy": round(baseline_acc, 6),
        "perturbations": {},
        "n_images_tested": len(idx),
    }

    for ptype, levels in perturbations.items():
        results["perturbations"][ptype] = {}
        for level in levels:
            print(f"  [{model_key}] {ptype} level={level} ...")
            t0 = time.time()
            perturbed_emb = _extract_perturbed_embeddings(
                model, sub_paths, ptype, level, transform, device
            )
            acc = _topk_accuracy_against_clean(
                perturbed_emb, clean_embeddings, sub_labels, labels, k
            )
            elapsed = time.time() - t0
            degradation = baseline_acc - acc
            results["perturbations"][ptype][str(level)] = {
                "accuracy": round(acc, 6),
                "degradation": round(degradation, 6),
                "degradation_pct": round(degradation / baseline_acc * 100, 2) if baseline_acc > 0 else 0,
                "time_seconds": round(elapsed, 2),
            }

    return results


def evaluate_all_models_robustness(
    embeddings_dir: str | Path,
    model_keys: Tuple[str, ...] = ("resnet", "zfnet", "googlenet"),
    perturbations: Optional[Dict] = None,
    k: int = 10,
    max_images: int = 500,
) -> Dict:
    """Run robustness evaluation across all models."""
    all_results: Dict = {}
    for key in model_keys:
        print(f"[Robustness] Evaluating {key} ...")
        all_results[key] = evaluate_model_robustness(
            key, embeddings_dir, perturbations, k, max_images
        )
    return all_results


def export_robustness_json(results: Dict, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return str(output_path)
