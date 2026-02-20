"""
feature_analysis.py
-------------------
Feature-space analysis module for embedding quality assessment.

Provides:
  - Intra-class and inter-class distance computation
  - Fisher discriminant ratio
  - Per-class embedding statistics
  - k-NN confusion matrix generation
  - Embedding norm distribution analysis
  - Cosine vs Euclidean similarity comparison
  - Runtime & memory benchmarking per model
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize


# ═══════════════════════════════════════════════
# Intra / Inter class distances
# ═══════════════════════════════════════════════

def compute_class_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> Dict:
    """
    Compute intra-class and inter-class distances.

    Returns per-class intra distances, global inter distance,
    and the Fisher discriminant ratio.
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1.0 - sim_matrix
    else:
        dist_matrix = euclidean_distances(embeddings)

    intra_distances: Dict[int, List[float]] = {}
    class_centroids: Dict[int, np.ndarray] = {}

    for lbl in unique_labels:
        mask = labels == lbl
        class_emb = embeddings[mask]
        class_centroids[int(lbl)] = class_emb.mean(axis=0)

        # Intra-class: pairwise distances within the class
        if np.sum(mask) > 1:
            class_dist = dist_matrix[np.ix_(mask, mask)]
            upper_tri = class_dist[np.triu_indices(len(class_dist), k=1)]
            intra_distances[int(lbl)] = upper_tri.tolist()
        else:
            intra_distances[int(lbl)] = [0.0]

    # Inter-class: distances between class centroids
    centroid_labels = sorted(class_centroids.keys())
    centroids = np.array([class_centroids[l] for l in centroid_labels])
    if metric == "cosine":
        inter_dist_matrix = 1.0 - cosine_similarity(centroids)
    else:
        inter_dist_matrix = euclidean_distances(centroids)

    inter_pairs = inter_dist_matrix[np.triu_indices(n_classes, k=1)]

    # Per-class summary
    per_class = {}
    for lbl in unique_labels:
        dists = intra_distances[int(lbl)]
        per_class[int(lbl)] = {
            "mean_intra_distance": round(float(np.mean(dists)), 6),
            "std_intra_distance": round(float(np.std(dists)), 6),
            "n_samples": int(np.sum(labels == lbl)),
        }

    global_intra = np.mean([np.mean(v) for v in intra_distances.values()])
    global_inter = np.mean(inter_pairs)
    fisher_ratio = global_inter / (global_intra + 1e-10)

    return {
        "metric": metric,
        "global_intra_distance": round(float(global_intra), 6),
        "global_inter_distance": round(float(global_inter), 6),
        "fisher_ratio": round(float(fisher_ratio), 4),
        "per_class": per_class,
        "n_classes": n_classes,
    }


# ═══════════════════════════════════════════════
# Confusion matrix via k-NN classification
# ═══════════════════════════════════════════════

def knn_confusion_matrix(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a confusion matrix using leave-one-out k-NN classification.

    Returns (confusion_matrix, sorted_unique_labels).
    """
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    n = len(labels)
    unique_labels = sorted(np.unique(labels).tolist())
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    n_classes = len(unique_labels)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(n):
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        top_k_labels = labels[top_k_idx]
        # Majority vote
        counts = np.bincount(top_k_labels, minlength=max(unique_labels) + 1)
        predicted = np.argmax(counts)
        true_idx = label_to_idx[labels[i]]
        pred_idx = label_to_idx[predicted]
        cm[true_idx, pred_idx] += 1

    return cm, unique_labels


# ═══════════════════════════════════════════════
# Embedding statistics
# ═══════════════════════════════════════════════

def embedding_statistics(embeddings: np.ndarray) -> Dict:
    """Compute statistics about the embedding space."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "n_samples": int(embeddings.shape[0]),
        "dimensionality": int(embeddings.shape[1]),
        "mean_l2_norm": round(float(np.mean(norms)), 4),
        "std_l2_norm": round(float(np.std(norms)), 4),
        "min_l2_norm": round(float(np.min(norms)), 4),
        "max_l2_norm": round(float(np.max(norms)), 4),
        "mean_feature_value": round(float(np.mean(embeddings)), 6),
        "std_feature_value": round(float(np.std(embeddings)), 6),
        "sparsity": round(float(np.mean(embeddings == 0)), 4),
    }


# ═══════════════════════════════════════════════
# Normalization comparison
# ═══════════════════════════════════════════════

def normalization_comparison(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> Dict:
    """Compare retrieval accuracy with and without L2 normalization."""
    from .statistical_benchmark import leave_one_out_evaluation

    # Raw embeddings
    raw_results = leave_one_out_evaluation(embeddings, labels, k_values=(1, 5, k))
    raw_results.pop("per_query_ap", None)

    # L2-normalized
    normed = normalize(embeddings, axis=1)
    norm_results = leave_one_out_evaluation(normed, labels, k_values=(1, 5, k))
    norm_results.pop("per_query_ap", None)

    return {
        "raw": raw_results,
        "l2_normalized": norm_results,
    }


# ═══════════════════════════════════════════════
# Cosine vs Euclidean comparison
# ═══════════════════════════════════════════════

def similarity_metric_comparison(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> Dict:
    """Compare cosine vs Euclidean similarity for retrieval."""
    n = len(labels)

    # Cosine
    cos_sim = cosine_similarity(embeddings)
    np.fill_diagonal(cos_sim, -np.inf)
    cos_correct = 0
    for i in range(n):
        top_idx = np.argsort(cos_sim[i])[::-1][:k]
        cos_correct += np.sum(labels[top_idx] == labels[i])
    cos_acc = cos_correct / (n * k)

    # Euclidean (use negative distance for ranking)
    euc_dist = euclidean_distances(embeddings)
    np.fill_diagonal(euc_dist, np.inf)
    euc_correct = 0
    for i in range(n):
        top_idx = np.argsort(euc_dist[i])[:k]
        euc_correct += np.sum(labels[top_idx] == labels[i])
    euc_acc = euc_correct / (n * k)

    return {
        "cosine_accuracy": round(float(cos_acc), 6),
        "euclidean_accuracy": round(float(euc_acc), 6),
        "cosine_better": cos_acc > euc_acc,
        "difference": round(float(cos_acc - euc_acc), 6),
    }


# ═══════════════════════════════════════════════
# Runtime benchmarking
# ═══════════════════════════════════════════════

def runtime_benchmark(
    model_key: str,
    image_path: str,
    n_runs: int = 10,
) -> Dict:
    """Measure average inference time for a single image."""
    import torch
    from .feature_extractor import get_model, extract_single_embedding

    model = get_model(model_key)
    device = next(model.parameters()).device

    # Warmup
    _ = extract_single_embedding(model, image_path)
    _ = extract_single_embedding(model, image_path)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = extract_single_embedding(model, image_path)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    # Memory
    params = sum(p.numel() for p in model.parameters())
    param_mb = params * 4 / (1024 * 1024)  # float32

    return {
        "model_key": model_key,
        "avg_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "n_parameters": int(params),
        "model_size_mb": round(param_mb, 2),
    }


# ═══════════════════════════════════════════════
# Full analysis pipeline
# ═══════════════════════════════════════════════

def full_feature_analysis(
    embeddings_dir: str | Path,
    model_keys: Tuple[str, ...] = ("resnet", "zfnet", "googlenet"),
    k: int = 10,
) -> Dict:
    """Run complete feature space analysis for all models."""
    emb_dir = Path(embeddings_dir)
    labels = np.load(emb_dir / "labels.npy")
    with open(emb_dir / "image_paths.json") as f:
        image_paths = json.load(f)

    results: Dict = {}

    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            continue
        print(f"[Analysis] Analysing {key} ...")
        embeddings = np.load(emb_path)

        # Class distances
        distances = compute_class_distances(embeddings, labels, metric="cosine")

        # Confusion matrix
        cm, unique_labels = knn_confusion_matrix(embeddings, labels, k=1)

        # Embedding stats
        emb_stats = embedding_statistics(embeddings)

        # Normalization comparison
        norm_cmp = normalization_comparison(embeddings, labels, k=k)

        # Similarity metric comparison
        sim_cmp = similarity_metric_comparison(embeddings, labels, k=k)

        # Runtime benchmark (use first available image)
        sample_img = image_paths[0] if image_paths else None
        rt = {}
        if sample_img and Path(sample_img).exists():
            rt = runtime_benchmark(key, sample_img)

        results[key] = {
            "class_distances": distances,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": unique_labels,
            "embedding_stats": emb_stats,
            "normalization_comparison": norm_cmp,
            "similarity_comparison": sim_cmp,
            "runtime": rt,
        }

    return results


def export_analysis_json(results: Dict, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return str(output_path)
