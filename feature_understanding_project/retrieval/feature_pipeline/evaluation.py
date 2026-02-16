"""
evaluation.py
-------------
Computes retrieval quality metrics across all three CNN models.

Metrics
~~~~~~~
* **Top-K Retrieval Accuracy** – fraction of queries whose *k* nearest
  neighbours belong to the same class.
* **Mean Average Precision (mAP)** – averaged over all queries; each query's
  AP is computed over the full ranked list of the database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════
# Core metric functions
# ══════════════════════════════════════════════

def top_k_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Leave-one-out Top-K retrieval accuracy.

    For every image in the database, treat it as the query and
    check whether the *k* closest other images share the same label.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    k : int

    Returns
    -------
    float
        Fraction of retrieved items that match the query class, averaged
        over all queries.
    """
    sim_matrix = cosine_similarity(embeddings)  # (N, N)
    np.fill_diagonal(sim_matrix, -np.inf)       # exclude self

    n = len(labels)
    correct = 0
    total = 0

    for i in range(n):
        top_indices = np.argsort(sim_matrix[i])[::-1][:k]
        matches = np.sum(labels[top_indices] == labels[i])
        correct += matches
        total += k

    return float(correct / total)


def mean_average_precision(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Mean Average Precision (mAP) over all queries (leave-one-out).

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)

    Returns
    -------
    float
        mAP score.
    """
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)

    n = len(labels)
    ap_sum = 0.0

    for i in range(n):
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        relevant = (labels[sorted_indices] == labels[i]).astype(np.float64)

        # Number of relevant items (same class, excluding self)
        n_relevant = np.sum(labels == labels[i]) - 1
        if n_relevant == 0:
            continue

        cumulative = np.cumsum(relevant)
        precision_at_k = cumulative / np.arange(1, n + 1, dtype=np.float64)

        # AP = (1/n_relevant) * sum of precision at each relevant position
        ap = np.sum(precision_at_k * relevant) / n_relevant
        ap_sum += ap

    return float(ap_sum / n)


# ══════════════════════════════════════════════
# Batch evaluation across models
# ══════════════════════════════════════════════

def evaluate_all_models(
    embeddings_dir: str,
    model_keys: List[str] = ("resnet", "zfnet", "googlenet"),
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate every model's embeddings stored in *embeddings_dir*.

    Parameters
    ----------
    embeddings_dir : str or Path
    model_keys : list[str]
    k : int

    Returns
    -------
    dict
        ``{model_key: {"top_k_accuracy": float, "mAP": float}}``
    """
    emb_dir = Path(embeddings_dir)
    labels = np.load(emb_dir / "labels.npy")

    results: Dict[str, Dict[str, float]] = {}

    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            print(f"[WARNING] Embeddings not found for '{key}': {emb_path}")
            continue

        embeddings = np.load(emb_path)
        print(f"[Evaluation] Computing metrics for {key} (shape={embeddings.shape}) ...")

        acc = top_k_accuracy(embeddings, labels, k=k)
        map_score = mean_average_precision(embeddings, labels)

        results[key] = {
            "top_k_accuracy": round(acc, 4),
            "mAP": round(map_score, 4),
        }
        print(f"  Top-{k} Accuracy: {acc:.4f}  |  mAP: {map_score:.4f}")

    # Persist to JSON for Django metrics page
    results_path = emb_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluation] Results saved to {results_path}")

    return results
