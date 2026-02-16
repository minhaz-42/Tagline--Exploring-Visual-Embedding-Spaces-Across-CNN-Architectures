"""
nearest_neighbors.py
--------------------
Cosine-similarity-based nearest-neighbour retrieval using scikit-learn.

Provides:
    - ``build_index``  – fits a NearestNeighbors model on stored embeddings
    - ``query_index``  – retrieves the top-K most similar items for a query
    - ``load_embeddings`` – loads .npy embedding / label / path files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════
# Loading helpers
# ══════════════════════════════════════════════

def load_embeddings(
    embeddings_dir: str,
    model_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-computed embeddings, labels, and image paths from disk.

    Expected files inside *embeddings_dir*::

        {model_key}_embeddings.npy   – (N, D) float32
        labels.npy                   – (N,) int64
        image_paths.json             – list[str] of length N

    Parameters
    ----------
    embeddings_dir : str or Path
        Directory where embeddings were saved.
    model_key : str
        One of ``"resnet"``, ``"zfnet"``, ``"googlenet"``.

    Returns
    -------
    embeddings : np.ndarray
    labels : np.ndarray
    paths : list[str]
    """
    emb_dir = Path(embeddings_dir)
    embeddings = np.load(emb_dir / f"{model_key}_embeddings.npy")
    labels = np.load(emb_dir / "labels.npy")

    paths_file = emb_dir / "image_paths.json"
    with open(paths_file, "r") as f:
        paths = json.load(f)

    assert len(embeddings) == len(labels) == len(paths), (
        f"Mismatch: embeddings={len(embeddings)}, labels={len(labels)}, paths={len(paths)}"
    )
    return embeddings, labels, paths


# ══════════════════════════════════════════════
# Index building & querying
# ══════════════════════════════════════════════

def build_index(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> NearestNeighbors:
    """
    Fit a scikit-learn NearestNeighbors model on *embeddings*.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    metric : str
        Distance metric (``"cosine"`` recommended).

    Returns
    -------
    NearestNeighbors
        Fitted index ready for queries.
    """
    nn_model = NearestNeighbors(n_neighbors=10, metric=metric, algorithm="brute")
    nn_model.fit(embeddings)
    return nn_model


def query_index(
    nn_model: NearestNeighbors,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: List[str],
    top_k: int = 10,
) -> List[Dict]:
    """
    Retrieve the *top_k* nearest neighbours for a single query embedding.

    Parameters
    ----------
    nn_model : NearestNeighbors
        Fitted index.
    query_embedding : np.ndarray, shape (D,)
    embeddings : np.ndarray, shape (N, D)
        Full database embeddings (used for similarity computation).
    labels : np.ndarray, shape (N,)
    paths : list[str]
    top_k : int

    Returns
    -------
    list[dict]
        Each dict: ``{"rank", "path", "label", "similarity"}``.
    """
    query = query_embedding.reshape(1, -1)
    distances, indices = nn_model.kneighbors(query, n_neighbors=top_k)

    results: List[Dict] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        # Cosine distance → cosine similarity
        similarity = 1.0 - dist
        results.append({
            "rank": rank,
            "path": paths[idx],
            "label": int(labels[idx]),
            "similarity": round(float(similarity), 4),
        })

    return results


def compute_cosine_similarities(
    query_embedding: np.ndarray,
    database_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a single query and all database vectors.

    Returns
    -------
    np.ndarray, shape (N,)
        Similarity scores in [-1, 1].
    """
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    db_norm = normalize(database_embeddings, axis=1)
    return db_norm @ query_norm
