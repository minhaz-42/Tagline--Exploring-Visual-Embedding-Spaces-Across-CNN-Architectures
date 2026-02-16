"""
visualize_tsne.py
-----------------
Generates t-SNE scatter plots of the embedding spaces for each CNN model.

Each plot colours points by class label, with a legend showing the 8
selected Caltech-101 categories.  Plots are saved as PNG files under
``retrieval/static/tsne_plots/`` so Django's ``{% static %}`` tag can serve them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (server-safe)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ══════════════════════════════════════════════
# Colour palette (8 classes)
# ══════════════════════════════════════════════

CLASS_COLORS = [
    "#e6194B",  # accordion  – red
    "#3cb44b",  # airplane   – green
    "#4363d8",  # camera     – blue
    "#f58231",  # elephant   – orange
    "#911eb4",  # laptop     – purple
    "#42d4f4",  # motorbike  – cyan
    "#f032e6",  # watch      – magenta
    "#bfef45",  # wheelchair – lime
]


def generate_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> str:
    """
    Run t-SNE on *embeddings* and save a scatter plot to *save_path*.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    class_names : list[str]
        Ordered class names (index corresponds to label int).
    model_name : str
        Human-readable model name used in the plot title.
    save_path : str or Path
        Destination PNG file.
    perplexity : int
    n_iter : int
    random_state : int

    Returns
    -------
    str
        Absolute path to the saved plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[t-SNE] Running for {model_name} (n={len(embeddings)}, D={embeddings.shape[1]}) ...")

    # sklearn's TSNE uses `max_iter` (some versions accept `n_iter`);
    # pass the requested iterations as `max_iter` for compatibility.
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)  # (N, 2)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 9))

    unique_labels = sorted(np.unique(labels))
    for lbl in unique_labels:
        mask = labels == lbl
        color = CLASS_COLORS[lbl % len(CLASS_COLORS)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=class_names[lbl],
            s=18,
            alpha=0.75,
            edgecolors="none",
        )

    ax.set_title(f"t-SNE Visualization — {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(
        loc="best",
        fontsize=10,
        markerscale=2.5,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)

    print(f"[t-SNE] Plot saved → {save_path}")
    return str(save_path)


def generate_all_tsne_plots(
    embeddings_dir: str,
    output_dir: str,
    class_names: List[str],
    model_keys: List[str] = ("resnet", "zfnet", "googlenet"),
    model_display_names: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Generate t-SNE plots for every model whose embeddings exist in *embeddings_dir*.

    Parameters
    ----------
    embeddings_dir : str or Path
    output_dir : str or Path
        Where PNG files will be written.
    class_names : list[str]
    model_keys : list[str]
    model_display_names : dict, optional
        Mapping from model key to a pretty display name.

    Returns
    -------
    dict
        ``{model_key: path_to_png}``
    """
    if model_display_names is None:
        model_display_names = {
            "resnet": "ResNet-101",
            "zfnet": "ZFNet",
            "googlenet": "GoogLeNet (Inception v1)",
        }

    emb_dir = Path(embeddings_dir)
    labels = np.load(emb_dir / "labels.npy")

    plot_paths: Dict[str, str] = {}

    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            print(f"[WARNING] Skipping t-SNE for '{key}': embeddings not found.")
            continue

        embeddings = np.load(emb_path)
        display = model_display_names.get(key, key)
        save_file = Path(output_dir) / f"tsne_{key}.png"

        generate_tsne_plot(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            model_name=display,
            save_path=str(save_file),
        )
        plot_paths[key] = str(save_file)

    return plot_paths
