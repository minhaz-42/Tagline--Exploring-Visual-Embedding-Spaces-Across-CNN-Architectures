"""
visualize_tsne.py
-----------------
Generates t-SNE scatter plots of the embedding spaces for each CNN model.

Each plot colours points by class label. For small numbers of classes (<=20),
a legend is included. For larger datasets (e.g. CIFAR-100), a colorbar is
used. Plots are saved as PNG files under
``retrieval/static/tsne_plots/`` so Django's ``{% static %}`` tag can serve them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (server-safe)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


# ══════════════════════════════════════════════
# Colour palette (for <=20 classes)
# ══════════════════════════════════════════════

CLASS_COLORS_20 = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
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
    max_points: int = 5000,
) -> str:
    """
    Run t-SNE on *embeddings* and save a scatter plot to *save_path*.

    For large datasets, automatically subsamples to *max_points* for
    computational feasibility. Adapts coloring strategy based on the
    number of unique classes.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Sub-sample for large datasets
    if len(embeddings) > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    print(f"[t-SNE] Running for {model_name} (n={len(embeddings)}, D={embeddings.shape[1]}) ...")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) // 4),
        max_iter=n_iter,
        random_state=random_state,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)  # (N, 2)

    # ── Plot ──
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)

    fig, ax = plt.subplots(figsize=(14, 10))

    if n_classes <= 20:
        # Discrete legend for small number of classes
        for lbl in unique_labels:
            mask = labels == lbl
            color = CLASS_COLORS_20[lbl % len(CLASS_COLORS_20)]
            name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, label=name, s=20, alpha=0.75, edgecolors="none",
            )
        ax.legend(
            loc="best", fontsize=9, markerscale=2.5,
            framealpha=0.9, ncol=max(1, n_classes // 10),
        )
    else:
        # Colormap for many classes (e.g. CIFAR-100)
        cmap = cm.get_cmap("tab20", n_classes) if n_classes <= 100 else cm.get_cmap("viridis")
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap=cmap, s=10, alpha=0.6, edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Class Index", fontsize=11)

    ax.set_title(f"t-SNE Visualization — {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
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
