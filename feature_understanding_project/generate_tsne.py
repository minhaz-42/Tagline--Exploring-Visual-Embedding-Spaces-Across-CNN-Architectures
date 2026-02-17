#!/usr/bin/env python3
"""Generate t-SNE plots for all 3 models using CIFAR-100 embeddings."""

import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

emb_dir = Path("embeddings")
out_dir = Path("retrieval/static/tsne_plots")
out_dir.mkdir(parents=True, exist_ok=True)

labels = np.load(emb_dir / "labels.npy")

# subsample for speed
rng = np.random.RandomState(42)
n = min(5000, len(labels))
idx = rng.choice(len(labels), n, replace=False)
labels_sub = labels[idx]

for key in ["resnet", "zfnet", "googlenet"]:
    emb_path = emb_dir / f"{key}_embeddings.npy"
    if not emb_path.exists():
        print(f"Skip {key}")
        continue
    embeddings = np.load(emb_path)[idx]
    print(f"t-SNE for {key} ({len(embeddings)} pts, D={embeddings.shape[1]})...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels_sub, cmap="tab20", alpha=0.6, s=8,
    )
    ax.set_title(f"t-SNE — {key.upper()} on CIFAR-100", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Class Index")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plot_path = out_dir / f"tsne_{key}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {plot_path}")

print("Done!")
