"""
advanced_viz.py
---------------
Publication-ready visualisation generator.

All plots are saved as both PNG and PDF for latex / paper embedding.
Uses a dark theme consistent with the UI design language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── Dark theme palette ──
DARK_BG = "#0f172a"
DARK_CARD = "#1e293b"
DARK_TEXT = "#f1f5f9"
DARK_MUTED = "#64748b"
DARK_GRID = "#334155"
ACCENT_CYAN = "#22d3ee"
ACCENT_PURPLE = "#7c3aed"
ACCENT_EMERALD = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_ROSE = "#f43f5e"

MODEL_COLORS = {
    "resnet": ACCENT_CYAN,
    "zfnet": ACCENT_AMBER,
    "googlenet": ACCENT_EMERALD,
}

MODEL_DISPLAY = {
    "resnet": "ResNet-101",
    "zfnet": "ZFNet",
    "googlenet": "GoogLeNet",
}

CLASS_COLORS_20 = [
    "#22d3ee", "#7c3aed", "#10b981", "#f59e0b", "#f43f5e",
    "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
    "#06b6d4", "#a855f7", "#84cc16", "#eab308", "#ef4444",
    "#6366f1", "#d946ef", "#0ea5e9", "#22c55e", "#e11d48",
]


def _apply_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_CARD,
        "axes.edgecolor": DARK_GRID,
        "axes.labelcolor": DARK_TEXT,
        "text.color": DARK_TEXT,
        "xtick.color": DARK_MUTED,
        "ytick.color": DARK_MUTED,
        "grid.color": DARK_GRID,
        "grid.alpha": 0.3,
        "legend.facecolor": DARK_CARD,
        "legend.edgecolor": DARK_GRID,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _save_plot(fig, path: Path):
    """Save figure as both PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════
# t-SNE with dark theme
# ═══════════════════════════════════════════════

def plot_tsne_dark(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str | Path,
    max_points: int = 5000,
    perplexity: int = 30,
) -> str:
    """Generate a dark-themed t-SNE scatter plot."""
    _apply_dark_style()
    save_path = Path(save_path)

    if len(embeddings) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) // 4),
        max_iter=1000,
        random_state=42,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(np.unique(labels))
    fig, ax = plt.subplots(figsize=(14, 10))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = CLASS_COLORS_20[i % len(CLASS_COLORS_20)]
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, s=18, alpha=0.8, edgecolors="none",
        )

    ax.legend(
        loc="best", fontsize=8, markerscale=2, framealpha=0.9,
        ncol=max(1, len(unique_labels) // 10),
    )
    ax.set_title(f"t-SNE · {model_name}", fontsize=16, fontweight="bold", color=ACCENT_CYAN)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.15)
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Confusion matrix
# ═══════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str | Path,
) -> str:
    """Generate a dark-themed confusion matrix heatmap."""
    _apply_dark_style()
    save_path = Path(save_path)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.5)))

    # Custom dark colormap
    colors = [DARK_CARD, ACCENT_PURPLE, ACCENT_CYAN]
    cmap = LinearSegmentedColormap.from_list("dark_cm", colors, N=256)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [name[:12] for name in class_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            color = DARK_BG if cm[i, j] > thresh else DARK_TEXT
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=7, fontweight="bold")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"Confusion Matrix · {model_name}",
        fontsize=14, fontweight="bold", color=ACCENT_CYAN,
    )
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Model performance comparison bar charts
# ═══════════════════════════════════════════════

def plot_performance_comparison(
    benchmark_results: Dict,
    save_path: str | Path,
) -> str:
    """Bar chart comparing models on key metrics."""
    _apply_dark_style()
    save_path = Path(save_path)

    models = benchmark_results.get("model_keys", [])
    model_data = benchmark_results.get("model_results", {})

    metrics = ["mAP", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy"]
    metric_labels = ["mAP", "Top-1 Acc", "Top-5 Acc", "Top-10 Acc"]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_key in enumerate(models):
        loo = model_data[model_key].get("leave_one_out", {})
        values = [loo.get(m, 0) for m in metrics]
        color = MODEL_COLORS.get(model_key, ACCENT_CYAN)
        bars = ax.bar(x + i * width, values, width, label=MODEL_DISPLAY.get(model_key, model_key),
                      color=color, alpha=0.9, edgecolor="none")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=DARK_TEXT)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", color=ACCENT_CYAN)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Robustness degradation graph
# ═══════════════════════════════════════════════

def plot_robustness_degradation(
    robustness_results: Dict,
    save_path: str | Path,
) -> str:
    """Line chart showing accuracy degradation per perturbation type."""
    _apply_dark_style()
    save_path = Path(save_path)

    model_keys = list(robustness_results.keys())
    ptypes = list(robustness_results[model_keys[0]]["perturbations"].keys())

    n_ptypes = len(ptypes)
    fig, axes = plt.subplots(1, n_ptypes, figsize=(5 * n_ptypes, 5), squeeze=False)

    for pi, ptype in enumerate(ptypes):
        ax = axes[0, pi]
        for model_key in model_keys:
            pdata = robustness_results[model_key]["perturbations"].get(ptype, {})
            levels = sorted(pdata.keys(), key=lambda x: float(x))
            accs = [pdata[l]["accuracy"] for l in levels]
            baseline = robustness_results[model_key]["baseline_accuracy"]
            color = MODEL_COLORS.get(model_key, ACCENT_CYAN)
            ax.plot(range(len(levels)), accs, "o-", color=color, linewidth=2,
                    markersize=6, label=MODEL_DISPLAY.get(model_key, model_key))
            ax.axhline(y=baseline, color=color, linestyle="--", alpha=0.3)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_xlabel(f"{ptype.capitalize()} Level", fontsize=11)
        ax.set_ylabel("Accuracy" if pi == 0 else "", fontsize=11)
        ax.set_title(ptype.capitalize(), fontsize=12, fontweight="bold", color=ACCENT_CYAN)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Robustness: Accuracy Under Perturbations",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN, y=1.02)
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Embedding distance histograms
# ═══════════════════════════════════════════════

def plot_distance_histograms(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: str | Path,
    max_pairs: int = 50000,
) -> str:
    """Histogram of intra-class vs inter-class cosine distances."""
    _apply_dark_style()
    save_path = Path(save_path)

    sim_matrix = cosine_similarity(embeddings)
    n = len(labels)

    intra_sims, inter_sims = [], []
    rng = np.random.RandomState(42)
    pairs = rng.choice(n, size=(min(max_pairs, n * (n - 1) // 2), 2))

    for i, j in pairs:
        if i >= j:
            continue
        if labels[i] == labels[j]:
            intra_sims.append(sim_matrix[i, j])
        else:
            inter_sims.append(sim_matrix[i, j])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intra_sims, bins=50, alpha=0.7, color=ACCENT_EMERALD, label="Intra-class", density=True)
    ax.hist(inter_sims, bins=50, alpha=0.7, color=ACCENT_ROSE, label="Inter-class", density=True)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Similarity Distribution · {model_name}",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Dimensionality vs performance scatter
# ═══════════════════════════════════════════════

def plot_dim_vs_performance(
    benchmark_results: Dict,
    model_dims: Dict[str, int],
    save_path: str | Path,
) -> str:
    """Scatter plot: embedding dimensionality vs mAP."""
    _apply_dark_style()
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    model_data = benchmark_results.get("model_results", {})

    for key, data in model_data.items():
        dim = model_dims.get(key, 0)
        mAP = data.get("leave_one_out", {}).get("mAP", 0)
        color = MODEL_COLORS.get(key, ACCENT_CYAN)
        ax.scatter(dim, mAP, c=color, s=200, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(MODEL_DISPLAY.get(key, key), (dim, mAP),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Embedding Dimensionality", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("Dimensionality vs Retrieval Performance",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Runtime comparison bar chart
# ═══════════════════════════════════════════════

def plot_runtime_comparison(
    runtime_data: Dict[str, Dict],
    save_path: str | Path,
) -> str:
    """Bar chart comparing inference time and model size across models."""
    _apply_dark_style()
    save_path = Path(save_path)

    models = list(runtime_data.keys())
    times = [runtime_data[m].get("avg_ms", 0) for m in models]
    sizes = [runtime_data[m].get("model_size_mb", 0) for m in models]
    colors = [MODEL_COLORS.get(m, ACCENT_CYAN) for m in models]
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Inference time
    bars1 = ax1.bar(display_names, times, color=colors, alpha=0.9, edgecolor="none")
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=10, color=DARK_TEXT)
    ax1.set_ylabel("Inference Time (ms)", fontsize=12)
    ax1.set_title("Avg Inference Time", fontsize=13, fontweight="bold", color=ACCENT_CYAN)
    ax1.grid(axis="y", alpha=0.2)

    # Model size
    bars2 = ax2.bar(display_names, sizes, color=colors, alpha=0.9, edgecolor="none")
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}MB", ha="center", va="bottom", fontsize=10, color=DARK_TEXT)
    ax2.set_ylabel("Model Size (MB)", fontsize=12)
    ax2.set_title("Model Parameter Size", fontsize=13, fontweight="bold", color=ACCENT_CYAN)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("Runtime & Memory Comparison",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN, y=1.02)
    fig.tight_layout()
    _save_plot(fig, save_path)
    return str(save_path)


# ═══════════════════════════════════════════════
# Generate all plots
# ═══════════════════════════════════════════════

def generate_all_plots(
    embeddings_dir: str | Path,
    output_dir: str | Path,
    class_names: List[str],
    benchmark_results: Optional[Dict] = None,
    robustness_results: Optional[Dict] = None,
    analysis_results: Optional[Dict] = None,
    model_keys: Tuple[str, ...] = ("resnet", "zfnet", "googlenet"),
) -> Dict[str, str]:
    """Generate a complete set of publication-ready plots."""
    emb_dir = Path(embeddings_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(emb_dir / "labels.npy")

    generated: Dict[str, str] = {}

    # t-SNE plots
    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            continue
        embeddings = np.load(emb_path)
        save = out_dir / f"tsne_{key}.png"
        plot_tsne_dark(embeddings, labels, class_names, MODEL_DISPLAY.get(key, key), save)
        generated[f"tsne_{key}"] = str(save)

    # Confusion matrices
    if analysis_results:
        for key in model_keys:
            if key not in analysis_results:
                continue
            cm_data = analysis_results[key].get("confusion_matrix")
            cm_labels = analysis_results[key].get("confusion_labels", [])
            if cm_data is not None:
                cm = np.array(cm_data)
                names = [class_names[l] if l < len(class_names) else f"Class {l}" for l in cm_labels]
                save = out_dir / f"confusion_{key}.png"
                plot_confusion_matrix(cm, names, MODEL_DISPLAY.get(key, key), save)
                generated[f"confusion_{key}"] = str(save)

    # Distance histograms
    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            continue
        embeddings = np.load(emb_path)
        save = out_dir / f"distances_{key}.png"
        plot_distance_histograms(embeddings, labels, MODEL_DISPLAY.get(key, key), save)
        generated[f"distances_{key}"] = str(save)

    # Performance comparison
    if benchmark_results:
        save = out_dir / "performance_comparison.png"
        plot_performance_comparison(benchmark_results, save)
        generated["performance_comparison"] = str(save)

        # Dimensionality vs performance
        model_dims = {"resnet": 2048, "zfnet": 4096, "googlenet": 1024}
        save = out_dir / "dim_vs_performance.png"
        plot_dim_vs_performance(benchmark_results, model_dims, save)
        generated["dim_vs_performance"] = str(save)

    # Robustness degradation
    if robustness_results:
        save = out_dir / "robustness_degradation.png"
        plot_robustness_degradation(robustness_results, save)
        generated["robustness_degradation"] = str(save)

    # Runtime comparison
    if analysis_results:
        runtime_data = {}
        for key in model_keys:
            if key in analysis_results and analysis_results[key].get("runtime"):
                runtime_data[key] = analysis_results[key]["runtime"]
        if runtime_data:
            save = out_dir / "runtime_comparison.png"
            plot_runtime_comparison(runtime_data, save)
            generated["runtime_comparison"] = str(save)

    print(f"[Viz] Generated {len(generated)} plots in {out_dir}")
    return generated
