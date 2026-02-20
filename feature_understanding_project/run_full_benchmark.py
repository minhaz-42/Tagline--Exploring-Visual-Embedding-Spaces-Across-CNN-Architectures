#!/usr/bin/env python3
"""
run_full_benchmark.py
─────────────────────
Offline pipeline that runs all research-grade evaluations:

  1. Statistical benchmarking   (leave-one-out, 80/20 split, t-tests, CI)
  2. Robustness testing         (rotation, noise, blur, brightness, contrast)
  3. Feature space analysis     (distances, confusion, runtime, normalization)
  4. Publication-ready plots    (t-SNE, confusion, bar charts, degradation)

Results are saved as JSON files and PNG/PDF plots in the embeddings directory.

Usage
~~~~~
    python run_full_benchmark.py                  # default paths
    python run_full_benchmark.py --skip-robustness  # skip slow robustness tests
    python run_full_benchmark.py --quick            # fast mode (fewer images)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project packages are importable
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np


# ── Paths ──
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DATASET_DIR = BASE_DIR / "dataset"
PLOTS_DIR = BASE_DIR / "retrieval" / "static" / "plots"
MODEL_KEYS = ("resnet", "zfnet", "googlenet")


def _load_class_names() -> list[str]:
    path = EMBEDDINGS_DIR / "class_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return [
        "accordion", "airplane", "camera", "elephant",
        "laptop", "motorbike", "watch", "wheelchair",
    ]


def _load(model_key: str):
    emb = np.load(EMBEDDINGS_DIR / f"{model_key}_embeddings.npy")
    labels = np.load(EMBEDDINGS_DIR / "labels.npy")
    with open(EMBEDDINGS_DIR / "image_paths.json") as f:
        paths = json.load(f)
    return emb, labels, paths


# ═══════════════════════════════════════════════
# Step 1: Statistical Benchmark
# ═══════════════════════════════════════════════

def run_statistical_benchmark(class_names: list[str]) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 1 : Statistical Benchmark")
    print("=" * 60)

    from retrieval.feature_pipeline.statistical_benchmark import cross_model_comparison, export_results_json

    results = cross_model_comparison(
        str(EMBEDDINGS_DIR),
        model_keys=MODEL_KEYS,
        k_values=[1, 5, 10],
    )

    out = EMBEDDINGS_DIR / "benchmark_results.json"
    export_results_json(results, str(out))
    print(f"  ✓ Saved benchmark results → {out}")
    return results


# ═══════════════════════════════════════════════
# Step 2: Robustness Testing
# ═══════════════════════════════════════════════

def run_robustness(class_names: list[str], quick: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 2 : Robustness Testing")
    print("=" * 60)

    from retrieval.feature_pipeline.robustness import evaluate_all_models_robustness, export_robustness_json

    results = evaluate_all_models_robustness(
        dataset_dir=str(DATASET_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        model_keys=MODEL_KEYS,
        max_images=100 if quick else 500,
    )

    out = EMBEDDINGS_DIR / "robustness_results.json"
    export_robustness_json(results, str(out))
    print(f"  ✓ Saved robustness results → {out}")
    return results


# ═══════════════════════════════════════════════
# Step 3: Feature Space Analysis
# ═══════════════════════════════════════════════

def run_feature_analysis(class_names: list[str]) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 3 : Feature Space Analysis")
    print("=" * 60)

    from retrieval.feature_pipeline.feature_analysis import full_feature_analysis, export_analysis_json

    results = full_feature_analysis(
        str(EMBEDDINGS_DIR),
        model_keys=MODEL_KEYS,
        class_names=class_names,
    )

    out = EMBEDDINGS_DIR / "analysis_results.json"
    export_analysis_json(results, str(out))
    print(f"  ✓ Saved analysis results → {out}")
    return results


# ═══════════════════════════════════════════════
# Step 4: Visualization
# ═══════════════════════════════════════════════

def run_visualizations(
    class_names: list[str],
    benchmark_results: dict | None,
    robustness_results: dict | None,
    analysis_results: dict | None,
) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 4 : Generating Publication-Ready Plots")
    print("=" * 60)

    from retrieval.feature_pipeline.advanced_viz import generate_all_plots

    generated = generate_all_plots(
        embeddings_dir=str(EMBEDDINGS_DIR),
        output_dir=str(PLOTS_DIR),
        class_names=class_names,
        benchmark_results=benchmark_results,
        robustness_results=robustness_results,
        analysis_results=analysis_results,
        model_keys=MODEL_KEYS,
    )

    print(f"  ✓ Generated {len(generated)} plots → {PLOTS_DIR}")
    return generated


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run full research benchmark pipeline")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip the robustness testing step (saves ~20 min)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: use fewer images for robustness testing")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate plots from existing JSON results")
    args = parser.parse_args()

    start = time.time()
    class_names = _load_class_names()

    # Ensure output dirs exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_results = None
    robustness_results = None
    analysis_results = None

    if args.plots_only:
        # Load existing results
        bp = EMBEDDINGS_DIR / "benchmark_results.json"
        rp = EMBEDDINGS_DIR / "robustness_results.json"
        ap = EMBEDDINGS_DIR / "analysis_results.json"
        if bp.exists():
            with open(bp) as f:
                benchmark_results = json.load(f)
        if rp.exists():
            with open(rp) as f:
                robustness_results = json.load(f)
        if ap.exists():
            with open(ap) as f:
                analysis_results = json.load(f)
    else:
        benchmark_results = run_statistical_benchmark(class_names)
        if not args.skip_robustness:
            robustness_results = run_robustness(class_names, quick=args.quick)
        analysis_results = run_feature_analysis(class_names)

    run_visualizations(class_names, benchmark_results, robustness_results, analysis_results)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  DONE — Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
