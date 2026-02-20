"""
statistical_benchmark.py
------------------------
Comprehensive statistical benchmarking for multi-backbone CNN retrieval.

Provides:
  - 80/20 train-test split evaluation
  - Leave-one-out cross-validation
  - Per-query metrics: Precision@K, Recall@K, F1@K, AP
  - Aggregate metrics: Top-1/5/10 accuracy, mAP
  - Cross-model paired t-test with 95 % confidence intervals
  - CSV / JSON export of publication-ready tables
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit


# ═══════════════════════════════════════════════
# Per-query helpers
# ═══════════════════════════════════════════════

def _precision_at_k(retrieved_labels: np.ndarray, query_label: int, k: int) -> float:
    """Fraction of top-K neighbours sharing the query label."""
    return float(np.sum(retrieved_labels[:k] == query_label)) / k


def _recall_at_k(retrieved_labels: np.ndarray, query_label: int, k: int, n_relevant: int) -> float:
    """Fraction of relevant items found in the top-K."""
    if n_relevant == 0:
        return 0.0
    return float(np.sum(retrieved_labels[:k] == query_label)) / n_relevant


def _average_precision(retrieved_labels: np.ndarray, query_label: int, n_relevant: int) -> float:
    """Average precision for a single query over the full ranked list."""
    if n_relevant == 0:
        return 0.0
    relevant = (retrieved_labels == query_label).astype(np.float64)
    cumsum = np.cumsum(relevant)
    precision_at_rank = cumsum / np.arange(1, len(relevant) + 1, dtype=np.float64)
    return float(np.sum(precision_at_rank * relevant) / n_relevant)


# ═══════════════════════════════════════════════
# Leave-one-out evaluation
# ═══════════════════════════════════════════════

def leave_one_out_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict:
    """
    Full leave-one-out evaluation.

    Returns a dict with aggregate metrics *and* per-query AP scores
    (needed for paired t-tests across models).
    """
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    n = len(labels)

    per_query_ap: List[float] = []
    per_query_topk: Dict[int, List[float]] = {k: [] for k in k_values}
    per_query_precision: Dict[int, List[float]] = {k: [] for k in k_values}
    per_query_recall: Dict[int, List[float]] = {k: [] for k in k_values}

    for i in range(n):
        sorted_idx = np.argsort(sim_matrix[i])[::-1]
        retrieved = labels[sorted_idx]
        n_relevant = int(np.sum(labels == labels[i])) - 1  # exclude self

        ap = _average_precision(retrieved, labels[i], n_relevant)
        per_query_ap.append(ap)

        for k in k_values:
            prec = _precision_at_k(retrieved, labels[i], k)
            rec = _recall_at_k(retrieved, labels[i], k, n_relevant)
            per_query_precision[k].append(prec)
            per_query_recall[k].append(rec)
            # Top-K "accuracy" = 1 if the query class appears at least once in top-K
            per_query_topk[k].append(1.0 if np.any(retrieved[:k] == labels[i]) else 0.0)

    results: Dict = {
        "mAP": round(float(np.mean(per_query_ap)), 6),
        "per_query_ap": per_query_ap,  # keep for t-test
    }

    for k in k_values:
        prec_arr = np.array(per_query_precision[k])
        rec_arr = np.array(per_query_recall[k])
        f1_arr = np.where(
            (prec_arr + rec_arr) > 0,
            2 * prec_arr * rec_arr / (prec_arr + rec_arr),
            0.0,
        )
        results[f"top_{k}_accuracy"] = round(float(np.mean(per_query_topk[k])), 6)
        results[f"precision_at_{k}"] = round(float(np.mean(prec_arr)), 6)
        results[f"recall_at_{k}"] = round(float(np.mean(rec_arr)), 6)
        results[f"f1_at_{k}"] = round(float(np.mean(f1_arr)), 6)

    return results


# ═══════════════════════════════════════════════
# 80/20 train-test split evaluation
# ═══════════════════════════════════════════════

def train_test_split_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    k_values: Tuple[int, ...] = (1, 5, 10),
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Stratified 80/20 split evaluation repeated over *n_splits* folds.

    The train set is used as the retrieval database; test images are queries.
    Returns mean ± std across splits.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    fold_results: List[Dict] = []

    for train_idx, test_idx in splitter.split(embeddings, labels):
        train_emb = embeddings[train_idx]
        train_lbl = labels[train_idx]
        test_emb = embeddings[test_idx]
        test_lbl = labels[test_idx]

        sim = cosine_similarity(test_emb, train_emb)  # (n_test, n_train)

        fold: Dict = {}
        per_query_ap: List[float] = []

        for qi in range(len(test_lbl)):
            sorted_idx = np.argsort(sim[qi])[::-1]
            retrieved = train_lbl[sorted_idx]
            n_relevant = int(np.sum(train_lbl == test_lbl[qi]))
            per_query_ap.append(_average_precision(retrieved, test_lbl[qi], n_relevant))

            for k in k_values:
                prec = _precision_at_k(retrieved, test_lbl[qi], k)
                rec = _recall_at_k(retrieved, test_lbl[qi], k, n_relevant)
                fold.setdefault(f"prec_{k}", []).append(prec)
                fold.setdefault(f"rec_{k}", []).append(rec)
                hit = 1.0 if np.any(retrieved[:k] == test_lbl[qi]) else 0.0
                fold.setdefault(f"top_{k}", []).append(hit)

        fold_summary = {"mAP": float(np.mean(per_query_ap))}
        for k in k_values:
            p = np.mean(fold[f"prec_{k}"])
            r = np.mean(fold[f"rec_{k}"])
            fold_summary[f"top_{k}_accuracy"] = float(np.mean(fold[f"top_{k}"]))
            fold_summary[f"precision_at_{k}"] = float(p)
            fold_summary[f"recall_at_{k}"] = float(r)
            fold_summary[f"f1_at_{k}"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        fold_results.append(fold_summary)

    # Aggregate across folds: mean ± std
    metric_keys = list(fold_results[0].keys())
    aggregated: Dict = {}
    for mk in metric_keys:
        vals = [f[mk] for f in fold_results]
        aggregated[f"{mk}_mean"] = round(float(np.mean(vals)), 6)
        aggregated[f"{mk}_std"] = round(float(np.std(vals)), 6)
    aggregated["n_splits"] = n_splits
    aggregated["test_size"] = test_size
    return aggregated


# ═══════════════════════════════════════════════
# Cross-model comparison + statistical tests
# ═══════════════════════════════════════════════

def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """Paired t-test on per-query AP scores of two models."""
    a, b = np.array(scores_a), np.array(scores_b)
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se = float(np.std(diff, ddof=1)) / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se)
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "mean_difference": round(mean_diff, 6),
        "ci_95_lower": round(float(ci_95[0]), 6),
        "ci_95_upper": round(float(ci_95[1]), 6),
        "significant_at_005": bool(p_value < 0.05),
    }


def confidence_interval(scores: List[float], confidence: float = 0.95) -> Dict:
    """Compute confidence interval for a list of scores."""
    arr = np.array(scores)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1)) / np.sqrt(n)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return {
        "mean": round(mean, 6),
        "std": round(float(np.std(arr, ddof=1)), 6),
        "ci_lower": round(float(ci[0]), 6),
        "ci_upper": round(float(ci[1]), 6),
        "n": n,
    }


def cross_model_comparison(
    embeddings_dir: str | Path,
    model_keys: Tuple[str, ...] = ("resnet", "zfnet", "googlenet"),
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict:
    """
    Run full benchmark across all models and compute pairwise t-tests.
    """
    emb_dir = Path(embeddings_dir)
    labels = np.load(emb_dir / "labels.npy")

    model_results: Dict = {}
    per_query_aps: Dict[str, List[float]] = {}

    for key in model_keys:
        emb_path = emb_dir / f"{key}_embeddings.npy"
        if not emb_path.exists():
            print(f"[Benchmark] Skipping {key}: embeddings not found")
            continue

        print(f"[Benchmark] Evaluating {key} ...")
        embeddings = np.load(emb_path)

        t0 = time.time()
        loo = leave_one_out_evaluation(embeddings, labels, k_values)
        loo_time = time.time() - t0

        t0 = time.time()
        split = train_test_split_evaluation(embeddings, labels, k_values=k_values)
        split_time = time.time() - t0

        per_query_aps[key] = loo.pop("per_query_ap")
        ci = confidence_interval(per_query_aps[key])

        model_results[key] = {
            "leave_one_out": loo,
            "train_test_split": split,
            "mAP_confidence_interval": ci,
            "loo_time_seconds": round(loo_time, 2),
            "split_time_seconds": round(split_time, 2),
        }

    # Pairwise t-tests
    pairwise: Dict = {}
    keys_list = list(per_query_aps.keys())
    for i in range(len(keys_list)):
        for j in range(i + 1, len(keys_list)):
            a, b = keys_list[i], keys_list[j]
            pair_key = f"{a}_vs_{b}"
            pairwise[pair_key] = paired_t_test(per_query_aps[a], per_query_aps[b])

    return {
        "model_results": model_results,
        "pairwise_tests": pairwise,
        "model_keys": list(model_results.keys()),
        "k_values": list(k_values),
    }


# ═══════════════════════════════════════════════
# Export utilities
# ═══════════════════════════════════════════════

def export_results_csv(results: Dict, output_path: str | Path) -> str:
    """Export cross-model comparison results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, data in results.get("model_results", {}).items():
        loo = data.get("leave_one_out", {})
        split = data.get("train_test_split", {})
        ci = data.get("mAP_confidence_interval", {})
        row = {
            "model": key,
            "loo_mAP": loo.get("mAP", ""),
            "loo_top_1_acc": loo.get("top_1_accuracy", ""),
            "loo_top_5_acc": loo.get("top_5_accuracy", ""),
            "loo_top_10_acc": loo.get("top_10_accuracy", ""),
            "loo_precision_10": loo.get("precision_at_10", ""),
            "loo_recall_10": loo.get("recall_at_10", ""),
            "loo_f1_10": loo.get("f1_at_10", ""),
            "split_mAP_mean": split.get("mAP_mean", ""),
            "split_mAP_std": split.get("mAP_std", ""),
            "mAP_ci_lower": ci.get("ci_lower", ""),
            "mAP_ci_upper": ci.get("ci_upper", ""),
        }
        rows.append(row)

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return str(output_path)


def export_results_json(results: Dict, output_path: str | Path) -> str:
    """Export full benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove per-query arrays for cleaner JSON
    clean = json.loads(json.dumps(results, default=str))
    for key in clean.get("model_results", {}):
        clean["model_results"][key].pop("per_query_ap", None)

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    return str(output_path)
