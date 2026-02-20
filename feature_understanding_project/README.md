# Tagline — Exploring Visual Embedding Spaces Across CNN Architectures

> A research-grade Django web application for **content-based image retrieval (CBIR)** that extracts, compares, and analyses deep CNN embeddings from ResNet-101, ZFNet, and GoogLeNet. Designed for conference/journal-level reproducibility.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Dataset Preparation](#dataset-preparation)
7. [Running the Offline Pipeline](#running-the-offline-pipeline)
8. [Running the Web Application](#running-the-web-application)
9. [Research Modules](#research-modules)
10. [Web Interface Pages](#web-interface-pages)
11. [Evaluation Methodology](#evaluation-methodology)
12. [Technology Stack](#technology-stack)
13. [Configuration](#configuration)
14. [Citation](#citation)
15. [License](#license)

---

## Overview

**Tagline** is an interactive tool for exploring how different CNN architectures represent visual information in their embedding spaces. Given a query image, the system extracts a feature vector with a chosen model and retrieves the 10 most similar images from a pre-indexed database using cosine similarity.

Beyond simple retrieval, the project provides a full **statistical benchmarking suite** — leave-one-out evaluation, stratified train/test splits, paired t-tests with Bonferroni correction, robustness testing under five families of image perturbations, and detailed feature-space analysis (Fisher discriminant ratios, normalization/metric ablations, runtime benchmarks).

All results are presented in a professional **dark-themed web UI** with sidebar navigation, research-grade plot rendering, and paper-ready export capabilities.

Built with **PyTorch** · **Django** · **scikit-learn** · **SciPy** · **Matplotlib** · **Seaborn**

---

## Key Features

| Area | Capability |
|------|-----------|
| **Retrieval** | Top-10 cosine-similarity search with per-query confidence metrics |
| **Multi-Model Compare** | Side-by-side retrieval across all 3 CNNs from a single query image |
| **Statistical Benchmark** | Leave-one-out, 5-fold stratified splits, paired t-tests, 95 % CIs |
| **Robustness Testing** | Rotation, Gaussian noise, blur, brightness, contrast perturbations |
| **Feature Analysis** | Intra/inter-class distances, Fisher ratios, confusion matrices, normalization ablation, similarity-metric comparison |
| **Advanced Visualization** | Dark-themed t-SNE plots, degradation curves, performance bars, confusion matrices (PNG + PDF) |
| **Professional UI** | Deep navy/indigo dark theme, fixed sidebar navigation, responsive layout |
| **Search History** | Persistent per-user history with revisitable result pages |
| **Authentication** | Registration + login system with user-scoped dashboards |

---

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  Browser UI  │◄───►│  Django Web Server   │◄───►│  SQLite3 DB     │
│  (Dark theme)│     │  (views, templates)  │     │  (history, users)│
└─────────────┘     └────────┬────────────┘     └─────────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Feature Pipeline    │
                    │  ┌────────────────┐  │
                    │  │ feature_extractor│  │     ┌──────────────────┐
                    │  │ (PyTorch CNNs) │◄─┼─────│ Pre-computed     │
                    │  └────────────────┘  │     │ .npy embeddings  │
                    │  ┌────────────────┐  │     │ + labels + paths │
                    │  │nearest_neighbors│  │     └──────────────────┘
                    │  │ (scikit-learn) │  │
                    │  └────────────────┘  │
                    │  ┌────────────────┐  │
                    │  │stat_benchmark  │  │     ┌──────────────────┐
                    │  │robustness      │  │────►│  JSON results    │
                    │  │feature_analysis│  │     │  + PNG/PDF plots │
                    │  │advanced_viz    │  │     └──────────────────┘
                    │  └────────────────┘  │
                    └─────────────────────┘
```

### CNN Models

| Model | Layers | Embedding Dim | Source Layer |
|-------|--------|--------------|-------------|
| **ResNet-101** | 101 | 2048 | Global average pooling |
| **ZFNet** (AlexNet proxy) | 8 | 4096 | FC-7 (classifier[4]) |
| **GoogLeNet** (Inception v1) | 22 | 1024 | Global average pooling |

All models use ImageNet-pretrained weights from `torchvision.models`.

---

## Project Structure

```
feature_understanding_project/
├── manage.py                          # Django management CLI
├── run_pipeline.py                    # Offline: extract embeddings + evaluate
├── run_cifar100_pipeline.py           # Offline: CIFAR-100 extraction
├── run_full_benchmark.py              # Offline: full statistical benchmark
├── generate_tsne.py                   # Offline: t-SNE plot generation
├── requirements.txt
├── README.md
│
├── feature_understanding_project/     # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── retrieval/                         # Main Django app
│   ├── views.py                       # All view functions
│   ├── urls.py                        # URL routing
│   ├── models.py                      # SearchHistory, QueryImage
│   ├── forms.py                       # RegisterForm
│   ├── admin.py
│   │
│   ├── feature_pipeline/             # Core ML modules
│   │   ├── feature_extractor.py       #   CNN model loading + embedding extraction
│   │   ├── nearest_neighbors.py       #   Index building + cosine retrieval
│   │   ├── dataset_loader.py          #   Image loading + preprocessing
│   │   ├── evaluation.py              #   Leave-one-out evaluation + mAP
│   │   ├── statistical_benchmark.py   #   t-tests, CIs, cross-model comparison
│   │   ├── robustness.py              #   Perturbation robustness evaluation
│   │   ├── feature_analysis.py        #   Feature space quality analysis
│   │   └── advanced_viz.py            #   Dark-themed publication-ready plots
│   │
│   ├── templates/
│   │   ├── base.html                  # Sidebar layout shell
│   │   ├── landing.html               # Full-width project landing page
│   │   ├── index.html                 # Search (upload + model select)
│   │   ├── results.html               # Retrieval results + metrics
│   │   ├── compare.html               # Multi-model comparison upload
│   │   ├── compare_results.html       # Side-by-side comparison results
│   │   ├── metrics.html               # Evaluation metrics + t-SNE plots
│   │   ├── benchmark.html             # Statistical benchmark results
│   │   ├── robustness.html            # Robustness test results
│   │   ├── analysis.html              # Feature space analysis
│   │   ├── dashboard.html             # User dashboard
│   │   ├── history.html               # Search history table
│   │   └── registration/
│   │       ├── login.html
│   │       └── register.html
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── dark-theme.css         # Complete dark UI theme (~1100 lines)
│   │   ├── tsne_plots/                # Generated t-SNE images
│   │   └── plots/                     # Generated benchmark/analysis plots
│   │
│   └── templatetags/
│       └── retrieval_tags.py          # Custom tags (dataset_image, class_name)
│
├── dataset/                           # Image dataset (class folders)
│   ├── accordion/
│   ├── airplane/
│   ├── camera/
│   └── ...                            # 8+ classes from Caltech-101 / CIFAR-100
│
└── embeddings/                        # Pre-computed data
    ├── resnet_embeddings.npy
    ├── zfnet_embeddings.npy
    ├── googlenet_embeddings.npy
    ├── labels.npy
    ├── image_paths.json
    ├── class_names.json
    ├── evaluation_results.json
    ├── benchmark_results.json         # Generated by run_full_benchmark.py
    ├── robustness_results.json        # Generated by run_full_benchmark.py
    └── analysis_results.json          # Generated by run_full_benchmark.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip (or conda)
- ~4 GB disk space for PyTorch + model weights

### 1. Create & Activate Virtual Environment

```bash
cd feature_understanding_project
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you need a platform-specific PyTorch wheel (e.g. CUDA / MPS), follow [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 3. Initialize Django

```bash
python manage.py migrate
python manage.py createsuperuser   # optional
```

---

## Dataset Preparation

Place class-labelled images in the `dataset/` directory. Each sub-folder name becomes the class label:

```
dataset/
├── accordion/    (~250–300 images)
├── airplane/
├── camera/
├── elephant/
├── laptop/
├── motorbike/
├── watch/
└── wheelchair/
```

Each folder should contain `.jpg` / `.png` images. The pipeline resizes everything to 224×224 and applies ImageNet normalisation.

**Download link:** [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02)

You can also use the included management command:

```bash
python manage.py download_caltech --target-dir ./dataset --max-per-class 300
```

To change classes, edit `SELECTED_CLASSES` in `feature_understanding_project/settings.py`.

---

## Running the Offline Pipeline

### 1. Extract Embeddings & Basic Evaluation

```bash
python run_pipeline.py --dataset_dir ./dataset
```

This will:
1. Load all images from the dataset
2. Extract embeddings with ResNet-101, ZFNet, and GoogLeNet
3. Run leave-one-out evaluation (Top-10 accuracy + mAP)
4. Generate t-SNE visualisation plots
5. Save everything to `embeddings/`

**Optional arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | `./dataset` | Path to dataset root |
| `--embeddings_dir` | `./embeddings` | Where to store .npy files |
| `--tsne_dir` | `./retrieval/static/tsne_plots/` | Where to save t-SNE PNGs |
| `--batch_size` | `32` | Batch size for extraction |
| `--num_workers` | `4` | DataLoader workers |
| `--max_per_class` | `300` | Max images per class |

### 2. Full Statistical Benchmark (research-grade)

```bash
python run_full_benchmark.py
```

Runs the complete research pipeline:
1. **Statistical benchmark** — LOO + 5-fold stratified splits + paired t-tests + CIs
2. **Robustness evaluation** — 5 perturbation types × 3 severities × 3 models
3. **Feature space analysis** — class distances, Fisher ratios, confusion matrices, normalization/metric ablation, runtime
4. **Visualization generation** — all dark-themed plots as PNG + PDF

Options:

| Flag | Description |
|------|-------------|
| `--quick` | Use 100 images instead of 500 for faster iteration |
| `--skip-robustness` | Skip the perturbation tests (most time-consuming step) |
| `--plots-only` | Re-generate plots from existing JSON results |

```bash
# Quick test run
python run_full_benchmark.py --quick --skip-robustness

# Regenerate plots only
python run_full_benchmark.py --plots-only
```

### 3. CIFAR-100 Pipeline (alternative dataset)

```bash
python run_cifar100_pipeline.py --max_per_class 50 --batch_size 64 --save_per_class 30
```

---

## Running the Web Application

```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Research Modules

### `statistical_benchmark.py`

| Function | Description |
|----------|-------------|
| `leave_one_out_evaluation()` | LOO Top-K accuracy and mAP for a single model |
| `train_test_split_evaluation()` | Stratified 80/20 split evaluation, repeated 5 times |
| `paired_t_test()` | Two-tailed paired t-test between model accuracy vectors |
| `confidence_interval()` | 95 % CI from split-based accuracy estimates |
| `cross_model_comparison()` | Run all of the above across all 3 models |
| `export_results_csv()` / `export_results_json()` | Save results for downstream use |

### `robustness.py`

| Perturbation | Severities |
|-------------|-----------|
| Rotation | 15°, 45°, 90° |
| Gaussian Noise | σ = 0.01, 0.05, 0.1 |
| Gaussian Blur | kernel = 3, 5, 7 |
| Brightness | ×0.5, ×1.5, ×2.0 |
| Contrast | ×0.5, ×1.5, ×2.0 |

For each combination, the module re-extracts embeddings from perturbed images, queries against the clean database, and reports Top-1/5/10 accuracy and average similarity.

### `feature_analysis.py`

| Analysis | Output |
|----------|--------|
| `compute_class_distances()` | Intra/inter-class cosine distances + Fisher discriminant ratio per class |
| `knn_confusion_matrix()` | k-NN confusion matrix (leave-one-out) |
| `embedding_statistics()` | Dimensionality, mean norm, variance, sparsity |
| `normalization_comparison()` | None vs L2 vs Min-Max normalization |
| `similarity_metric_comparison()` | Cosine vs Euclidean vs Manhattan |
| `runtime_benchmark()` | Feature extraction + search latency |

### `advanced_viz.py`

All plots use a consistent dark theme matching the web UI colour palette:

| Plot | Function |
|------|----------|
| t-SNE scatter | `plot_tsne_dark()` |
| Confusion matrix | `plot_confusion_matrix()` |
| Performance bars | `plot_performance_comparison()` |
| Robustness curves | `plot_robustness_degradation()` |
| Distance histograms | `plot_distance_histograms()` |
| Dim vs performance | `plot_dim_vs_performance()` |
| Runtime bars | `plot_runtime_comparison()` |

Outputs: PNG (300 DPI) + PDF in `retrieval/static/plots/`.

---

## Web Interface Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Full-width project overview with model cards |
| `/search/` | Search | Upload form with model selection and drag & drop |
| `/search/go/` | Results | Top-10 results with per-query metrics |
| `/compare/` | Compare | Upload form for all-model comparison |
| `/compare/go/` | Compare Results | Side-by-side results from 3 models |
| `/metrics/` | Metrics | Evaluation table + t-SNE plots + last query metrics |
| `/benchmark/` | Benchmark | Statistical benchmark tables + CI + t-tests |
| `/robustness/` | Robustness | Perturbation robustness tables + degradation plots |
| `/analysis/` | Analysis | Feature space analysis tables + plots |
| `/dashboard/` | Dashboard | User activity overview (login required) |
| `/history/` | History | Searchable history table with result links |
| `/login/` | Login | Authentication |
| `/register/` | Register | Account creation |

---

## Evaluation Methodology

### Top-K Accuracy (Precision@K)

For each image *q* in the dataset, we treat it as a query (leave-one-out), retrieve the *K* nearest neighbours by cosine similarity, and compute:

> P@K(q) = |{r ∈ Top-K(q) : class(r) = class(q)}| / K

### Mean Average Precision (mAP)

> AP(q) = (1 / |Rel(q)|) × Σ_{k=1}^{N} P@k(q) · rel(k)
>
> mAP = (1 / |Q|) × Σ_{q ∈ Q} AP(q)

### Paired t-Test

We use a two-tailed paired t-test with Bonferroni correction (α = 0.05/3) to determine whether accuracy differences between model pairs are statistically significant.

### Fisher Discriminant Ratio

> FDR(c) = d_inter(c) / d_intra(c)

Higher values indicate better class separability in the embedding space.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Django 4.2+ |
| Deep Learning | PyTorch 2.0+, torchvision |
| ML Utilities | scikit-learn 1.3+ |
| Statistical Tests | SciPy 1.11+ |
| Visualization | Matplotlib, Seaborn |
| Data Processing | NumPy, Pandas, Pillow |
| Database | SQLite3 |
| Frontend | Custom CSS (dark theme), vanilla JavaScript |

---

## Configuration

All project-level settings are in `feature_understanding_project/settings.py`:

| Setting | Description |
|---------|-------------|
| `SELECTED_CLASSES` | List of class names matching dataset folder names |
| `MODEL_CONFIGS` | CNN model definitions (name, dim, file paths) |
| `DATASET_DIR` | Path to the image dataset |
| `EMBEDDINGS_DIR` | Path to pre-computed embeddings |
| `UPLOADS_DIR` | Path for user-uploaded query images |
| `TSNE_PLOTS_DIR` | Path for t-SNE plot images |
| `PLOTS_DIR` | Path for benchmark/analysis plots |

---

## Selected Caltech-101 Classes

1. accordion
2. airplane
3. camera
4. elephant
5. laptop
6. motorbike
7. watch
8. wheelchair

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{tagline2025,
  title   = {Tagline: Exploring Visual Embedding Spaces Across CNN Architectures},
  author  = {Tanvir},
  year    = {2025},
  note    = {GitHub repository},
}
```

---

## License

This project is for academic and research purposes.
