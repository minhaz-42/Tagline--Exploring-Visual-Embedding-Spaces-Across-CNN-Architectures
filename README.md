# Tagline — Exploring Visual Embedding Spaces Across CNN Architectures

A Django-based web application and research pipeline for comparative analysis of visual feature embeddings extracted from three influential CNN architectures: **ResNet-101**, **ZFNet**, and **GoogLeNet**. The system performs content-based image retrieval (CBIR) on a curated subset of the Caltech-101 dataset and provides quantitative evaluation through retrieval metrics and t-SNE visualizations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Django](https://img.shields.io/badge/Django-5.x%2F6.x-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Problem Statement

Content-based image retrieval (CBIR) systems rely on deep feature representations to measure visual similarity between images. While numerous CNN architectures have been proposed, each produces embedding spaces with fundamentally different geometric and structural properties due to variations in network depth, architectural design (residual connections, inception modules, deconvolution layers), and embedding dimensionality (1024-d to 4096-d). 

**The core problem** is the lack of a unified, interactive framework that enables direct comparison of how different CNN architectures organize visual information in their embedding spaces and how these structural differences impact retrieval quality. Existing studies typically evaluate models in isolation or focus solely on classification accuracy, overlooking the nuanced differences in embedding space topology that critically affect similarity-based retrieval tasks.

This research addresses: *How do architectural choices in CNNs — specifically residual learning (ResNet-101), deconvolution-guided feature learning (ZFNet), and multi-scale inception modules (GoogLeNet) — affect the structure, quality, and discriminative power of visual embedding spaces for content-based image retrieval?*

## Novelty

1. **Cross-Architecture Embedding Space Analysis**: Unlike prior work that benchmarks CNNs on classification tasks, this project provides a systematic comparison of embedding space geometry across three architecturally distinct CNNs through both quantitative metrics and interactive t-SNE visualizations.

2. **Unified Interactive Framework**: A full-stack web application that allows real-time querying against all three embedding spaces simultaneously, enabling researchers and practitioners to visually inspect how architectural choices affect retrieval behaviour on identical queries.

3. **Multi-Dimensional Evaluation Protocol**: Combines leave-one-out Top-K Precision with Mean Average Precision (mAP) across the full ranked list, providing complementary views of retrieval quality that capture both local neighbourhood accuracy and global ranking consistency.

4. **Dimensionality-Aware Comparison**: By comparing embeddings of significantly different dimensionalities (1024-d, 2048-d, 4096-d), the framework reveals the trade-off between representation capacity and retrieval efficiency, demonstrating that higher dimensionality does not necessarily yield superior retrieval performance.

5. **Reproducible Research Pipeline**: End-to-end pipeline from dataset curation through embedding extraction, evaluation, and visualization — enabling full reproducibility and extensibility to additional architectures.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Django Web Application                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Landing   │  │  Search   │  │ Metrics  │  │ Dashboard  │  │
│  │  Page     │  │   Page    │  │   Page   │  │ & History  │  │
│  └──────────┘  └─────┬─────┘  └──────────┘  └────────────┘  │
│                       │                                        │
│              ┌────────▼────────┐                              │
│              │  Feature Pipeline │                             │
│              │  ┌──────────────┐│                             │
│              │  │ ResNet-101   ││                             │
│              │  │ ZFNet        ││──→ Cosine Similarity        │
│              │  │ GoogLeNet    ││    KNN Retrieval            │
│              │  └──────────────┘│                             │
│              └──────────────────┘                             │
└──────────────────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                  ┌──────▼──────┐
    │Caltech-101│                │  Embeddings  │
    │ 8 Classes │                │  .npy files  │
    │ ~1148 imgs│                │  + JSON/PNG  │
    └──────────┘                └──────────────┘
```

## Features

- **Image Retrieval**: Upload any image and retrieve the top-10 most visually similar images using cosine similarity
- **3 CNN Models**: Compare ResNet-101 (2048-d), ZFNet (4096-d), and GoogLeNet (1024-d)
- **Evaluation Metrics**: Top-10 retrieval accuracy and mean average precision (mAP) with visual progress bars
- **t-SNE Visualizations**: 2D projections of embedding spaces colour-coded by class
- **User Authentication**: Register/login to track search history on a personal dashboard
- **Modern 3D UI**: Clean light theme with glassmorphism, depth shadows, and smooth animations
- **Responsive Design**: Works across desktop, tablet, and mobile devices

## Dataset

8 selected classes from **Caltech-101** (~1,148 images total):

| Class | Approx. Count |
|-------|:------------:|
| Accordion | ~55 |
| Airplane | ~250 |
| Camera | ~50 |
| Elephant | ~65 |
| Laptop | ~82 |
| Motorbike | ~250 |
| Watch | ~239 |
| Wheelchair | ~59 |

## Results

| Model | Embedding Dim | Top-10 Accuracy | mAP |
|-------|:------------:|:--------------:|:---:|
| **ResNet-101** | 2048-d | **0.9942** | **0.9768** |
| GoogLeNet | 1024-d | 0.9802 | 0.9336 |
| ZFNet | 4096-d | 0.9566 | 0.8623 |

**Key Finding**: ResNet-101 achieves the highest retrieval quality despite having a mid-range embedding dimensionality, demonstrating that architectural depth and skip connections produce more discriminative feature spaces than higher-dimensional alternatives.

## Quick Start

### Prerequisites

- Python 3.10+
- pip / virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tagline-visual-embeddings.git
cd tagline-visual-embeddings

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install django torch torchvision scikit-learn matplotlib numpy tqdm pillow

# Navigate to project
cd feature_understanding_project

# Run database migrations
python manage.py migrate
```

### Download Dataset & Run Pipeline

```bash
# Download Caltech-101 (8 selected classes)
python manage.py download_caltech

# Run the offline embedding extraction pipeline
python run_pipeline.py --dataset_dir ./dataset
```

This will:
1. Load images from 8 Caltech-101 classes
2. Extract embeddings using all 3 CNN models
3. Compute retrieval metrics (Top-10 accuracy, mAP)
4. Generate t-SNE visualization plots

### Start the Server

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000** in your browser.

## Project Structure

```
feature_understanding_project/
├── feature_understanding_project/   # Django settings & config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── retrieval/                       # Main Django app
│   ├── feature_pipeline/            # ML pipeline modules
│   │   ├── dataset_loader.py        # Caltech-101 data loading
│   │   ├── feature_extractor.py     # CNN model builders & extraction
│   │   ├── nearest_neighbors.py     # KNN retrieval with cosine similarity
│   │   ├── evaluation.py            # Top-K accuracy & mAP computation
│   │   └── visualize_tsne.py        # t-SNE plot generation
│   ├── templates/                   # HTML templates
│   │   ├── base.html                # Base template with 3D nav & footer
│   │   ├── landing.html             # Project landing page
│   │   ├── index.html               # Search/upload page
│   │   ├── results.html             # Retrieval results grid
│   │   ├── metrics.html             # Evaluation metrics & t-SNE
│   │   ├── dashboard.html           # User dashboard
│   │   ├── history.html             # Search history
│   │   └── registration/            # Login & register templates
│   ├── static/
│   │   ├── css/style.css            # 3D light theme stylesheet
│   │   ├── tsne_plots/              # Generated t-SNE PNGs
│   │   └── uploads/                 # User-uploaded query images
│   ├── models.py                    # SearchHistory model
│   ├── views.py                     # All view functions
│   ├── urls.py                      # URL routing
│   └── forms.py                     # Registration form
├── dataset/                         # Caltech-101 subset (8 classes)
├── embeddings/                      # Pre-computed .npy embeddings
│   ├── resnet_embeddings.npy
│   ├── zfnet_embeddings.npy
│   ├── googlenet_embeddings.npy
│   ├── labels.npy
│   ├── image_paths.json
│   └── evaluation_results.json
├── run_pipeline.py                  # Offline pipeline script
└── manage.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Django 5.x / 6.x |
| ML Framework | PyTorch + torchvision |
| Retrieval | scikit-learn (NearestNeighbors, cosine) |
| Visualization | Matplotlib + t-SNE |
| Frontend | HTML5, CSS3 (custom 3D light theme) |
| Database | SQLite (search history) |
| Dataset | Caltech-101 (8-class subset) |

## API / Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with project overview |
| `/search/` | Image upload & model selection |
| `/search/go/` | POST endpoint for retrieval |
| `/metrics/` | Evaluation metrics table & t-SNE |
| `/dashboard/` | User dashboard (auth required) |
| `/history/` | Search history (auth required) |
| `/login/` | User login |
| `/register/` | User registration |

## Extending

To add a new CNN architecture:

1. Add a builder function in `retrieval/feature_pipeline/feature_extractor.py`
2. Register it in `MODEL_BUILDERS` dict
3. Add config entry in `settings.py` → `MODEL_CONFIGS`
4. Re-run `python run_pipeline.py`

## License

This project is for academic/research purposes. The Caltech-101 dataset is subject to its original license terms.

## Acknowledgements

- [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02) — Fei-Fei et al.
- [PyTorch](https://pytorch.org/) — Pre-trained CNN models
- [scikit-learn](https://scikit-learn.org/) — NearestNeighbors, t-SNE
- [Django](https://djangoproject.com/) — Web framework
