# Tagline — Exploring Visual Embedding Spaces Across CNN Architectures

A complete research + web system for comparing feature embeddings from pretrained CNN models (**ResNet-101**, **ZFNet**, **GoogLeNet**) using the **Caltech-101** dataset (8 selected classes).

Built with **PyTorch** · **Django** · **Scikit-learn** · **Matplotlib**

---

## Project Structure

```
feature_understanding_project/
│
├── manage.py                          # Django management script
├── run_pipeline.py                    # Offline embedding extraction script
├── requirements.txt                   # Python dependencies
│
├── feature_understanding_project/     # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── __init__.py
│
├── retrieval/                         # Django app
│   ├── views.py                       # Homepage, search, metrics views
│   ├── models.py                      # Django ORM models
│   ├── urls.py                        # App URL routes
│   ├── apps.py
│   ├── templates/
│   │   ├── index.html                 # Upload & model selection form
│   │   ├── results.html               # Top-10 retrieval results
│   │   └── metrics.html               # Evaluation table + t-SNE plots
│   ├── templatetags/
│   │   └── retrieval_tags.py          # Custom tags for serving dataset images
│   ├── static/
│   │   ├── css/style.css              # Global stylesheet
│   │   ├── tsne_plots/                # Generated t-SNE PNGs
│   │   └── uploads/                   # User-uploaded query images
│   └── feature_pipeline/
│       ├── dataset_loader.py          # Caltech-101 dataset loading
│       ├── feature_extractor.py       # CNN model builders & embedding extraction
│       ├── nearest_neighbors.py       # Cosine similarity retrieval
│       ├── evaluation.py              # Top-K accuracy & mAP computation
│       └── visualize_tsne.py          # t-SNE plot generation
│
├── dataset/                           # Place Caltech-101 images here
│   ├── accordion/
│   ├── airplane/
│   ├── camera/
│   ├── elephant/
│   ├── laptop/
│   ├── motorbike/
│   ├── watch/
│   └── wheelchair/
│
└── embeddings/                        # Generated embedding files (auto-created)
    ├── resnet_embeddings.npy
    ├── zfnet_embeddings.npy
    ├── googlenet_embeddings.npy
    ├── labels.npy
    ├── image_paths.json
    └── evaluation_results.json
```

---

## Setup Instructions

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

### 3. Prepare the Dataset

Download the **Caltech-101** dataset and place the 8 selected class folders inside `dataset/`:

```
dataset/
    accordion/    (~250–300 images)
    airplane/
    camera/
    elephant/
    laptop/
    motorbike/
    watch/
    wheelchair/
```

Each folder should contain `.jpg` / `.png` images. The pipeline resizes everything to 224×224 and applies ImageNet normalisation.

**Download link:** [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02)

After downloading, extract and copy only the 8 required class folders into `dataset/`.

### 4. Run the Offline Pipeline

This extracts embeddings for all three models, computes metrics, and generates t-SNE plots.

If you don't yet have the Caltech-101 files locally you can use the included management command to download and prepare the 8 selected classes. NOTE: run the command in the same Python environment that has `torch`/`torchvision` installed (activate your venv first).

```bash
# Download & copy the 8 selected Caltech-101 classes into ./dataset/
python manage.py download_caltech --target-dir ./dataset --max-per-class 300

# Then run the offline pipeline
python run_pipeline.py --dataset_dir ./dataset
```

Important: the offline pipeline and the web *search* feature require the ML dependencies (`torch`, `torchvision`). If you see an error mentioning `torch` or `torchvision` when using the web UI, install the requirements (or follow the PyTorch installation guide for CUDA/MPS):

```bash
# inside your activated virtual environment
pip install -r requirements.txt
# if you need a platform-specific PyTorch wheel, follow:
# https://pytorch.org/get-started/locally/
```

Run the pipeline:

```bash
python run_pipeline.py --dataset_dir ./dataset
```

**Optional arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_dir` | `./dataset` | Path to dataset root |
| `--embeddings_dir` | `./embeddings` | Where to store .npy files |
| `--tsne_dir` | `./retrieval/static/tsne_plots/` | Where to save t-SNE PNGs |
| `--batch_size` | `32` | Batch size for extraction |
| `--num_workers` | `4` | DataLoader workers |
| `--max_per_class` | `300` | Max images per class |

The pipeline will:
1. Load ~250–300 images per class (≈2000 total)
2. Extract embeddings: ResNet (2048-d), ZFNet (4096-d), GoogLeNet (1024-d)
3. Save `.npy` files to `embeddings/`
4. Compute Top-10 retrieval accuracy and mAP
5. Generate t-SNE scatter plots

### 5. Run Django Migrations

```bash
python manage.py migrate
```

### 6. Start the Django Server

```bash
python manage.py runserver
```

Visit: **http://127.0.0.1:8000/**

---

## Using the Web App

### Image Retrieval (Homepage — `/`)

1. Upload any image (JPG, PNG, BMP)
2. Select a CNN model from the radio cards:
   - **ResNet-101** (2048-d)
   - **ZFNet** (4096-d)
   - **GoogLeNet** (1024-d)
3. Click **Search Similar Images**
4. View the top-10 most similar images with:
   - Similarity scores (cosine)
   - Class labels
   - Grid layout with hover effects

### Evaluation Metrics (`/metrics/`)

- **Comparison table:** Model | Top-10 Accuracy | mAP
- **t-SNE scatter plots** for each model (colour-coded by class)

---

## Models

| Model | Architecture | Embedding Dim | Source |
|-------|-------------|---------------|--------|
| ResNet-101 | 101-layer residual network | 2048 | `avgpool` output |
| ZFNet | AlexNet-style (ZF approximation) | 4096 | FC-7 output |
| GoogLeNet | Inception v1 | 1024 | Global avg pool |

All models use **pretrained ImageNet weights**, have their final classification layers removed, and are run in **eval mode with frozen parameters**.

---

## Tech Stack

- **PyTorch** + **torchvision** — Model loading & feature extraction
- **Django 4.2+** — Web framework
- **Scikit-learn** — NearestNeighbors, t-SNE, cosine similarity
- **Matplotlib** — t-SNE plot generation
- **NumPy** — Embedding storage & manipulation
- **Pillow** — Image I/O

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
