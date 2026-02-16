"""
views.py
--------
Django views for the visual embedding retrieval system.

Routes
~~~~~~
* ``/``              – **landing**   : project landing page
* ``/search/``       – **search_page** : upload form (select model + image)
* ``/search/go/``    – **search**    : perform retrieval and show results
* ``/metrics/``      – **metrics**   : evaluation table + t-SNE plots
* ``/dashboard/``    – **dashboard** : user dashboard (login required)
* ``/history/``      – **history**   : search history (login required)
* ``/login/``        – **user_login**
* ``/register/``     – **user_register**
* ``/logout/``       – **user_logout**
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage

from .forms import RegisterForm
from .models import SearchHistory

# ══════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════

MODELS_AVAILABLE = {
    "resnet": "ResNet-101 (2048-d)",
    "zfnet": "ZFNet (4096-d)",
    "googlenet": "GoogLeNet (1024-d)",
}


def _get_eval_rows():
    """Load evaluation results and return formatted rows for templates."""
    eval_path = settings.EMBEDDINGS_DIR / "evaluation_results.json"
    eval_data = {}
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

    rows = []
    model_configs = settings.MODEL_CONFIGS
    for key in ("resnet", "zfnet", "googlenet"):
        cfg = model_configs[key]
        metrics_item = eval_data.get(key, {})
        top_k = metrics_item.get("top_k_accuracy", "N/A")
        mAP = metrics_item.get("mAP", "N/A")

        rows.append({
            "model_key": key,
            "model_name": cfg["name"],
            "embedding_dim": cfg["embedding_dim"],
            "top_k_accuracy": top_k,
            "top_k_accuracy_fmt": f"{top_k:.4f}" if isinstance(top_k, (int, float)) else "N/A",
            "top_k_pct": f"{top_k * 100:.2f}" if isinstance(top_k, (int, float)) else "0",
            "mAP": mAP,
            "mAP_fmt": f"{mAP:.4f}" if isinstance(mAP, (int, float)) else "N/A",
            "mAP_pct": f"{mAP * 100:.2f}" if isinstance(mAP, (int, float)) else "0",
        })
    return rows


def _count_dataset_images():
    """Count total images in the dataset directory."""
    dataset_dir = settings.DATASET_DIR
    count = 0
    if dataset_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            count += len(list(dataset_dir.rglob(ext)))
    return count


# ══════════════════════════════════════════════
# Landing
# ══════════════════════════════════════════════

def landing(request):
    """Project landing page."""
    return render(request, "landing.html")


# ══════════════════════════════════════════════
# Search page (upload form)
# ══════════════════════════════════════════════

def search_page(request):
    """Render the query upload form."""
    return render(request, "index.html", {"models_available": MODELS_AVAILABLE})


# ══════════════════════════════════════════════
# Search / retrieval
# ══════════════════════════════════════════════

def search(request):
    """
    Handle POST from the upload form:
    1. Save the uploaded image.
    2. Load the chosen model and extract the query embedding.
    3. Load the pre-computed database embeddings.
    4. Build a nearest-neighbour index and retrieve the top-10 results.
    5. Save to history if user is authenticated.
    6. Render the results page.
    """
    if request.method != "POST":
        return redirect("search_page")

    # ── 1. Validate inputs ──
    uploaded_file = request.FILES.get("query_image")
    model_key = request.POST.get("model_key", "resnet")

    if not uploaded_file:
        return render(request, "index.html", {
            "error": "Please upload an image.",
            "models_available": MODELS_AVAILABLE,
        })

    # ── 2. Save uploaded image ──
    ext = Path(uploaded_file.name).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    save_dir = settings.UPLOADS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    with open(save_path, "wb+") as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)

    # ── 3. Ensure offline embeddings exist ──
    model_configs = settings.MODEL_CONFIGS
    embeddings_dir = settings.EMBEDDINGS_DIR

    model_cfg = model_configs.get(model_key)
    if model_cfg is None:
        _cleanup(save_path)
        return render(request, "index.html", {
            "error": f"Unknown model '{model_key}'. Please choose a valid model.",
            "models_available": MODELS_AVAILABLE,
        })

    required_files = [
        embeddings_dir / model_cfg["embeddings_file"],
        embeddings_dir / "labels.npy",
        embeddings_dir / "image_paths.json",
    ]
    missing_files = [str(p.name) for p in required_files if not p.exists()]
    if missing_files:
        _cleanup(save_path)
        return render(request, "index.html", {
            "error": (
                "Offline embeddings are missing. Run the offline pipeline first:\n"
                "    python run_pipeline.py --dataset_dir ./dataset\n\n"
                f"Missing files: {', '.join(missing_files)}"
            ),
            "models_available": MODELS_AVAILABLE,
        })

    # ── 4. Extract query embedding ──
    try:
        from .feature_pipeline.feature_extractor import get_model, extract_single_embedding
        from .feature_pipeline.nearest_neighbors import load_embeddings, build_index, query_index
    except Exception as exc:
        _cleanup(save_path)
        return render(request, "index.html", {
            "error": (
                "ML dependencies missing: torch/torchvision. "
                "Install with: pip install -r requirements.txt\n\n"
                f"{exc.__class__.__name__}: {exc}"
            ),
            "models_available": MODELS_AVAILABLE,
        })

    model = get_model(model_key)
    query_embedding = extract_single_embedding(model, str(save_path))

    # ── 5. Load database embeddings & build index ──
    db_embeddings, db_labels, db_paths = load_embeddings(str(embeddings_dir), model_key)
    nn_model = build_index(db_embeddings)

    # ── 6. Retrieve top-10 ──
    results = query_index(
        nn_model=nn_model,
        query_embedding=query_embedding,
        embeddings=db_embeddings,
        labels=db_labels,
        paths=db_paths,
        top_k=10,
    )

    # Map labels → class names
    class_names = settings.SELECTED_CLASSES
    for r in results:
        r["class_name"] = class_names[r["label"]]
        r["image_path"] = r["path"]

    # ── 7. Save to history ──
    history_kwargs = {
        "model_key": model_key,
        "query_image_filename": filename,
        "result_count": len(results),
    }
    if results:
        history_kwargs["top_result_class"] = results[0].get("class_name", "")
        history_kwargs["top_similarity"] = results[0].get("similarity", 0)
    if request.user.is_authenticated:
        history_kwargs["user"] = request.user
    SearchHistory.objects.create(**history_kwargs)

    # Model display name
    model_display = settings.MODEL_CONFIGS[model_key]["name"]
    query_image_url = f"uploads/{filename}"

    context = {
        "query_image_url": query_image_url,
        "model_key": model_key,
        "model_name": model_display,
        "results": results,
        "class_names": class_names,
    }
    return render(request, "results.html", context)


def _cleanup(path):
    """Remove uploaded file on error."""
    try:
        Path(path).unlink()
    except Exception:
        pass


# ══════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════

def metrics(request):
    """Display the evaluation metrics table and t-SNE plots."""
    rows = _get_eval_rows()
    total_images = _count_dataset_images()

    # Locate t-SNE plots
    tsne_plots = {}
    for key in ("resnet", "zfnet", "googlenet"):
        plot_file = settings.TSNE_PLOTS_DIR / f"tsne_{key}.png"
        if plot_file.exists():
            tsne_plots[key] = f"tsne_plots/tsne_{key}.png"

    context = {
        "rows": rows,
        "tsne_plots": tsne_plots,
        "model_configs": settings.MODEL_CONFIGS,
        "total_images": total_images,
    }
    return render(request, "metrics.html", context)


# ══════════════════════════════════════════════
# Dashboard (login required)
# ══════════════════════════════════════════════

@login_required(login_url="user_login")
def dashboard(request):
    """User dashboard with stats and recent activity."""
    total_searches = SearchHistory.objects.filter(user=request.user).count()
    total_images = _count_dataset_images()
    recent_searches = SearchHistory.objects.filter(user=request.user)[:5]
    model_stats = _get_eval_rows()

    # Attach display info
    search_list = []
    for s in recent_searches:
        search_list.append({
            "model_display": s.model_display,
            "query_image_url": s.query_image_url,
            "top_result_class": s.top_result_class,
            "result_count": s.result_count,
            "created_at": s.created_at,
        })

    context = {
        "total_searches": total_searches,
        "total_images": total_images,
        "recent_searches": search_list,
        "model_stats": model_stats,
    }
    return render(request, "dashboard.html", context)


# ══════════════════════════════════════════════
# History (login required)
# ══════════════════════════════════════════════

@login_required(login_url="user_login")
def history(request):
    """User search history."""
    all_searches = SearchHistory.objects.filter(user=request.user)[:50]
    search_list = []
    for s in all_searches:
        search_list.append({
            "model_display": s.model_display,
            "query_image_url": s.query_image_url,
            "top_result_class": s.top_result_class,
            "top_similarity": f"{s.top_similarity:.4f}" if s.top_similarity else None,
            "result_count": s.result_count,
            "created_at": s.created_at,
        })

    return render(request, "history.html", {"searches": search_list})


# ══════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════

def user_login(request):
    """Login view."""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            next_url = request.POST.get("next") or request.GET.get("next") or "dashboard"
            return redirect(next_url)
    else:
        form = AuthenticationForm()

    return render(request, "registration/login.html", {"form": form})


def user_register(request):
    """Registration view."""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Account created! Welcome, {user.username}.")
            return redirect("dashboard")
    else:
        form = RegisterForm()

    return render(request, "registration/register.html", {"form": form})


def user_logout(request):
    """Logout view."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect("landing")
