"""
Django settings for feature_understanding_project.

Production-level configuration for the visual embedding comparison system.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-change-me-in-production-x9$k2m!q@f7&w3"

DEBUG = True

ALLOWED_HOSTS = ["*"]

# ──────────────────────────────────────────────
# Applications
# ──────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "retrieval",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "feature_understanding_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "feature_understanding_project.wsgi.application"

# ──────────────────────────────────────────────
# Database (SQLite for simplicity)
# ──────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ──────────────────────────────────────────────
# Static & Media files
# ──────────────────────────────────────────────
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    BASE_DIR / "retrieval" / "static",
]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ──────────────────────────────────────────────
# File upload settings
# ──────────────────────────────────────────────
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# ──────────────────────────────────────────────
# Project-specific paths
# ──────────────────────────────────────────────
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
TSNE_PLOTS_DIR = BASE_DIR / "retrieval" / "static" / "tsne_plots"
UPLOADS_DIR = BASE_DIR / "retrieval" / "static" / "uploads"
PLOTS_DIR = BASE_DIR / "retrieval" / "static" / "plots"

# Ensure directories exist
for _dir in [EMBEDDINGS_DIR, TSNE_PLOTS_DIR, UPLOADS_DIR, MEDIA_ROOT, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Selected classes (auto-detected from embeddings or fallback)
# ──────────────────────────────────────────────
_CLASS_NAMES_FILE = EMBEDDINGS_DIR / "class_names.json"
if _CLASS_NAMES_FILE.exists():
    import json as _json
    with open(_CLASS_NAMES_FILE, "r") as _f:
        SELECTED_CLASSES = _json.load(_f)
else:
    SELECTED_CLASSES = [
        "accordion",
        "airplane",
        "camera",
        "elephant",
        "laptop",
        "motorbike",
        "watch",
        "wheelchair",
    ]

# ──────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────
MODEL_CONFIGS = {
    "resnet": {
        "name": "ResNet-101",
        "embedding_dim": 2048,
        "embeddings_file": "resnet_embeddings.npy",
    },
    "zfnet": {
        "name": "ZFNet",
        "embedding_dim": 4096,
        "embeddings_file": "zfnet_embeddings.npy",
    },
    "googlenet": {
        "name": "GoogLeNet",
        "embedding_dim": 1024,
        "embeddings_file": "googlenet_embeddings.npy",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────
LOGIN_URL = "user_login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "landing"
