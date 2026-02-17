"""
Custom template tags for serving dataset images.

Usage in templates::

    {% load retrieval_tags %}
    <img src="{% dataset_image path %}" />
"""

import base64
from pathlib import Path

from django import template
from django.conf import settings
from django.utils.safestring import mark_safe

register = template.Library()


@register.simple_tag
def dataset_image(image_path: str) -> str:
    """
    Return a base64-encoded data URI for a dataset image so it can be
    displayed without needing to serve the dataset directory.

    Handles both absolute paths and relative paths (e.g. cifar100/class/idx.png).
    Falls back to an empty string if the file doesn't exist.
    """
    p = Path(image_path)

    # If path doesn't exist as-is, try resolving relative to DATASET_DIR
    if not p.exists():
        dataset_dir = getattr(settings, "DATASET_DIR", None)
        if dataset_dir:
            # For CIFAR-100 synthetic paths like 'cifar100/class_name/123.png'
            # strip the 'cifar100/' prefix and look in dataset dir
            rel = str(image_path)
            if rel.startswith("cifar100/"):
                rel = rel[len("cifar100/"):]
            candidate = Path(dataset_dir) / rel
            if candidate.exists():
                p = candidate

    if not p.exists():
        return ""

    suffix = p.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(suffix, "image/jpeg")

    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    return f"data:{mime};base64,{data}"


@register.filter
def class_name_from_path(image_path: str) -> str:
    """
    Extract the class name (parent folder) from a full image path.

    Example: /data/dataset/airplane/img001.jpg → airplane
    """
    return Path(image_path).parent.name
