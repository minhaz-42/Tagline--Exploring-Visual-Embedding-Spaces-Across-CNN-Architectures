"""
feature_extractor.py
--------------------
Builds headless (no-classifier) versions of ResNet-101, ZFNet, and GoogLeNet,
and provides a unified interface for extracting feature embeddings.

Architecture notes
~~~~~~~~~~~~~~~~~~
* **ResNet-101** – 2 048-dim vector from the global average-pool layer.
* **ZFNet**      – Approximated by AlexNet with ZFNet-style usage;
                   4 096-dim vector from classifier[5] (FC-7 equivalent).
* **GoogLeNet**  – 1 024-dim vector from the global average-pool layer.

All models use pretrained ImageNet weights, remove the final classification
head, freeze all parameters, and run in eval mode.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_loader import get_transform

# ──────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────

def _get_device() -> torch.device:
    """Return CUDA device if available, MPS if on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════

def _build_resnet101() -> nn.Module:
    """
    ResNet-101 with the final FC layer removed.
    Output: 2 048-dim from avgpool.
    """
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)
    # Remove final FC — use Identity so forward() returns the pooled features
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _build_zfnet() -> nn.Module:
    """
    ZFNet approximation using AlexNet architecture with ImageNet weights.
    Extract 4 096-dim embedding from classifier[5] (FC-7).

    We register a forward hook to capture the intermediate activation.
    """
    weights = models.AlexNet_Weights.IMAGENET1K_V1
    base = models.alexnet(weights=weights)
    base.eval()
    for param in base.parameters():
        param.requires_grad = False

    class ZFNetWrapper(nn.Module):
        """Wraps AlexNet and extracts the FC-7 (4096-d) activation."""

        def __init__(self, alexnet: nn.Module):
            super().__init__()
            self.features = alexnet.features
            self.avgpool = alexnet.avgpool
            # classifier layers 0-5 gives us through FC7 (ReLU output)
            self.classifier_head = nn.Sequential(*list(alexnet.classifier.children())[:6])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier_head(x)
            return x

    wrapper = ZFNetWrapper(base)
    wrapper.eval()
    return wrapper


def _build_googlenet() -> nn.Module:
    """
    GoogLeNet (Inception v1) with the final FC removed.
    Output: 1 024-dim from the global average pool.

    Note: torchvision's pretrained weights expect `aux_logits=True` during
    model construction; the forward pass still returns the main output in
    eval mode, so we construct with `aux_logits=True` and then remove the
    final `fc` layer.
    """
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1
    # torchvision requires aux_logits to match the pretrained weight signature
    model = models.googlenet(weights=weights, aux_logits=True)
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

MODEL_BUILDERS: Dict[str, callable] = {
    "resnet": _build_resnet101,
    "zfnet": _build_zfnet,
    "googlenet": _build_googlenet,
}


def get_model(model_key: str) -> nn.Module:
    """
    Return the headless pretrained model for a given key.

    Parameters
    ----------
    model_key : str
        One of ``"resnet"``, ``"zfnet"``, ``"googlenet"``.

    Returns
    -------
    nn.Module
        The model moved to the best available device, in eval mode.
    """
    if model_key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Choose from {list(MODEL_BUILDERS.keys())}."
        )
    device = _get_device()
    model = MODEL_BUILDERS[model_key]()
    model = model.to(device)
    return model


# ══════════════════════════════════════════════
# Batch embedding extraction
# ══════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings for every sample in *dataloader* using *model*.

    Parameters
    ----------
    model : nn.Module
        A headless CNN (from ``get_model``).
    dataloader : DataLoader
        Must yield ``(images, labels, paths)`` triples.
    device : torch.device, optional
        Override device (defaults to model's current device).

    Returns
    -------
    embeddings : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    paths : list[str], length N
    """
    if device is None:
        device = next(model.parameters()).device

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    all_paths: List[str] = []

    model.eval()
    for images, labels, paths in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)  # (B, D)
        all_embeddings.append(features.cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)
    return embeddings, labels_arr, all_paths


# ══════════════════════════════════════════════
# Single-image embedding (for web inference)
# ══════════════════════════════════════════════

@torch.no_grad()
def extract_single_embedding(
    model: nn.Module,
    image_path: str,
) -> np.ndarray:
    """
    Extract the embedding vector for a single image.

    Parameters
    ----------
    model : nn.Module
        Headless CNN.
    image_path : str
        Path to the query image on disk.

    Returns
    -------
    np.ndarray, shape (D,)
    """
    from PIL import Image

    device = next(model.parameters()).device
    transform = get_transform()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    model.eval()
    embedding = model(tensor).cpu().numpy().squeeze(0)  # (D,)
    return embedding
