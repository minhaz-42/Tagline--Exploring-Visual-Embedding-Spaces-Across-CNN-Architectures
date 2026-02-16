from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = "Download Caltech-101 and copy the selected 8 class folders into the project's dataset/ directory."

    def add_arguments(self, parser):
        parser.add_argument(
            "--target-dir",
            dest="target_dir",
            default=str(settings.DATASET_DIR),
            help="Where to copy the selected class folders (default: project's DATASET_DIR)",
        )
        parser.add_argument(
            "--max-per-class",
            dest="max_per_class",
            type=int,
            default=300,
            help="Maximum images to copy per class (default: 300)",
        )
        parser.add_argument(
            "--download-root",
            dest="download_root",
            default=None,
            help="Temporary root where Caltech101 will be downloaded. If omitted a temporary folder is used.",
        )

    def handle(self, *args, **options):
        try:
            # Import here so the command can fail gracefully if torchvision is absent
            from torchvision.datasets import Caltech101
        except Exception as exc:
            raise CommandError(
                "`torchvision` is required to download Caltech-101. "
                "Activate the virtualenv that has torchvision installed and re-run this command."
            ) from exc

        target_dir = Path(options["target_dir"]).resolve()
        max_per_class = int(options["max_per_class"])
        download_root = (
            Path(options["download_root"]) if options["download_root"] else target_dir.parent / "_caltech_download"
        )
        download_root.mkdir(parents=True, exist_ok=True)

        self.stdout.write(self.style.MIGRATE_HEADING("Caltech-101 downloader"))
        self.stdout.write(f"Downloading into: {download_root}")

        # Download dataset using torchvision helper
        Caltech101(root=str(download_root), download=True)

        extracted = download_root / "101_ObjectCategories"
        if not extracted.exists():
            raise CommandError(f"Expected extracted folder not found: {extracted}")

        selected = settings.SELECTED_CLASSES
        available = [p.name for p in extracted.iterdir() if p.is_dir()]

        # Smart matching: exact name, plural, or substring match (case-insensitive)
        def find_folder(cls: str) -> Path | None:
            candidates = {c.lower(): c for c in available}
            if cls in available:
                return extracted / cls
            if cls.lower() in candidates:
                return extracted / candidates[cls.lower()]
            # try plural
            if (cls + "s") in available:
                return extracted / (cls + "s")
            if (cls + "es") in available:
                return extracted / (cls + "es")
            # substring match
            for c in available:
                if cls.lower() in c.lower():
                    return extracted / c
            return None

        target_dir.mkdir(parents=True, exist_ok=True)

        missing: List[str] = []
        for cls in selected:
            src = find_folder(cls)
            if src is None:
                missing.append(cls)
                continue

            dst = target_dir / cls
            dst.mkdir(parents=True, exist_ok=True)

            # copy up to max_per_class images
            imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
            if not imgs:
                self.stdout.write(self.style.WARNING(f"No images found for '{cls}' in source folder {src}"))
                continue

            ncopy = min(len(imgs), max_per_class)
            for i, p in enumerate(imgs[:ncopy]):
                shutil.copy2(p, dst / p.name)
            self.stdout.write(self.style.SUCCESS(f"Copied {ncopy} images for class '{cls}'"))

        if missing:
            self.stdout.write(self.style.WARNING(f"These selected classes were NOT found in the Caltech archive: {', '.join(missing)}"))

        self.stdout.write(self.style.SUCCESS(f"Caltech-101 selected classes are ready in: {target_dir}"))
        self.stdout.write("")
        self.stdout.write(self.style.NOTICE("Next: run the offline pipeline:\n    python run_pipeline.py --dataset_dir ./dataset"))
