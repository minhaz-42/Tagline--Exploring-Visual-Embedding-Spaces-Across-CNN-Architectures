#!/usr/bin/env python
"""Top-level manage.py for convenience — allows running Django from the
repository root by adding the inner project folder to PYTHONPATH.
"""
import os
import sys
from pathlib import Path

# Ensure the inner Django project package is importable when running from repo root
BASE_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = BASE_DIR / "feature_understanding_project"
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "feature_understanding_project.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
