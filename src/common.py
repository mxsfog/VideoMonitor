"""Общие настройки путей, кэшей и логирования для всех этапов проекта."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

# Модели и кэши держим внутри проекта, чтобы не засорять системный диск.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(MODELS_DIR / "easyocr"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR / "ultralytics"))

for d in (MODELS_DIR / "hf", MODELS_DIR / "torch", MODELS_DIR / "easyocr",
          MODELS_DIR / "ultralytics", MODELS_DIR / "whisper"):
    d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Вернуть `cuda`, если доступен PyTorch с GPU, иначе `cpu`."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_cuda() -> bool:
    """Проверить, доступна ли CUDA для текущего окружения."""
    return get_device() == "cuda"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Настроить единый UTF-8 logger для CLI-скриптов проекта."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(Path(sys.argv[0]).stem if sys.argv[0] else "main")
