"""ПЗ 6. Классификация кадров видео моделью ResNet50.

Этот этап дополняет объектную детекцию общей характеристикой сцены: интерьер,
техника, аудитория, сцена и другие классы ImageNet. Результат используется как
вспомогательный контекст для итогового анализа.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from tqdm import tqdm

from common import OUTPUT_DIR, is_cuda, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz6"
OUT.mkdir(parents=True, exist_ok=True)


def load_imagenet_classes() -> list[str]:
    """Получить список классов ImageNet из метаданных torchvision."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    return list(weights.meta["categories"])


def build_model() -> tuple[torch.nn.Module, transforms.Compose, list[str], str]:
    """Подготовить ResNet50 и устройство выполнения."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    device = "cuda" if is_cuda() else "cpu"
    model.eval().to(device)
    classes = list(weights.meta["categories"])
    preprocess = weights.transforms()
    return model, preprocess, classes, device


def classify_frames(frames_dir: Path, top_k: int = 3) -> list[dict]:
    """Классифицировать все кадры в каталоге и вернуть top-k прогнозов."""
    model, preprocess, classes, device = build_model()
    files = sorted(frames_dir.glob("frame_*.jpg"))
    out: list[dict] = []
    for f in tqdm(files, desc=frames_dir.name):
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits[0], dim=0)
            top_p, top_i = torch.topk(probs, top_k)
        out.append({
            "frame": f.name,
            "predictions": [
                {"class": classes[int(i)], "prob": round(float(p), 4)}
                for p, i in zip(top_p.tolist(), top_i.tolist(), strict=True)
            ],
        })
    return out


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("frames", type=Path, help="каталог с кадрами")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main() -> None:
    """Запустить классификацию кадров и сохранить результаты ПЗ 6."""
    setup_logging()
    args = parse_args()
    out_dir = OUT / args.frames.name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = classify_frames(args.frames, top_k=args.top_k)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds),
        encoding="utf-8",
    )
    # Сводка по top-1 нужна для быстрой оценки преобладающих сцен.
    from collections import Counter
    top1 = Counter(p["predictions"][0]["class"] for p in preds if p["predictions"])
    (out_dir / "summary.json").write_text(
        json.dumps({"frames": len(preds), "by_top1": dict(top1.most_common())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("кадров: %d, top1 уник.: %d", len(preds), len(top1))


if __name__ == "__main__":
    main()
