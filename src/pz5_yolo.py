"""ПЗ 5. Детекция объектов на видео с помощью YOLOv8.

Используются предварительно обученные веса Ultralytics. Скрипт сохраняет
покадровые детекции в JSONL и краткую сводку по найденным классам.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

from common import MODELS_DIR, OUTPUT_DIR, is_cuda, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz5"
OUT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--weights", default="yolov8n.pt", help="веса (yolov8n.pt по умолчанию)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--every-n", type=int, default=1, help="обрабатывать каждый N-й кадр")
    return p.parse_args()


def main() -> None:
    """Запустить YOLOv8 по видео и сохранить детекции ПЗ 5."""
    setup_logging()
    args = parse_args()
    from ultralytics import YOLO

    weights_dir = MODELS_DIR / "ultralytics"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / args.weights
    model = YOLO(str(weights_path) if weights_path.exists() else args.weights)
    if not weights_path.exists():
        # Ultralytics может скачать веса в CWD; переносим их в проектный cache.
        cwd_pt = Path(args.weights)
        if cwd_pt.exists():
            shutil.move(str(cwd_pt), str(weights_path))

    out_dir = OUT / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("YOLO inference (cuda): %s", args.video)

    device = 0 if is_cuda() else "cpu"
    results = model.predict(
        source=str(args.video),
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=str(OUT),
        name=args.video.stem + "_pred",
        verbose=False,
        stream=True,
        vid_stride=args.every_n,
    )

    detections = []
    class_counter: Counter[str] = Counter()
    for i, r in enumerate(results):
        names = r.names
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            label = names[cls]
            class_counter[label] += 1
            detections.append({
                "frame": i,
                "label": label,
                "conf": round(conf, 3),
                "xyxy": [round(v, 1) for v in xyxy],
            })

    (out_dir / "detections.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in detections),
        encoding="utf-8",
    )
    summary = {
        "total_detections": len(detections),
        "by_class": dict(class_counter.most_common()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("детекций: %d, классов: %d", len(detections), len(class_counter))
    logger.info("топ классов: %s", ", ".join(f"{k}={v}" for k, v in class_counter.most_common(10)))


if __name__ == "__main__":
    main()
