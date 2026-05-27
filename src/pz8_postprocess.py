"""ПЗ 8. Постобработка результатов видеоанализа.

Этап убирает повторяющиеся текстовые фрагменты и объединяет близкие YOLO-детекции
в простые треки. Результат делает выход пайплайна компактнее и удобнее для
итогового отчета.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from common import OUTPUT_DIR, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz8"
OUT.mkdir(parents=True, exist_ok=True)


def dedup_segments(segments: list[dict], threshold: int = 85, key: str = "text") -> list[dict]:
    """Удалить подряд идущие текстовые сегменты с высокой похожестью."""
    out: list[dict] = []
    for s in segments:
        text = s.get(key, "")
        if not text:
            continue
        if out and fuzz.token_set_ratio(out[-1][key], text) >= threshold:
            out[-1]["end"] = s.get("end", out[-1].get("end"))
            if len(text) > len(out[-1][key]):
                out[-1][key] = text
        else:
            out.append(dict(s))
    return out


def iou(a: list[float], b: list[float]) -> float:
    """Посчитать IoU для двух bounding box в формате xyxy."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_detections(detections: list[dict], iou_thr: float = 0.4,
                     gap: int = 5) -> list[dict]:
    """Объединить близкие YOLO-детекции в треки.

    Группирует подряд идущие детекции одного класса с пересекающимся bbox.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for d in detections:
        by_class[d["label"]].append(d)
    tracks: list[dict] = []
    for label, items in by_class.items():
        items.sort(key=lambda x: x["frame"])
        active: list[dict] = []
        for d in items:
            matched = None
            for tr in active:
                if d["frame"] - tr["end_frame"] <= gap and iou(tr["bbox"], d["xyxy"]) >= iou_thr:
                    matched = tr
                    break
            if matched is None:
                active.append({
                    "label": label,
                    "start_frame": d["frame"],
                    "end_frame": d["frame"],
                    "bbox": d["xyxy"],
                    "max_conf": d["conf"],
                    "n_hits": 1,
                })
            else:
                matched["end_frame"] = d["frame"]
                matched["bbox"] = d["xyxy"]
                matched["max_conf"] = max(matched["max_conf"], d["conf"])
                matched["n_hits"] += 1
            # Старые активные треки больше не участвуют в сопоставлении.
            active = [t for t in active if d["frame"] - t["end_frame"] <= gap]
        tracks.extend(active)
    tracks.sort(key=lambda t: t["start_frame"])
    return tracks


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("--subs", type=Path, help="subtitles.json (от ПЗ3) или transcript.json (ПЗ4)")
    p.add_argument("--detections", type=Path, help="detections.jsonl (от ПЗ5)")
    p.add_argument("--out-name", required=True)
    p.add_argument("--sim", type=int, default=85, help="порог дедупа (0..100)")
    p.add_argument("--iou", type=float, default=0.4)
    p.add_argument("--gap", type=int, default=5)
    return p.parse_args()


def main() -> None:
    """Запустить дедупликацию текста и склейку детекций для ПЗ 8."""
    setup_logging()
    args = parse_args()
    out_dir = OUT / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subs:
        raw = json.loads(args.subs.read_text(encoding="utf-8"))
        # `subtitles.json` хранит список, `transcript.json` - словарь с `segments`.
        segments = raw.get("segments", raw) if isinstance(raw, dict) else raw
        deduped = dedup_segments(segments, threshold=args.sim,
                                  key="text")
        (out_dir / "subs_dedup.json").write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("дедуп: %d → %d", len(segments), len(deduped))

    if args.detections:
        dets = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line.strip()]
        tracks = merge_detections(dets, iou_thr=args.iou, gap=args.gap)
        (out_dir / "tracks.json").write_text(
            json.dumps(tracks, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("детекции: %d → треки: %d", len(dets), len(tracks))


if __name__ == "__main__":
    main()
