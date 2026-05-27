"""ПЗ 3. Распознавание титров и текста на кадрах видео.

На вход подается каталог кадров из ПЗ 2. Скрипт распознает текст в нижней части
кадров, объединяет повторяющиеся подряд титры и сохраняет результат в JSON,
JSONL и SRT-представлении.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
from rapidfuzz import fuzz
from tqdm import tqdm

from common import OUTPUT_DIR, is_cuda, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz3"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class FrameOCR:
    """Текст, найденный на одном кадре видео."""

    frame_index: int
    timestamp_s: float
    text: str


def ocr_frames(reader, frames_dir: Path, src_fps: float, langs: tuple[str, ...]) -> list[FrameOCR]:
    """Распознать текст на всех кадрах указанного каталога."""
    files = sorted(frames_dir.glob("frame_*.jpg"))
    out: list[FrameOCR] = []
    for f in tqdm(files, desc=f"OCR {frames_dir.name}"):
        img = cv2.imdecode(__import__("numpy").fromfile(str(f), dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # В большинстве учебных роликов титры находятся в нижней части кадра.
        h = img.shape[0]
        roi = img[int(h * 0.6):, :]
        try:
            text = " ".join(reader.readtext(roi, detail=0, paragraph=True)).strip()
        except Exception as e:
            logger.debug("ocr fail %s: %s", f.name, e)
            text = ""
        idx = int(f.stem.split("_")[-1])
        out.append(FrameOCR(idx, idx / src_fps, text))
    return out


def deduplicate(items: list[FrameOCR], threshold: int = 85) -> list[dict]:
    """Объединить подряд идущие похожие титры в временные группы."""
    groups: list[dict] = []
    for it in items:
        if not it.text:
            continue
        if groups and fuzz.token_set_ratio(groups[-1]["text"], it.text) >= threshold:
            groups[-1]["end_s"] = it.timestamp_s
            # Более длинная версия обычно содержит меньше потерь OCR.
            if len(it.text) > len(groups[-1]["text"]):
                groups[-1]["text"] = it.text
        else:
            groups.append({"start_s": it.timestamp_s, "end_s": it.timestamp_s, "text": it.text})
    return groups


def to_srt(groups: list[dict]) -> str:
    """Преобразовать сгруппированные титры в формат SRT."""
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for i, g in enumerate(groups, 1):
        lines.append(str(i))
        lines.append(f"{fmt(g['start_s'])} --> {fmt(max(g['end_s'], g['start_s'] + 0.5))}")
        lines.append(g["text"])
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True, help="каталог с кадрами (output/pz2/<video>/)")
    p.add_argument("--src-fps", type=float, default=2.0, help="fps исходного потока кадров (по которому шло сохранение)")
    p.add_argument("--langs", default="ru,en", help="языки OCR (через запятую)")
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def main() -> None:
    """Запустить OCR по кадрам и сохранить результаты ПЗ 3."""
    setup_logging()
    args = parse_args()
    import easyocr
    langs = tuple(s.strip() for s in args.langs.split(","))
    reader = easyocr.Reader(list(langs), gpu=is_cuda(), verbose=False)

    items = ocr_frames(reader, args.frames, args.src_fps, langs)
    groups = deduplicate(items)

    name = args.frames.name
    out_dir: Path = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(it.__dict__, ensure_ascii=False) for it in items),
        encoding="utf-8",
    )
    (out_dir / "subtitles.srt").write_text(to_srt(groups), encoding="utf-8")
    (out_dir / "subtitles.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("кадров: %d, уникальных титров: %d → %s", len(items), len(groups), out_dir)


if __name__ == "__main__":
    main()
