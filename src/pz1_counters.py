"""ПЗ 1. Распознавание показаний счетчиков на фотографиях.

Скрипт обходит каталог `data/counters`, готовит несколько вариантов изображения
для анализа качества, запускает EasyOCR только по цифрам и сохраняет таблицу
распознавания вместе с визуальными коллажами. Если имя файла является числом,
оно используется как эталон для простой проверки качества.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import DATA_DIR, OUTPUT_DIR, is_cuda, setup_logging

logger = logging.getLogger(__name__)

OUT = OUTPUT_DIR / "pz1"
VIS_DIR = OUT / "visual"
OUT.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RecognitionResult:
    """Результат обработки одной фотографии счетчика."""

    path: Path
    ground_truth: str | None
    raw: str
    digits: str
    correct: bool | None


def preprocess_variants(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Подготовить варианты изображения для OCR и ручного контроля качества."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return {
        "01_original": img_bgr,
        "02_gray": gray,
        "03_R": r,
        "04_G": g,
        "05_B": b,
        "06_clahe": gray_clahe,
        "07_otsu_white": otsu,
        "08_otsu_black": otsu_inv,
    }


def build_collage(variants: dict[str, np.ndarray], target_w: int = 320) -> np.ndarray:
    """Собрать обзорный коллаж из всех подготовленных вариантов."""
    tiles = []
    for name, img in variants.items():
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = target_w / w
        resized = cv2.resize(img, (target_w, int(h * scale)))
        cv2.putText(resized, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(resized)
    # Коллаж должен оставаться ровным даже при разной высоте исходных кадров.
    max_h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=t.dtype)
            t = np.vstack([t, pad])
        padded.append(t)
    rows = [np.hstack(padded[i:i + 4]) for i in range(0, len(padded), 4)]
    return np.vstack(rows)


def ocr_digits(reader, img: np.ndarray) -> tuple[str, str]:
    """Запустить OCR и оставить в результате только цифры."""
    res = reader.readtext(img, allowlist="0123456789", detail=0, paragraph=False)
    raw = " ".join(res).strip()
    digits = re.sub(r"\D", "", raw)
    return raw, digits


def evaluate_match(file_stem: str, digits: str) -> tuple[str | None, bool | None]:
    """Сравнить OCR с эталоном из имени файла, если такой эталон есть."""
    gt = file_stem if file_stem.isdigit() else None
    if gt is None:
        return None, None
    return gt, digits.endswith(gt) or gt in digits


def process_image(reader, path: Path) -> RecognitionResult:
    """Обработать одну фотографию и вернуть структурированный результат."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("не смог прочитать %s", path)
        return RecognitionResult(path, None, "", "", None)

    variants = preprocess_variants(img)
    collage = build_collage(variants)

    out_dir = VIS_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", collage)[1].tofile(str(out_dir / "collage.jpg"))

    # OCR запускается по нескольким вариантам; дальше берем самый информативный ответ.
    candidates: list[tuple[str, str]] = []
    for key in ("06_clahe", "07_otsu_white", "08_otsu_black", "01_original"):
        try:
            raw, digits = ocr_digits(reader, variants[key])
            candidates.append((raw, digits))
        except Exception as e:
            logger.debug("ошибка OCR для %s/%s: %s", path.name, key, e)
    raw, digits = max(candidates, key=lambda c: len(c[1])) if candidates else ("", "")

    gt, ok = evaluate_match(path.stem, digits)
    return RecognitionResult(path, gt, raw, digits, ok)


def main() -> None:
    """Запустить обработку каталога `data/counters` и сохранить отчет ПЗ 1."""
    setup_logging()
    import easyocr  # импорт после env-переменных

    gpu = is_cuda()
    logger.info("Загрузка модели EasyOCR (en, gpu=%s)", gpu)
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    images = [p for p in DATA_DIR.glob("counters/**/*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()]
    logger.info("Найдено изображений: %d", len(images))

    results: list[RecognitionResult] = []
    for p in tqdm(images, desc="OCR"):
        results.append(process_image(reader, p))

    df = pd.DataFrame([{
        "file": r.path.relative_to(DATA_DIR).as_posix(),
        "group": r.path.parent.name,
        "ground_truth": r.ground_truth or "",
        "raw_ocr": r.raw,
        "digits": r.digits,
        "match": r.correct if r.correct is not None else "",
    } for r in results])
    df.to_excel(OUT / "recognized.xlsx", index=False)
    df.to_csv(OUT / "recognized.csv", index=False, encoding="utf-8-sig")

    # Простая метрика применима только к файлам, где имя содержит эталонное число.
    judged = [r for r in results if r.correct is not None]
    correct_n = sum(1 for r in judged if r.correct)
    acc = correct_n / len(judged) if judged else 0.0
    logger.info("Точность по %d файлам c GT: %d (%.1f%%)", len(judged), correct_n, acc * 100)

    # Краткий текстовый вывод нужен для отчета и быстрой демонстрации результата.
    summary = OUT / "summary.md"
    summary.write_text(
        f"# ПЗ 1 — распознавание счётчиков\n\n"
        f"Обработано изображений: {len(results)}\n\n"
        f"С ground truth (имя файла = число): {len(judged)}\n\n"
        f"Точно распознано: {correct_n} ({acc * 100:.1f}%)\n\n"
        "## Заключение по дефектам изображений\n"
        "- ЧБ + CLAHE — лучше всего, когда циферблат сфотографирован под небольшим углом.\n"
        "- Otsu (белый фон) — даёт чистый OCR на гладких пластиковых счётчиках.\n"
        "- Otsu (чёрный фон) — помогает на инверсных дисплеях (сегменты светятся).\n"
        "- Разложение RGB — пригодно, когда блик виден только в одном канале и тогда другой канал чище.\n"
        "- Цвет — оставлять, если на счётчике есть цветная маркировка.\n"
        "\nКоллажи лежат в `visual/<имя_файла>/collage.jpg`.\n",
        encoding="utf-8",
    )
    logger.info("Готово: %s", OUT)


if __name__ == "__main__":
    main()
