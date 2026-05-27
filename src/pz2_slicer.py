"""ПЗ 2. Нарезка видео на кадры с заданной частотой.

Скрипт принимает локальный файл или ссылку, при необходимости скачивает видео
через загрузчик видео и сохраняет кадры в `output/pz2/<video>/`. Частота
выбирается параметром `--fps`, качество скачивания - `--quality`.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import cv2

from common import DATA_DIR, OUTPUT_DIR, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = OUTPUT_DIR / "pz2"


def is_url(s: str) -> bool:
    """Проверить, является ли входная строка HTTP/HTTPS-ссылкой."""
    return bool(re.match(r"^https?://", s))


def download_video(url: str, out_dir: Path, quality: str = "best") -> Path:
    """Скачать видео в рабочий каталог и вернуть путь к локальному файлу."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = quality if quality != "best" else "bv*+ba/b"
    if quality.endswith("p") and quality[:-1].isdigit():
        h = quality[:-1]
        fmt = f"bv*[height<={h}]+ba/b[height<={h}]"
    out_template = str(out_dir / "%(title).80s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt, "--merge-output-format", "mp4",
        "-o", out_template, "--no-progress", url,
    ]
    logger.info("загрузка видео: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return files[-1]


def slice_video(path: Path, fps_target: float, out_dir: Path) -> int:
    """Сохранить кадры с целевой частотой и вернуть их количество."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается видео: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps_target))
    logger.info("исходный fps=%.2f, кадров=%d, шаг=%d → ~%.2f fps",
                src_fps, total, step, src_fps / step)

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = out_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        idx += 1
    cap.release()
    logger.info("сохранено кадров: %d → %s", saved, out_dir)
    return saved


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser(description="Нарезка видео на кадры")
    p.add_argument("source", help="ссылка или путь к локальному файлу/папке")
    p.add_argument("--fps", type=float, default=2.0, help="целевая частота кадров (default 2)")
    p.add_argument("--quality", default="best", help="качество скачивания: best/720p/480p")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="выходной каталог")
    return p.parse_args()


def main() -> None:
    """Запустить скачивание при необходимости и нарезать видео на кадры."""
    setup_logging()
    args = parse_args()
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    if is_url(args.source):
        dl_dir = DATA_DIR / "videos"
        video_path = download_video(args.source, dl_dir, args.quality)
    else:
        sp = Path(args.source)
        if sp.is_dir():
            videos = [p for p in sp.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}]
            for v in videos:
                slice_video(v, args.fps, out_root / v.stem)
            return
        video_path = sp

    out_dir = out_root / video_path.stem
    slice_video(video_path, args.fps, out_dir)


if __name__ == "__main__":
    main()
