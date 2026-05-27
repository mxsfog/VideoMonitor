"""ПЗ 4. Извлечение речи из видео через faster-whisper.

Скрипт извлекает моно-аудио 16 kHz через `ffmpeg`, запускает распознавание речи
и сохраняет транскрипт в JSON, SRT и TXT. Модели кэшируются в `models/whisper`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from common import MODELS_DIR, OUTPUT_DIR, is_cuda, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz4"
OUT.mkdir(parents=True, exist_ok=True)


def extract_audio(video: Path, out_wav: Path) -> Path:
    """Извлечь из видео моно-аудио WAV 16 kHz."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-ac", "1", "-ar", "16000", "-vn", str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def transcribe(wav: Path, model_size: str, language: str | None) -> dict:
    """Распознать речь и вернуть сегменты с временными интервалами."""
    from faster_whisper import WhisperModel
    cache = MODELS_DIR / "whisper"
    device = "cuda" if is_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Загрузка модели faster-whisper %s (%s, %s)",
                model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type,
                        download_root=str(cache))
    segments, info = model.transcribe(str(wav), language=language, vad_filter=True,
                                       beam_size=1)
    out_segments = []
    full_text = []
    for seg in segments:
        out_segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        full_text.append(seg.text.strip())
    return {
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments,
        "full_text": " ".join(full_text),
    }


def to_srt(segments: list[dict]) -> str:
    """Преобразовать сегменты распознавания в SRT."""
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    out = []
    for i, s in enumerate(segments, 1):
        out += [str(i), f"{fmt(s['start'])} --> {fmt(s['end'])}", s["text"], ""]
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--lang", default="ru", help="язык распознавания (ru/en/auto)")
    return p.parse_args()


def main() -> None:
    """Извлечь аудио, распознать речь и сохранить результаты ПЗ 4."""
    setup_logging()
    args = parse_args()
    out_dir = OUT / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = extract_audio(args.video, out_dir / "audio.wav")
    lang = None if args.lang == "auto" else args.lang
    result = transcribe(wav, args.model, lang)
    (out_dir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "transcript.srt").write_text(to_srt(result["segments"]), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(result["full_text"], encoding="utf-8")
    logger.info("сегментов: %d, lang=%s → %s", len(result["segments"]), result["language"], out_dir)


if __name__ == "__main__":
    main()
