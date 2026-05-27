"""ПЗ 7, вариант 3. Распознавание объектов на кадрах через VLM.

В этом варианте LLM работает как vision-language model: каждый выбранный кадр
отправляется в OpenRouter, а модель возвращает структурированный список
объектов и признак деструктивности. Этап закрывает требование распознавания
объектов с использованием LLM.

Использование:
    python pz7_vlm_gemini.py output/pz2/<video>/ --out-name <video> --every-n 30
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

from common import OUTPUT_DIR, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz7"
OUT.mkdir(parents=True, exist_ok=True)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRIES = 2


SYSTEM_PROMPT = """Ты компьютерное зрение + модератор. На вход — кадр видео.
Перечисли все заметные объекты на изображении.

Для каждого укажи:
- object: краткое название (на русском)
- count: сколько таких объектов в кадре
- confidence: уверенность 0..1
- is_destructive: true, если объект относится к деструктивным категориям
  (огнестрельное/холодное оружие, наркотики, кровь, насилие, токсичные жесты,
  алкоголь несовершеннолетним, экстремистская символика, материалы 18+)
- subclass: если is_destructive=true, один код из списка:
  ALCOHOL, SMOKING, DRUGS, DRUGS2KIDS, VANDALISM, VIOLENCE, SUICIDE,
  KIDSSUICIDE, OBSCENE_LANGUAGE, TERROR, EXTREMISM, TERRORCONTENT,
  NUDE, SEX, KIDSPORN, LGBT, CHILDFREE, INOAGENT, INOAGENTCONTENT,
  ANTIWAR, LUDOMANIA.
  Если is_destructive=false, верни null.
- reason: краткое пояснение почему destructive (если is_destructive=true)

Ответ строго в формате JSON-массива, БЕЗ markdown и преамбулы:
[{"object":"...","count":N,"confidence":0..1,"is_destructive":bool,"subclass":"VIOLENCE","reason":"..."}]

Если объектов нет, верни пустой массив [].
"""


def encode_image(path: Path, max_side: int = 768, jpeg_q: int = 75) -> str:
    """Сжать кадр и представить его как base64 data URL."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"не открыть {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    if not ok:
        raise RuntimeError("не удалось закодировать кадр в JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_gemini(
    api_key: str,
    model: str,
    image_data_url: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
    retries: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Отправить в OpenRouter один VLM-запрос с текстом и изображением."""
    if retries is None:
        retries = int(os.environ.get("OPENROUTER_RETRIES", str(DEFAULT_RETRIES)))
    if timeout_seconds is None:
        timeout_seconds = float(
            os.environ.get("OPENROUTER_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "analyz-dannyx-coursework",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Перечисли объекты на кадре в JSON-формате."},
                {"type": "image_url",
                 "image_url": {"url": image_data_url}},
            ]},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if r.status_code in {429, 500, 502, 503, 504}:
                wait = 2 ** attempt
                logger.warning("HTTP %d, повтор через %d сек", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return extract_message_text(data["choices"][0]["message"])
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM API не отвечает: {last_err}")


def extract_message_text(message: dict) -> str:
    """Нормализовать содержимое ответа OpenRouter до plain text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    return "[]"


def parse_json_array(text: object) -> list[dict]:
    """Извлечь JSON-массив из ответа модели, если формат ответа корректен."""
    if not isinstance(text, str):
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def list_frames(frames_dir: Path, every_n: int) -> list[Path]:
    """Вернуть отсортированный список кадров с заданным прореживанием."""
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    return all_frames[::max(1, every_n)]


def classify_frames(frames: list[Path], api_key: str, model: str) -> list[dict]:
    """Классифицировать выбранные кадры через VLM."""
    out: list[dict] = []
    for frame_path in tqdm(frames, desc=f"VLM {model}"):
        try:
            url = encode_image(frame_path)
            resp = call_gemini(api_key, model, url)
            objects = parse_json_array(resp)
        except Exception as e:
            logger.warning("ошибка VLM на кадре %s: %s", frame_path.name, e)
            objects = []
        out.append({
            "frame": frame_path.name,
            "objects": objects,
            "destructive_count": sum(1 for o in objects if o.get("is_destructive")),
        })
    return out


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("frames", type=Path, help="каталог с кадрами (output/pz2/<v>/)")
    p.add_argument("--out-name", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--every-n", type=int, default=30,
                   help="прореживание: брать каждый N-й кадр")
    p.add_argument("--api-key", default=None)
    return p.parse_args()


def main() -> None:
    """Запустить VLM-распознавание кадров и сохранить результаты ПЗ 7."""
    setup_logging()
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Нет ключа: --api-key или OPENROUTER_API_KEY env")

    frames = list_frames(args.frames, args.every_n)
    if not frames:
        raise SystemExit(f"не нашёл frame_*.jpg в {args.frames}")
    logger.info("кадров после прореживания (every_n=%d): %d, модель: %s",
                args.every_n, len(frames), args.model)

    results = classify_frames(frames, api_key, args.model)

    name = args.out_name or args.frames.name
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classified.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )

    # Агрегация превращает покадровые ответы в удобную сводку для отчета.
    obj_counter: Counter[str] = Counter()
    destructive_counter: Counter[str] = Counter()
    destructive_by_subclass: Counter[str] = Counter()
    total_destructive_frames = 0
    for r in results:
        if r["destructive_count"] > 0:
            total_destructive_frames += 1
        for o in r["objects"]:
            label = o.get("object", "?")
            obj_counter[label] += int(o.get("count", 1))
            if o.get("is_destructive"):
                count = int(o.get("count", 1))
                destructive_counter[label] += count
                subclass = str(o.get("subclass") or "").upper()
                if subclass:
                    destructive_by_subclass[subclass] += count

    summary = {
        "model": args.model,
        "frames_processed": len(results),
        "frames_with_destructive": total_destructive_frames,
        "total_objects": sum(obj_counter.values()),
        "by_label": dict(obj_counter.most_common()),
        "destructive_by_label": dict(destructive_counter.most_common()),
        "destructive_by_subclass": dict(destructive_by_subclass.most_common()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("кадров с деструктивом: %d из %d",
                total_destructive_frames, len(results))
    logger.info("топ объектов: %s",
                ", ".join(f"{k}={v}" for k, v in obj_counter.most_common(8)))
    if destructive_counter:
        logger.info("деструктивные: %s",
                    ", ".join(f"{k}={v}" for k, v in destructive_counter.most_common()))


if __name__ == "__main__":
    main()
