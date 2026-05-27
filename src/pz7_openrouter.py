"""ПЗ 7, вариант 2. Классификация текста через OpenRouter.

Этот вариант используется, когда локальной модели недостаточно или требуется
облачная LLM. Формат входа и выхода совпадает с локальным вариантом ПЗ 7:

    python pz7_openrouter.py <input.json|jsonl|srt|txt> --out-name <name>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import requests

from common import OUTPUT_DIR, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz7"
OUT.mkdir(parents=True, exist_ok=True)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "inclusionai/ling-2.6-1t:free"


SYSTEM_PROMPT = """Ты модератор контента. Тебе дают пронумерованный список фрагментов (титры/транскрипт).
Классифицируй каждый фрагмент по схеме:
- safe: нейтральный, безопасный
- toxic: оскорбления, агрессия, мат
- harassment: травля конкретного лица/группы
- extremism: пропаганда насилия, терроризма, ненависти
- adult: материалы 18+
- other: иное

Ответ — массив JSON, по одному объекту на каждый фрагмент в исходном порядке.
Каждый объект: {"id": <номер>, "label": "...", "confidence": 0..1, "reason": "...кратко..."}
Без преамбулы, без markdown. Только JSON-массив.
"""


def call_openrouter(api_key: str, model: str, prompt: str,
                    system: str = SYSTEM_PROMPT, temperature: float = 0.0,
                    max_tokens: int = 800, retries: int = 4) -> str:
    """Отправить запрос в OpenRouter с повторами при временных ошибках."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "analyz-dannyx-coursework",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        if r.status_code in {429, 500, 502, 503, 504}:
            wait = 2 ** attempt
            logger.warning("HTTP %d, повтор через %d сек", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenRouter не отвечает после {retries} попыток")


def parse_json_array(text: str) -> list[dict]:
    """Извлечь JSON-массив из ответа модели, включая ответы в markdown-блоке."""
    # Некоторые модели оборачивают JSON в ```json ... ```, это нужно снять.
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


def classify_texts(items: list[str], api_key: str, model: str,
                    batch: int = 10) -> list[dict]:
    """Классифицировать текстовые фрагменты батчами через OpenRouter."""
    out: list[dict] = []
    total_batches = (len(items) + batch - 1) // batch
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        prompt_lines = [f"{i + 1}. {t}" for i, t in enumerate(chunk)]
        prompt = "Список фрагментов:\n" + "\n".join(prompt_lines)
        try:
            resp = call_openrouter(api_key, model, prompt)
            arr = parse_json_array(resp)
        except Exception as e:
            logger.warning("ошибка LLM на батче %d: %s", start // batch, e)
            arr = []
        if len(arr) != len(chunk):
            logger.warning("батч %d: ожидалось %d ответов, получено %d",
                           start // batch, len(chunk), len(arr))
        for i, txt in enumerate(chunk):
            obj = next((a for a in arr if a.get("id") == i + 1), None)
            if obj is None and i < len(arr):
                obj = arr[i]
            if obj is None:
                obj = {"label": "other", "confidence": 0.0, "reason": "ответ модели отсутствует"}
            obj["text"] = txt
            out.append(obj)
        logger.info("батч %d/%d: ok", start // batch + 1, total_batches)
    return out


def load_input(path: Path) -> list[str]:
    """Загрузить текстовые фрагменты из JSONL, JSON, SRT или TXT."""
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if items and isinstance(items[0], dict) and "text" in items[0]:
            return [it["text"] for it in items]
        return [str(it) for it in items]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d.get("text", str(d)) for d in data]
        if "segments" in data:
            return [s["text"] for s in data["segments"]]
        return [str(data)]
    if path.suffix == ".srt":
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        buf: list[str] = []
        for ln in lines + [""]:
            if ln.strip().isdigit() or "-->" in ln:
                continue
            if not ln.strip():
                if buf:
                    out.append(" ".join(buf))
                    buf = []
            else:
                buf.append(ln.strip())
        return out
    return [path.read_text(encoding="utf-8")]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="JSONL/JSON/SRT/TXT с текстами")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-name", default=None)
    p.add_argument("--api-key", default=None,
                   help="OpenRouter ключ (или OPENROUTER_API_KEY env)")
    p.add_argument("--batch", type=int, default=10)
    return p.parse_args()


def main() -> None:
    """Запустить OpenRouter-классификацию и сохранить результаты ПЗ 7."""
    setup_logging()
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Нет ключа: --api-key или OPENROUTER_API_KEY env")

    items = load_input(args.input)
    logger.info("элементов на вход: %d, модель: %s", len(items), args.model)
    results = classify_texts(items, api_key, args.model, batch=args.batch)

    name = args.out_name or args.input.stem
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classified.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    from collections import Counter
    by_label = Counter(r.get("label", "other") for r in results)
    (out_dir / "summary.json").write_text(
        json.dumps({"total": len(results), "model": args.model,
                    "by_label": dict(by_label)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("классификация готова: %s", dict(by_label))


if __name__ == "__main__":
    main()
