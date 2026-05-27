"""ПЗ 7. Классификация текста локальной LLM через Ollama.

Скрипт берет титры или транскрипт из предыдущих этапов, передает их локальной
LLM батчами и сохраняет классификацию фрагментов. Модель не получает исходное
видео, поэтому этот вариант анализирует только текстовые признаки.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import requests

from common import OUTPUT_DIR, setup_logging

logger = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "pz7"
OUT.mkdir(parents=True, exist_ok=True)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"


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


def call_ollama(model: str, prompt: str, system: str = SYSTEM_PROMPT,
                temperature: float = 0.0, num_ctx: int = 2048,
                num_predict: int = 768, num_gpu: int = 33) -> str:
    """Отправить один запрос в локальный Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def warmup(model: str) -> None:
    """Выполнить короткий запрос, чтобы модель загрузилась перед основной работой."""
    logger.info("Прогрев модели %s...", model)
    payload = {
        "model": model,
        "prompt": "ok",
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": 1, "num_ctx": 2048, "num_gpu": 33},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    logger.info("Модель загружена.")


def parse_json_array(text: str) -> list[dict]:
    """Извлечь JSON-массив из текстового ответа модели."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def classify_texts(items: list[str], model: str, batch: int = 10) -> list[dict]:
    """Классифицировать список текстовых фрагментов батчами."""
    out: list[dict] = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        prompt_lines = [f"{i + 1}. {t}" for i, t in enumerate(chunk)]
        prompt = "Список фрагментов:\n" + "\n".join(prompt_lines)
        try:
            resp = call_ollama(model, prompt)
            arr = parse_json_array(resp)
        except Exception as e:
            logger.warning("ошибка LLM на батче %d: %s", start // batch, e)
            arr = []
        # Результат принудительно выравнивается по исходному порядку фрагментов.
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
        logger.info("батч %d/%d: ok", start // batch + 1,
                    (len(items) + batch - 1) // batch)
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
        # В SRT оставляем только содержательные строки без номеров и таймкодов.
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
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
    p.add_argument("input", type=Path, help="JSONL/JSON/SRT/TXT с текстами для классификации")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-name", default=None)
    return p.parse_args()


def main() -> None:
    """Запустить локальную LLM-классификацию и сохранить результаты ПЗ 7."""
    setup_logging()
    args = parse_args()
    items = load_input(args.input)
    logger.info("элементов на вход: %d", len(items))
    warmup(args.model)
    results = classify_texts(items, args.model)
    name = args.out_name or args.input.stem
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classified.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8",
    )
    # Сводка по меткам нужна для быстрого просмотра результата.
    from collections import Counter
    by_label = Counter(r.get("label", "other") for r in results)
    (out_dir / "summary.json").write_text(
        json.dumps({"total": len(results), "by_label": dict(by_label)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("классификация готова: %s", dict(by_label))


if __name__ == "__main__":
    main()
