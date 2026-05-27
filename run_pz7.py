"""Удобный запуск ПЗ 7 через локальную Ollama-модель без ручных аргументов."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
OLLAMA_EXE = Path("D:/ollama/ollama.exe")
OLLAMA_URL = "http://127.0.0.1:11434"


def ensure_ollama() -> None:
    """Запустить `ollama serve`, если локальный API еще не отвечает."""
    try:
        if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
            print("[ok] ollama уже запущен")
            return
    except Exception:
        pass
    print("[..] стартую ollama serve")
    subprocess.Popen(
        [str(OLLAMA_EXE), "serve"],
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    for _ in range(30):
        time.sleep(1)
        try:
            if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
                print("[ok] ollama поднят")
                return
        except Exception:
            continue
    raise RuntimeError("не удалось поднять ollama")


def find_transcript() -> Path:
    """Найти самый свежий транскрипт из результатов ПЗ 4."""
    candidates = list((ROOT / "output" / "pz4").glob("*/transcript.json"))
    if not candidates:
        raise FileNotFoundError("нет файлов output/pz4/<video>/transcript.json — сначала запусти ПЗ 4")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Подготовить окружение и запустить текстовую LLM-классификацию."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TEMP"] = "D:/Temp"
    os.environ["TMP"] = "D:/Temp"

    ensure_ollama()
    tr = find_transcript()
    out_name = tr.parent.name
    print(f"[..] вход: {tr}")
    print(f"[..] out-name: {out_name}")

    cmd = [
        str(VENV_PY) if VENV_PY.exists() else sys.executable,
        str(ROOT / "src" / "pz7_llm.py"),
        str(tr),
        "--model", "qwen3.5:9b",
        "--out-name", out_name,
    ]
    print("[..] запуск:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    print("[ok] готово, результаты в output/pz7/" + out_name + "/")
    if sys.stdin.isatty():
        input("Нажми Enter для выхода...")


if __name__ == "__main__":
    main()
