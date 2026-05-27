"""Удобный запуск текстового варианта ПЗ 7 через OpenRouter.

Скрипт берет `OPENROUTER_API_KEY` из переменных окружения или из локального
`.env`, выбирает самый свежий транскрипт ПЗ 4 и запускает облачную
классификацию.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
HR_ENV = Path("D:/hr-breaker/.env")
DEFAULT_MODEL = "inclusionai/ling-2.6-1t:free"


def load_token() -> str | None:
    """Загрузить OpenRouter API key из окружения или локального `.env`."""
    env_token = os.environ.get("OPENROUTER_API_KEY")
    if env_token:
        return env_token
    if not HR_ENV.exists():
        return None
    for line in HR_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value
    return None


def find_transcript() -> Path:
    """Найти самый свежий транскрипт из результатов ПЗ 4."""
    candidates = list((ROOT / "output" / "pz4").glob("*/transcript.json"))
    if not candidates:
        raise FileNotFoundError(
            "Нет файлов output/pz4/<video>/transcript.json — сначала ПЗ 4"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Подготовить окружение и запустить OpenRouter-классификацию текста."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TEMP"] = "D:/Temp"
    os.environ["TMP"] = "D:/Temp"

    token = load_token()
    if not token:
        print("[err] нет OPENROUTER_API_KEY ни в env, ни в", HR_ENV)
        sys.exit(1)
    os.environ["OPENROUTER_API_KEY"] = token
    print(f"[ok] токен загружен ({token[:8]}...)")

    tr = find_transcript()
    out_name = tr.parent.name
    print(f"[..] вход: {tr}")
    print(f"[..] out-name: {out_name}")

    cmd = [
        str(VENV_PY) if VENV_PY.exists() else sys.executable,
        str(ROOT / "src" / "pz7_openrouter.py"),
        str(tr),
        "--model", DEFAULT_MODEL,
        "--out-name", out_name,
    ]
    print("[..] запуск:", " ".join(cmd))
    subprocess.run(cmd, check=False, env=os.environ)
    print(f"[ok] готово, результаты в output/pz7/{out_name}/")
    if sys.stdin.isatty():
        input("Нажми Enter для выхода...")


if __name__ == "__main__":
    main()
