"""Удобный запуск VLM-варианта ПЗ 7 через OpenRouter.

Скрипт берет API key из окружения или локального `.env`, выбирает самую свежую
нарезку кадров из ПЗ 2 и запускает VLM-распознавание объектов.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
HR_ENV = Path("D:/hr-breaker/.env")
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_EVERY_N = 30  # При 1 fps это примерно один анализируемый кадр за 30 секунд.


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
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def find_frames_dir() -> Path:
    """Найти самый свежий каталог кадров из результатов ПЗ 2."""
    dirs = [p for p in (ROOT / "output" / "pz2").iterdir() if p.is_dir()
            and any(p.glob("frame_*.jpg"))]
    if not dirs:
        raise FileNotFoundError("Нет нарезки в output/pz2/<video>/")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Подготовить окружение и запустить VLM-анализ кадров."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TEMP"] = "D:/Temp"
    os.environ["TMP"] = "D:/Temp"

    token = load_token()
    if not token:
        print("[err] нет OPENROUTER_API_KEY ни в env, ни в", HR_ENV)
        sys.exit(1)
    os.environ["OPENROUTER_API_KEY"] = token
    print(f"[ok] токен загружен ({token[:8]}...)")

    frames = find_frames_dir()
    print(f"[..] кадры: {frames}")
    print(f"[..] прореживание: каждый {DEFAULT_EVERY_N}-й кадр")

    cmd = [
        str(VENV_PY) if VENV_PY.exists() else sys.executable,
        str(ROOT / "src" / "pz7_vlm_gemini.py"),
        str(frames),
        "--out-name", frames.name,
        "--model", DEFAULT_MODEL,
        "--every-n", str(DEFAULT_EVERY_N),
    ]
    print("[..] запуск:", " ".join(cmd))
    subprocess.run(cmd, check=False, env=os.environ)
    print(f"[ok] готово: output/pz7/{frames.name}/")
    if sys.stdin.isatty():
        input("Нажми Enter...")


if __name__ == "__main__":
    main()
