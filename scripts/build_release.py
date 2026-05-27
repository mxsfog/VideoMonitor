"""Сборка безопасного архива курсовой для GitHub и передачи на ВМ."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist"
DEFAULT_ARCHIVE = DEFAULT_DIST / "coursework_release.zip"

INCLUDE_PATHS = [
    ".dockerignore",
    ".env.example",
    ".github",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "docker-compose.yml",
    "docs",
    "pyproject.toml",
    "requirements-server.txt",
    "requirements.txt",
    "run_pz7.py",
    "run_pz7_openrouter.py",
    "run_pz7_vlm.py",
    "scripts",
    "src",
    "tests",
]

FORBIDDEN_PARTS = {
    ".env",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "dist",
    "models",
    "output",
    "smb",
    "venv",
}
FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".onnx",
    ".pt",
    ".pyc",
    ".webm",
    ".xlsx",
    ".zip",
}


def is_forbidden(path: Path) -> bool:
    """Проверить, что файл нельзя включать в публичный архив."""
    relative = path.relative_to(ROOT)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return True
    return path.suffix.lower() in FORBIDDEN_SUFFIXES


def iter_release_files() -> list[Path]:
    """Собрать отсортированный список файлов из явного allowlist."""
    files: list[Path] = []
    for item in INCLUDE_PATHS:
        source = ROOT / item
        if not source.exists():
            raise FileNotFoundError(f"отсутствует обязательный путь релиза: {item}")
        if source.is_file():
            if is_forbidden(source):
                raise RuntimeError(f"обязательный файл релиза попал в запрещенные: {item}")
            files.append(source)
            continue
        for path in source.rglob("*"):
            if path.is_file() and not is_forbidden(path):
                files.append(path)
    return sorted(set(files))


def build_archive(target: Path) -> Path:
    """Собрать zip-архив релиза и вернуть путь к нему."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in iter_release_files():
            archive.write(path, path.relative_to(ROOT))
    verify_archive(target)
    write_manifest(target)
    return target


def verify_archive(path: Path) -> None:
    """Проверить архив на отсутствие приватных и тяжелых артефактов."""
    with zipfile.ZipFile(path) as archive:
        names = [Path(name) for name in archive.namelist()]
    forbidden = []
    for name in names:
        if any(part in FORBIDDEN_PARTS for part in name.parts):
            forbidden.append(name)
        elif name.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(name)
    if forbidden:
        joined = ", ".join(str(item) for item in sorted(forbidden))
        raise RuntimeError(f"архив релиза содержит запрещенные артефакты: {joined}")


def sha256_file(path: Path) -> str:
    """Посчитать SHA-256 для файла."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(archive_path: Path) -> Path:
    """Записать машинно-читаемый manifest рядом с архивом релиза."""
    with zipfile.ZipFile(archive_path) as archive:
        files = sorted(archive.namelist())
    manifest = {
        "archive": archive_path.name,
        "createdAt": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fileCount": len(files),
        "sizeBytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "containsPrivateArtifacts": False,
        "excluded": sorted(FORBIDDEN_PARTS),
        "files": files,
    }
    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка безопасного архива курсовой")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="путь к создаваемому zip-архиву",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить список файлов без создания архива",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_release_files()
    if args.check:
        print(f"список файлов релиза проверен: {len(files)} файлов")
        return
    archive = build_archive(args.target)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"архив релиза собран: {archive} ({size_mb:.2f} MB)")
    print(f"manifest релиза собран: {archive.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
