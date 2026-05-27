"""Локальная проверка проекта на соответствие контракту курсовой работы."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_openapi() -> None:
    """Проверить, что OpenAPI совпадает с приложенным контрактом."""
    from api import app

    bundled = yaml.safe_load((ROOT / "docs" / "linza.detector-rest-api.yml").read_text(encoding="utf-8"))
    actual = app.openapi()

    expected_paths = {
        path: methods
        for path, methods in bundled["paths"].items()
        if path.startswith("/api/")
    }
    actual_paths = {
        path: methods
        for path, methods in actual["paths"].items()
        if path.startswith("/api/")
    }
    _assert(actual_paths == expected_paths, "пути OpenAPI /api/* отличаются от приложенного контракта")
    _assert(
        actual["components"]["schemas"] == bundled["components"]["schemas"],
        "схемы OpenAPI отличаются от приложенного контракта",
    )
    _assert(
        actual["components"]["responses"] == bundled["components"]["responses"],
        "ответы OpenAPI отличаются от приложенного контракта",
    )


def verify_packaging() -> None:
    """Проверить Docker-файлы и зависимости, нужные для контрактного запуска."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    _assert("COPY docs/ /app/docs/" in dockerfile, "Dockerfile не копирует контракт из docs")
    _assert("COPY scripts/ /app/scripts/" in dockerfile, "Dockerfile не копирует runtime scripts")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for ignored in ("data/", "dist/", "output/", "models/", ".venv/", ".env", "smb/"):
        _assert(ignored in dockerignore, f".dockerignore не содержит {ignored}")

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api_service = compose["services"]["api"]
    env = api_service["environment"]
    _assert("COURSEWORK_SMB_MOUNT_ROOT" in env, "compose не содержит COURSEWORK_SMB_MOUNT_ROOT")
    _assert("COURSEWORK_HANDOVER_COMMAND" in env, "compose не содержит COURSEWORK_HANDOVER_COMMAND")
    _assert("./smb:/app/smb:ro" in api_service["volumes"], "compose не содержит readonly SMB mount")

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("fastapi==0.115.5", "uvicorn[standard]==0.32.1", "pydantic==2.10.3"):
        _assert(package in requirements, f"requirements.txt не содержит {package}")


def verify_acceptance_inventory() -> None:
    """Проверить наличие обязательных файлов курсовой и защитных исключений."""
    required_files = [
        "README.md",
        ".gitignore",
        ".dockerignore",
        ".env.example",
        ".github/workflows/ci.yml",
        "Dockerfile",
        "docker-compose.yml",
        "docs/deploy.md",
        "docs/coursework.md",
        "docs/contract.md",
        "docs/linza.detector-rest-api.yml",
        "docs/time_based_report.sample.json",
        "src/api.py",
        "src/contract.py",
        "src/coursework_pipeline.py",
        "src/dto.py",
        "scripts/build_release.py",
        "scripts/smoke_api.py",
        "scripts/verify_contract.py",
    ]
    required_files.extend(f"src/pz{idx}_{name}.py" for idx, name in {
        1: "counters",
        2: "slicer",
        3: "subtitles",
        4: "whisper",
        5: "yolo",
        6: "resnet",
        8: "postprocess",
    }.items())
    required_files.extend(
        ["src/pz7_llm.py", "src/pz7_openrouter.py", "src/pz7_vlm_gemini.py"]
    )
    required_files.extend(f"docs/pz{idx}.md" for idx in range(1, 9))
    for relative in required_files:
        _assert((ROOT / relative).exists(), f"отсутствует обязательный файл: {relative}")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for ignored in ("data/", "dist/", "models/", "output/", ".env", ".env.*", ".venv/"):
        _assert(ignored in gitignore, f".gitignore не исключает приватный/тяжелый артефакт: {ignored}")

    pipeline = (ROOT / "src" / "coursework_pipeline.py").read_text(encoding="utf-8")
    for step in ("pz2_slicer.py", "pz3_subtitles.py", "pz4_whisper.py", "pz5_yolo.py",
                 "pz6_resnet.py", "pz7_llm.py", "pz7_openrouter.py", "pz7_vlm_gemini.py",
                 "pz8_postprocess.py", "job_result.json", "time_based_report.json"):
        _assert(step in pipeline, f"пайплайн курсовой не ссылается на {step}")

    deploy_doc = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    for marker in ("Timeweb", "docker compose", "/api/jobs", "COURSEWORK_LLM_BACKEND=none"):
        _assert(marker in deploy_doc, f"deploy.md не содержит {marker}")

    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    contract_job = ci["jobs"]["contract"]
    step_text = "\n".join(
        step.get("run", "") for step in contract_job["steps"] if "run" in step
    )
    for command in (
        "ruff check",
        "pytest -q tests",
        "python scripts/verify_contract.py",
        "python scripts/build_release.py --check",
    ):
        _assert(command in step_text, f"GitHub Actions workflow не содержит {command}")

    coursework_doc = (ROOT / "docs" / "coursework.md").read_text(encoding="utf-8")
    for marker in ("Код на GitHub", "Развёрнутое решение на ВМ", "TIME_BASED_REPORT"):
        _assert(marker in coursework_doc, f"coursework.md не содержит {marker}")


def verify_release_file_list() -> None:
    """Проверить, что release builder публикует только разрешенные файлы."""
    from build_release import FORBIDDEN_PARTS, iter_release_files

    files = iter_release_files()
    _assert(files, "список файлов релиза пуст")
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}
    for required in (
        ".env.example",
        ".github/workflows/ci.yml",
        "Dockerfile",
        "README.md",
        "docs/linza.detector-rest-api.yml",
        "scripts/build_release.py",
        "scripts/smoke_api.py",
        "scripts/verify_contract.py",
        "src/api.py",
        "src/coursework_pipeline.py",
        "tests/test_api_contract.py",
    ):
        _assert(required in relative_files, f"список файлов релиза не содержит {required}")
    for path in files:
        relative = path.relative_to(ROOT)
        _assert(
            not any(part in FORBIDDEN_PARTS for part in relative.parts),
            f"список файлов релиза содержит запрещенный путь: {relative}",
        )
    release_builder = (ROOT / "scripts" / "build_release.py").read_text(encoding="utf-8")
    for marker in ("write_manifest", "sha256_file", "containsPrivateArtifacts"):
        _assert(marker in release_builder, f"release builder не содержит {marker}")


def verify_time_based_report_shape() -> None:
    """Проверить структуру `TIME_BASED_REPORT` по приложенному образцу."""
    from contract import build_time_based_report
    from dto import (
        Detection,
        DetectionClassStatistic,
        EstimatedPenalty,
        JobResult,
        Modality,
        SourceInfo,
    )

    sample = json.loads(
        (ROOT / "docs" / "time_based_report.sample.json").read_text(encoding="utf-8")
    )
    result = JobResult(
        processingDurationSeconds=12.0,
        sourceInfo=SourceInfo(frameCount=30, fps=10.0, durationSeconds=3.0),
        totalDetections=1,
        detectionClassStatistics=[
            DetectionClassStatistic(
                **{"class": "DRUGS"},
                subclass="ALCOHOL",
                durationFrames=10,
                durationSeconds=1.0,
                count=1,
                estimatedPenalty=EstimatedPenalty(amount=1000, currency="RUB"),
            )
        ],
        detections=[
            Detection(
                startFrame=0,
                endFrame=9,
                startSeconds=0.0,
                endSeconds=1.0,
                **{"class": "DRUGS"},
                subclass="ALCOHOL",
                confidence=0.9,
                modality=Modality.VIDEO,
            )
        ],
    )
    generated = build_time_based_report(result, video_path="sample.mp4")

    _assert(generated["report_type"] == sample["report_type"], "report_type не совпадает")
    _assert(set(generated) == set(sample), "верхнеуровневые ключи TIME_BASED_REPORT не совпадают")
    _assert(
        set(generated["source_info"]) == set(sample["source_info"]),
        "ключи TIME_BASED_REPORT source_info не совпадают",
    )
    _assert(
        set(generated["sourceInfo"]) == set(sample["sourceInfo"]),
        "ключи TIME_BASED_REPORT sourceInfo не совпадают",
    )
    _assert(
        set(generated["detections"][0]) == set(sample["detections"][0]),
        "ключи detection в TIME_BASED_REPORT не совпадают",
    )


def main() -> None:
    verify_openapi()
    verify_packaging()
    verify_acceptance_inventory()
    verify_release_file_list()
    verify_time_based_report_shape()
    print("проверка контракта пройдена")


if __name__ == "__main__":
    main()
