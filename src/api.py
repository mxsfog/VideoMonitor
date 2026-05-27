"""REST API учебного сервиса анализа видео.

Основные контрактные маршруты:
- GET    /api/jobs
- POST   /api/jobs
- GET    /api/jobs/{jobId}
- DELETE /api/jobs/{jobId}
- GET    /api/billing/{customerId}
- POST   /api/admin/handover-access

Маршруты `/process`, `/upload` и `/jobs...` оставлены как учебные совместимые
входы поверх той же очереди задач.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import DATA_DIR, OUTPUT_DIR  # noqa: E402
from dto import (  # noqa: E402
    VALID_SUBCLASSES,
    BillingResponse,
    DetectionClass,
    ErrorInfo,
    ErrorResponse,
    Job,
    JobCreateRequest,
    JobCreateResponse,
    JobRequest,
    JobStatus,
    Profile,
)

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTRACT_OPENAPI_PATH = ROOT / "docs" / "linza.detector-rest-api.yml"
PYTHON = sys.executable
JOBS_DIR = OUTPUT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("COURSEWORK_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
REMOTE_SOURCE_TIMEOUT_SECONDS = int(os.environ.get("COURSEWORK_REMOTE_SOURCE_TIMEOUT", "60"))
HANDOVER_TIMEOUT_SECONDS = int(os.environ.get("COURSEWORK_HANDOVER_TIMEOUT", "30"))
DEFAULT_LLM_BACKEND = os.environ.get("COURSEWORK_LLM_BACKEND", "none")
if DEFAULT_LLM_BACKEND not in {"ollama", "openrouter", "vlm", "none"}:
    DEFAULT_LLM_BACKEND = "none"
DEFAULT_LLM_MODEL = os.environ.get("COURSEWORK_LLM_MODEL", "qwen3.5:9b")

app = FastAPI(
    title="REST API Линза.Детектор",
    description="Pipeline ПЗ 1-8 для анализа видео в контрактном формате.",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("COURSEWORK_CORS_ORIGINS", "").split(",")
    if os.environ.get("COURSEWORK_CORS_ORIGINS")
    else [],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_futures: dict[str, Future] = {}
_processes: dict[str, subprocess.Popen] = {}
PUBLIC_JOB_KEYS = {
    "jobId",
    "status",
    "createdAt",
    "request",
    "startedAt",
    "finishedAt",
    "result",
    "error",
}
ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


class LegacyProcessRequest(BaseModel):
    """Упрощённый запрос для маршрутов `/process` и `/upload`."""

    source: str = Field(..., description="URL или путь к видео")
    fps: float = 1.0
    whisper_model: str = "tiny"
    lang: str = "ru"
    llm_backend: Literal["ollama", "openrouter", "vlm", "none"] = "none"
    llm_model: str = "qwen3.5:9b"
    vlm_every_n: int = 5
    skip: list[str] = Field(default_factory=list)


def utc_now() -> str:
    """Текущее время в формате OpenAPI date-time."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def error_response(message: str, status_code: int) -> JSONResponse:
    """Единая форма ошибки для контрактных маршрутов."""
    payload = ErrorResponse(error=ErrorInfo(message=message)).model_dump(by_alias=True)
    return JSONResponse(status_code=status_code, content=payload)


def contract_openapi() -> dict:
    """Собрать OpenAPI, сохранив канонические схемы `/api/*`."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    if CONTRACT_OPENAPI_PATH.exists():
        canonical = yaml.safe_load(CONTRACT_OPENAPI_PATH.read_text(encoding="utf-8"))
        schema["openapi"] = canonical.get("openapi", schema.get("openapi"))
        schema["info"] = canonical.get("info", schema.get("info", {}))
        schema.setdefault("components", {})
        for component_name, component_payload in canonical.get("components", {}).items():
            schema["components"][component_name] = component_payload
        for path, methods in canonical.get("paths", {}).items():
            if path.startswith("/api/"):
                schema.setdefault("paths", {})[path] = methods
    for path, methods in schema.get("paths", {}).items():
        if not path.startswith("/api/"):
            continue
        for operation in methods.values():
            operation.get("responses", {}).pop("422", None)
    app.openapi_schema = schema
    return app.openapi_schema


@app.exception_handler(HTTPException)
async def http_error_handler(_, exc: HTTPException) -> JSONResponse:
    """Привести HTTPException к контрактной форме ответа."""
    return error_response(str(exc.detail), exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    """Вернуть ошибку валидации как `400`, без стандартного `422`."""
    return error_response(str(exc.errors()), 400)


def state_path(job_id: str) -> Path:
    """Путь к сохранённому состоянию задачи."""
    return JOBS_DIR / job_id / "state.json"


def job_dir(job_id: str) -> Path:
    """Рабочий каталог задачи."""
    return JOBS_DIR / job_id


def write_state(job_id: str, record: dict) -> None:
    """Записать состояние задачи на диск."""
    path = state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jobs_from_disk() -> None:
    """Восстановить сохранённые задачи при старте сервиса."""
    for path in JOBS_DIR.glob("*/state.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("пропущено поврежденное состояние задачи: %s", path)
            continue
        if record.get("forgotten"):
            continue
        job_id = record.get("jobId")
        if job_id:
            if record.get("status") in {JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value}:
                record["status"] = JobStatus.ERROR.value
                record["finishedAt"] = record.get("finishedAt") or utc_now()
                record["error"] = {
                    "message": "сервер был перезапущен до завершения задачи",
                }
                write_state(job_id, record)
            _jobs[job_id] = record


def save_new_job(job_id: str, request: JobRequest) -> dict:
    """Создать задачу в статусе `PENDING` и сохранить её."""
    record = {
        "jobId": job_id,
        "status": JobStatus.PENDING.value,
        "createdAt": utc_now(),
        "request": request.model_dump(by_alias=True, exclude_none=True),
    }
    with _jobs_lock:
        _jobs[job_id] = record
    write_state(job_id, record)
    return record


def update_job(job_id: str, **fields) -> dict | None:
    """Обновить задачу, если она ещё не удалена из списка."""
    with _jobs_lock:
        record = _jobs.get(job_id)
        if record is None or record.get("forgotten"):
            return None
        record.update(fields)
        saved = dict(record)
    write_state(job_id, saved)
    return saved


def get_record(job_id: str) -> dict:
    """Получить задачу из памяти или с диска."""
    with _jobs_lock:
        record = _jobs.get(job_id)
    if record is None:
        path = state_path(job_id)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if not record.get("forgotten"):
                return record
    elif not record.get("forgotten"):
        return record
    raise HTTPException(404, "Задача не найдена.")


def public_job(record: dict) -> dict:
    """Оставить только поля публичного `Job`."""
    return {
        key: value
        for key, value in record.items()
        if key in PUBLIC_JOB_KEYS and value is not None
    }


def is_supported_source(source: str) -> bool:
    """Проверить, что источник видео поддерживается текущей сборкой."""
    if source.startswith(("http://", "https://")):
        return True
    if source.startswith("smb://"):
        smb_path = resolve_smb_source(source)
        return smb_path is not None and smb_path.exists()
    return Path(source).exists()


def resolve_smb_source(source: str) -> Path | None:
    """Преобразовать `smb://` в путь внутри заранее смонтированного каталога."""
    if not source.startswith("smb://"):
        return None
    mount_root = os.environ.get("COURSEWORK_SMB_MOUNT_ROOT")
    if not mount_root:
        return None
    parsed = urlparse(source)
    if not parsed.netloc:
        return None
    relative_parts = [parsed.netloc, *Path(unquote(parsed.path).lstrip("/")).parts]
    return Path(mount_root).joinpath(*relative_parts)


def remote_source_target(job_id: str, source: str) -> Path:
    """Локальный путь для скачанного удалённого источника."""
    parsed = urlparse(source)
    suffix = Path(unquote(parsed.path)).suffix or ".mp4"
    return job_dir(job_id) / f"source{suffix}"


def download_remote_source(job_id: str, source: str) -> Path:
    """Скачать HTTP(S)-источник с ограничением размера."""
    target = remote_source_target(job_id, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = UrlRequest(source, headers={"User-Agent": "LinzaDetectorCoursework/1.0"})
    size = 0
    try:
        with urlopen(request, timeout=REMOTE_SOURCE_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                raise ValueError("remote source is too large")
            with target.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        target.unlink(missing_ok=True)
                        raise ValueError("remote source is too large")
                    out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def prepare_pipeline_source(job_id: str, source: str) -> str:
    """Подготовить локальный путь, который можно передать pipeline."""
    if source.startswith(("http://", "https://")):
        return str(download_remote_source(job_id, source))
    if source.startswith("smb://"):
        smb_path = resolve_smb_source(source)
        if smb_path is None:
            raise ValueError("smb:// source requires COURSEWORK_SMB_MOUNT_ROOT")
        if not smb_path.exists():
            raise FileNotFoundError(f"SMB-источник не найден после разрешения пути: {smb_path}")
        return str(smb_path)
    return source


def ensure_unique_job_id(job_id: str) -> None:
    """Не допустить повторное использование `jobId`."""
    with _jobs_lock:
        exists_in_memory = job_id in _jobs and not _jobs[job_id].get("forgotten")
    if exists_in_memory:
        raise HTTPException(409, f"Задача с указанным jobId='{job_id}' уже существует")
    if state_path(job_id).exists():
        record = json.loads(state_path(job_id).read_text(encoding="utf-8"))
        if not record.get("forgotten"):
            raise HTTPException(409, f"Задача с указанным jobId='{job_id}' уже существует")


def sanitized_job_request(req: JobCreateRequest) -> JobRequest:
    """Сформировать запрос для хранения без `sourceCredentials`."""
    return req.to_job_request()


def write_request_file(job_id: str, req: JobCreateRequest) -> Path:
    """Сохранить обезличенный запрос рядом с состоянием задачи."""
    path = job_dir(job_id) / "request.json"
    payload = req.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude={"source_credentials"},
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def artifact_mapping(video_stem: str, pipeline_state: dict) -> dict[str, str]:
    """Собрать индекс артефактов для завершённой задачи."""
    return {
        "report.md": pipeline_state.get(
            "report_path",
            str(OUTPUT_DIR / "coursework" / video_stem / "report.md"),
        ),
        "findings.json": pipeline_state.get(
            "findings_path",
            str(OUTPUT_DIR / "coursework" / video_stem / "findings.json"),
        ),
        "job_result.json": pipeline_state.get(
            "job_result_path",
            str(OUTPUT_DIR / "coursework" / video_stem / "job_result.json"),
        ),
        "time_based_report.json": pipeline_state.get(
            "time_based_report_path",
            str(OUTPUT_DIR / "coursework" / video_stem / "time_based_report.json"),
        ),
        "transcript.json": str(OUTPUT_DIR / "pz4" / video_stem / "transcript.json"),
        "transcript.srt": str(OUTPUT_DIR / "pz4" / video_stem / "transcript.srt"),
        "transcript.txt": str(OUTPUT_DIR / "pz4" / video_stem / "transcript.txt"),
        "subtitles.json": str(OUTPUT_DIR / "pz3" / video_stem / "subtitles.json"),
        "subtitles.srt": str(OUTPUT_DIR / "pz3" / video_stem / "subtitles.srt"),
        "yolo_detections.jsonl": str(OUTPUT_DIR / "pz5" / video_stem / "detections.jsonl"),
        "yolo_summary.json": str(OUTPUT_DIR / "pz5" / video_stem / "summary.json"),
        "resnet_predictions.jsonl": str(OUTPUT_DIR / "pz6" / video_stem / "predictions.jsonl"),
        "resnet_summary.json": str(OUTPUT_DIR / "pz6" / video_stem / "summary.json"),
        "vlm_classified.jsonl": str(OUTPUT_DIR / "pz7" / video_stem / "classified.jsonl"),
        "vlm_summary.json": str(OUTPUT_DIR / "pz7" / video_stem / "summary.json"),
        "tracks.json": str(OUTPUT_DIR / "pz8" / video_stem / "tracks.json"),
        "subs_dedup.json": str(OUTPUT_DIR / "pz8" / video_stem / "subs_dedup.json"),
    }


def pipeline_command(
    job_id: str,
    req: JobCreateRequest,
    request_path: Path,
    state_path_: Path,
    pipeline_options: dict | None = None,
    source: str | None = None,
) -> list[str]:
    """Собрать команду запуска pipeline без shell-интерпретации."""
    options = pipeline_options or {}
    llm_backend = options.get("llm_backend", DEFAULT_LLM_BACKEND)
    llm_model = options.get("llm_model", DEFAULT_LLM_MODEL)
    cmd = [
        PYTHON,
        str(SRC / "coursework_pipeline.py"),
        source or req.source,
        "--fps",
        str(options.get("fps", os.environ.get("COURSEWORK_PIPELINE_FPS", "1.0"))),
        "--whisper-model",
        str(options.get("whisper_model", os.environ.get("COURSEWORK_WHISPER_MODEL", "tiny"))),
        "--lang",
        str(options.get("lang", os.environ.get("COURSEWORK_LANG", "ru"))),
        "--llm-backend",
        str(llm_backend),
        "--llm-model",
        str(llm_model),
        "--job-id",
        job_id,
        "--job-request",
        str(request_path),
        "--job-state",
        str(state_path_),
    ]
    if llm_backend == "vlm":
        cmd += [
            "--vlm-every-n",
            str(options.get("vlm_every_n", os.environ.get("COURSEWORK_VLM_EVERY_N", "5"))),
        ]
    skip = options.get("skip")
    if skip is None:
        skip_env = os.environ.get("COURSEWORK_PIPELINE_SKIP", "")
        skip = [item.strip() for item in skip_env.split(",") if item.strip()]
    if skip:
        cmd += ["--skip", ",".join(skip)]
    return cmd


def run_pipeline(
    job_id: str,
    req: JobCreateRequest,
    pipeline_options: dict | None = None,
) -> None:
    """Выполнить одну задачу в фоновом worker."""
    started_at = utc_now()
    update_job(job_id, status=JobStatus.IN_PROGRESS.value, startedAt=started_at)
    current_job_dir = job_dir(job_id)
    current_job_dir.mkdir(parents=True, exist_ok=True)
    request_path = write_request_file(job_id, req)
    pipeline_state_path = current_job_dir / "pipeline_state.json"
    log_path = current_job_dir / "log.txt"

    try:
        pipeline_source = prepare_pipeline_source(job_id, req.source)
        cmd = pipeline_command(
            job_id,
            req,
            request_path,
            pipeline_state_path,
            pipeline_options,
            source=pipeline_source,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                env={**os.environ},
            )
            with _jobs_lock:
                _processes[job_id] = proc
            exit_code = proc.wait()
        with _jobs_lock:
            _processes.pop(job_id, None)

        if exit_code != 0:
            update_job(
                job_id,
                status=JobStatus.ERROR.value,
                finishedAt=utc_now(),
                error={"message": f"pipeline exited with code {exit_code}"},
            )
            return

        if not pipeline_state_path.exists():
            update_job(
                job_id,
                status=JobStatus.ERROR.value,
                finishedAt=utc_now(),
                error={"message": "pipeline_state.json не был создан"},
            )
            return

        pipeline_state = json.loads(pipeline_state_path.read_text(encoding="utf-8"))
        update_job(
            job_id,
            status=JobStatus.DONE.value,
            startedAt=started_at,
            finishedAt=utc_now(),
            result=pipeline_state["result"],
            video_stem=pipeline_state.get("video_stem"),
            artifacts=artifact_mapping(pipeline_state["video_stem"], pipeline_state),
        )
    except Exception as exc:
        logger.exception("[%s] pipeline worker завершился ошибкой", job_id)
        with _jobs_lock:
            _processes.pop(job_id, None)
        update_job(
            job_id,
            status=JobStatus.ERROR.value,
            finishedAt=utc_now(),
            error={"message": str(exc)},
        )


def submit_job(req: JobCreateRequest, pipeline_options: dict | None = None) -> dict:
    """Проверить запрос, сохранить задачу и поставить её в очередь."""
    ensure_unique_job_id(req.job_id)
    if not is_supported_source(req.source):
        if req.source.startswith("smb://"):
            raise HTTPException(
                400,
                "smb:// source requires a mounted path configured by COURSEWORK_SMB_MOUNT_ROOT",
            )
        raise HTTPException(400, f"source не найден или не поддерживается: {req.source}")
    record = save_new_job(req.job_id, sanitized_job_request(req))
    future = _executor.submit(run_pipeline, req.job_id, req, pipeline_options)
    with _jobs_lock:
        _futures[req.job_id] = future
    return record


def all_detection_classes() -> list[DetectionClass]:
    """Список всех подклассов для упрощённых маршрутов."""
    return [
        DetectionClass.model_validate({"class": name})
        for name in VALID_SUBCLASSES
    ]


def legacy_to_job_request(req: LegacyProcessRequest, job_id: str) -> JobCreateRequest:
    """Преобразовать упрощённый запрос в контрактный формат."""
    return JobCreateRequest(
        jobId=job_id,
        source=req.source,
        profile=Profile.FULL,
        detectionClasses=all_detection_classes(),
    )


@app.get("/health")
def health() -> dict:
    """Проверка доступности сервиса."""
    return {"status": "ok", "time": utc_now()}


@app.get(
    "/api/jobs",
    response_model=list[Job],
    response_model_exclude_none=True,
    responses={500: ERROR_RESPONSES[500]},
)
def list_api_jobs() -> list[dict]:
    """Список задач, которые ещё не были удалены пользователем."""
    with _jobs_lock:
        return [
            public_job(record)
            for record in _jobs.values()
            if not record.get("forgotten")
        ]


@app.post(
    "/api/jobs",
    status_code=201,
    response_model=JobCreateResponse,
    responses={
        201: {
            "description": "Задача создана. В теле возвращается `jobId`; также устанавливается заголовок Location.",
            "headers": {
                "Location": {
                    "description": "URI созданного ресурса",
                    "schema": {"type": "string"},
                    "example": "/api/jobs/acme-20251006-0001",
                }
            },
        },
        400: ERROR_RESPONSES[400],
        409: ERROR_RESPONSES[409],
        500: ERROR_RESPONSES[500],
    },
)
def create_api_job(req: JobCreateRequest) -> JSONResponse:
    """Создать задачу через контрактный маршрут."""
    record = submit_job(req)
    return JSONResponse(
        status_code=201,
        content={"jobId": record["jobId"]},
        headers={"Location": f"/api/jobs/{record['jobId']}"},
    )


@app.get(
    "/api/jobs/{jobId}",
    response_model=Job,
    response_model_exclude_none=True,
    responses={404: ERROR_RESPONSES[404], 500: ERROR_RESPONSES[500]},
)
def get_api_job(jobId: str) -> dict:
    """Вернуть состояние одной задачи."""
    return public_job(get_record(jobId))


@app.delete(
    "/api/jobs/{jobId}",
    status_code=204,
    responses={500: ERROR_RESPONSES[500]},
)
def delete_api_job(jobId: str) -> Response:
    """Удалить задачу из списка и остановить процесс, если он ещё работает."""
    with _jobs_lock:
        record = _jobs.get(jobId)
        future = _futures.pop(jobId, None)
        proc = _processes.pop(jobId, None)
        if record is None and not state_path(jobId).exists():
            return Response(status_code=204)
        if record is None:
            record = json.loads(state_path(jobId).read_text(encoding="utf-8"))
        record["forgotten"] = True
        record["forgottenAt"] = utc_now()
        _jobs.pop(jobId, None)

    if future is not None:
        future.cancel()
    if proc is not None and proc.poll() is None:
        proc.terminate()
    write_state(jobId, record)
    return Response(status_code=204)


@app.get(
    "/api/billing/{customerId}",
    response_model=BillingResponse,
    response_model_exclude_none=True,
    responses={404: ERROR_RESPONSES[404], 500: ERROR_RESPONSES[500]},
)
def get_billing(customerId: str) -> dict:
    """Вернуть локальную учебную информацию по тарифу."""
    return {
        "customerId": customerId,
        "balance": {"amount": 0.0, "currency": "RUB"},
        "tariff": {
            "name": "CourseworkLocal",
            "start": "2026-01-01T00:00:00Z",
            "end": "2027-01-01T00:00:00Z",
        },
        "billingPeriod": {
            "start": "2026-05-01T00:00:00Z",
            "end": "2026-06-01T00:00:00Z",
        },
        "spent": {"timeMinutes": 0, "detections": 0},
        "remaining": {"timeMinutes": 0, "detections": 0},
    }


@app.post(
    "/api/admin/handover-access",
    status_code=204,
    responses={400: ERROR_RESPONSES[400], 500: ERROR_RESPONSES[500]},
)
def handover_access() -> Response:
    """Выполнить явно настроенную команду передачи доступа."""
    command = os.environ.get("COURSEWORK_HANDOVER_COMMAND")
    if command:
        try:
            subprocess.run(
                shlex.split(command),
                check=True,
                timeout=HANDOVER_TIMEOUT_SECONDS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(500, "команда передачи SSH-доступа превысила таймаут") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, "команда передачи SSH-доступа завершилась ошибкой") from exc
        return Response(status_code=204)
    if os.environ.get("COURSEWORK_ALLOW_HANDOVER_NOOP") == "1":
        return Response(status_code=204)
    raise HTTPException(400, "передача SSH-доступа отключена в этой сборке курсовой")


@app.post("/process")
def process(req: LegacyProcessRequest) -> dict:
    """Создать задачу через упрощённый маршрут `/process`."""
    job_id = uuid.uuid4().hex[:12]
    contract_req = legacy_to_job_request(req, job_id)
    submit_job(contract_req, req.model_dump())
    return {"job_id": job_id, "jobId": job_id, "status": JobStatus.PENDING.value}


@app.post("/upload")
async def upload(request: Request) -> dict:
    """Принять видео через multipart-form и создать задачу."""
    try:
        form = await request.form()
    except AssertionError as exc:
        raise HTTPException(400, "для загрузки файлов требуется python-multipart") from exc

    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(400, "в форме обязательно поле 'file'")

    job_id = uuid.uuid4().hex[:12]
    target_dir = DATA_DIR / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(getattr(file, "filename", None) or "upload.mp4").name
    target = target_dir / f"{job_id}_{safe_name}"

    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                raise HTTPException(400, "uploaded file is too large")
            out.write(chunk)

    legacy_req = LegacyProcessRequest(
        source=str(target),
        fps=float(form.get("fps", 1.0)),
        whisper_model=str(form.get("whisper_model", "tiny")),
        lang=str(form.get("lang", "ru")),
        llm_backend=str(form.get("llm_backend", "none")),  # type: ignore[arg-type]
        llm_model=str(form.get("llm_model", "qwen3.5:9b")),
        vlm_every_n=int(form.get("vlm_every_n", 5)),
    )
    contract_req = legacy_to_job_request(legacy_req, job_id)
    submit_job(contract_req, legacy_req.model_dump())
    return {"job_id": job_id, "jobId": job_id, "status": JobStatus.PENDING.value}


@app.get("/jobs")
def list_jobs() -> list[dict]:
    """Упрощённый список задач."""
    return list_api_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Упрощённое получение задачи."""
    return get_api_job(job_id)


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str) -> str:
    """Лог stdout/stderr для pipeline конкретной задачи."""
    log_path = job_dir(job_id) / "log.txt"
    if not log_path.exists():
        raise HTTPException(404, "лог не найден")
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.get("/jobs/{job_id}/report", response_class=PlainTextResponse)
def get_report(job_id: str) -> str:
    """Markdown-отчёт завершённой задачи."""
    record = get_record(job_id)
    report = record.get("artifacts", {}).get("report.md")
    if not report:
        raise HTTPException(404, "отчёт ещё не готов")
    path = Path(report)
    if not path.exists():
        raise HTTPException(404, "report.md не найден")
    return path.read_text(encoding="utf-8")


@app.get("/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str) -> FileResponse:
    """Отдать выбранный артефакт завершённой задачи."""
    record = get_record(job_id)
    artifacts = record.get("artifacts", {})
    path = artifacts.get(name)
    if path is None:
        raise HTTPException(400, f"неизвестный артефакт: {name}")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(404, f"{name} не найден")
    return FileResponse(file_path)


load_jobs_from_disk()
app.openapi = contract_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
