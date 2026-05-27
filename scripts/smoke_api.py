"""Проверка работающего REST API на базовый контракт Линза.Детектор."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

DEFAULT_VIDEO = Path(__file__).resolve().parents[1] / "data" / "videos" / (
    "sample.mp4"
)
DETECTION_CLASSES = [
    {"class": "DRUGS"},
    {"class": "DEVIANT"},
    {"class": "TERRORISM"},
    {"class": "SEX"},
    {"class": "ANTITRADITIONAL"},
    {"class": "ANTIPATRIOTIC"},
    {"class": "LUDOMANIA"},
]


def request_json(
    *,
    base_url: str,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict | list | str, object]:
    """Отправить JSON-запрос и вернуть статус, тело ответа и заголовки."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw else ""
            return response.status, payload, response.headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        payload = json.loads(raw) if raw else ""
        return exc.code, payload, exc.headers


def wait_for_done(
    *,
    base_url: str,
    job_id: str,
    timeout_seconds: float,
    poll_interval: float,
) -> dict:
    """Опросить задачу до финального статуса или таймаута."""
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict | list | str = ""
    while time.monotonic() < deadline:
        status, payload, _ = request_json(
            base_url=base_url,
            method="GET",
            path=f"/api/jobs/{job_id}",
        )
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"GET /api/jobs/{job_id} вернул ошибку: {status} {payload}")
        last_payload = payload
        if payload.get("status") == "DONE":
            return payload
        if payload.get("status") == "ERROR":
            raise RuntimeError(f"задача завершилась ошибкой: {payload.get('error')}")
        time.sleep(poll_interval)
    raise TimeoutError(f"задача не завершилась за {timeout_seconds}s: {last_payload}")


def default_source() -> str:
    """Вернуть путь к встроенному smoke-видео, если оно есть в пакете."""
    if DEFAULT_VIDEO.exists():
        return str(DEFAULT_VIDEO)
    raise FileNotFoundError(
        "не найдено smoke-видео по умолчанию; передайте --source /path/to/video.mp4"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка REST API по базовому контракту")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--source", default=None, help="локальный или удаленный видеофайл для /api/jobs")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--customer-id", default="smoke-customer")
    parser.add_argument("--profile", default="FULL")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--skip-artifact-check", action="store_true")
    parser.add_argument("--check-handover", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source or default_source()
    job_id = args.job_id or f"smoke-{uuid4().hex[:12]}"
    payload = {
        "jobId": job_id,
        "source": source,
        "customerId": args.customer_id,
        "profile": args.profile,
        "detectionClasses": DETECTION_CLASSES,
    }

    health_status, health, _ = request_json(
        base_url=args.base_url,
        method="GET",
        path="/health",
    )
    if health_status != 200:
        raise RuntimeError(f"healthcheck завершился ошибкой: {health_status} {health}")

    status, created, headers = request_json(
        base_url=args.base_url,
        method="POST",
        path="/api/jobs",
        body=payload,
    )
    if status != 201 or not isinstance(created, dict):
        raise RuntimeError(f"POST /api/jobs вернул ошибку: {status} {created}")
    expected_location = f"/api/jobs/{job_id}"
    if headers.get("Location") != expected_location:
        raise RuntimeError(f"некорректный Location header: {headers.get('Location')}")

    job = wait_for_done(
        base_url=args.base_url,
        job_id=job_id,
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
    )
    result = job.get("result") or {}
    if "totalDetections" not in result:
        raise RuntimeError(f"в завершенной задаче нет result.totalDetections: {job}")

    billing_status, billing, _ = request_json(
        base_url=args.base_url,
        method="GET",
        path=f"/api/billing/{args.customer_id}",
    )
    if billing_status != 200 or not isinstance(billing, dict):
        raise RuntimeError(f"проверка billing завершилась ошибкой: {billing_status} {billing}")

    if not args.skip_artifact_check:
        artifact_status, artifact, _ = request_json(
            base_url=args.base_url,
            method="GET",
            path=f"/jobs/{job_id}/artifacts/time_based_report.json",
        )
        if artifact_status != 200 or not isinstance(artifact, dict):
            raise RuntimeError(
                f"артефакт time_based_report недоступен: {artifact_status} {artifact}"
            )
        if artifact.get("report_type") != "TIME_BASED_REPORT":
            raise RuntimeError(f"некорректный time_based_report artifact: {artifact}")

    if args.check_handover:
        handover_status, _, _ = request_json(
            base_url=args.base_url,
            method="POST",
            path="/api/admin/handover-access",
            body={},
        )
        if handover_status != 204:
            raise RuntimeError(f"проверка handover завершилась ошибкой: {handover_status}")

    delete_status, _, _ = request_json(
        base_url=args.base_url,
        method="DELETE",
        path=f"/api/jobs/{job_id}",
    )
    if delete_status != 204:
        raise RuntimeError(f"DELETE /api/jobs/{job_id} вернул ошибку: {delete_status}")

    print(
        "smoke проверка пройдена: "
        f"jobId={job_id} totalDetections={result['totalDetections']}"
    )


if __name__ == "__main__":
    main()
