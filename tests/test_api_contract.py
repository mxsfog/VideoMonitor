from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

import api


def test_missing_job_raises_contract_404() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api.get_api_job("does-not-exist")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Задача не найдена."


def test_error_response_uses_contract_shape() -> None:
    response = api.error_response("bad request", 400)

    assert response.status_code == 400
    assert response.body == b'{"error":{"message":"bad request"}}'


def test_billing_contract_shape() -> None:
    payload = api.get_billing("customer-a")

    assert payload["customerId"] == "customer-a"
    assert payload["balance"]["currency"] == "RUB"
    assert "spent" in payload
    assert "remaining" in payload


def test_artifact_mapping_includes_contract_outputs() -> None:
    mapping = api.artifact_mapping("video-a", {"video_stem": "video-a"})

    assert "job_result.json" in mapping
    assert "time_based_report.json" in mapping
    assert "transcript.json" in mapping


def test_pipeline_command_uses_skip_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COURSEWORK_PIPELINE_SKIP", "pz2,pz3")
    request = api.JobCreateRequest.model_validate(
        {
            "jobId": "job-a",
            "source": "video.mp4",
            "profile": "FULL",
            "detectionClasses": [{"class": "DRUGS"}],
        }
    )

    cmd = api.pipeline_command(
        "job-a",
        request,
        tmp_path / "request.json",
        tmp_path / "state.json",
    )

    assert "--skip" in cmd
    assert cmd[cmd.index("--skip") + 1] == "pz2,pz3"


def test_pipeline_command_can_use_prepared_local_source(tmp_path) -> None:
    request = api.JobCreateRequest.model_validate(
        {
            "jobId": "job-a",
            "source": "https://example.test/video.mp4",
            "profile": "FULL",
            "detectionClasses": [{"class": "DRUGS"}],
        }
    )

    cmd = api.pipeline_command(
        "job-a",
        request,
        tmp_path / "request.json",
        tmp_path / "state.json",
        source=str(tmp_path / "source.mp4"),
    )

    assert cmd[2] == str(tmp_path / "source.mp4")


def test_job_id_rejects_path_unsafe_values() -> None:
    with pytest.raises(ValueError, match="jobId"):
        api.JobCreateRequest.model_validate(
            {
                "jobId": "../bad",
                "source": "video.mp4",
                "profile": "FULL",
                "detectionClasses": [{"class": "DRUGS"}],
            }
        )


def test_source_credentials_are_not_persisted() -> None:
    request = api.JobCreateRequest.model_validate(
        {
            "jobId": "job-a",
            "source": "smb://server/share/video.mp4",
            "sourceCredentials": {
                "login": "user",
                "password": "secret",
                "domain": "DOMAIN",
            },
            "profile": "FULL",
            "detectionClasses": [{"class": "DRUGS"}],
        }
    )

    stored = api.sanitized_job_request(request).model_dump(by_alias=True)

    assert "sourceCredentials" not in stored


def test_smb_source_resolves_through_mount_root(monkeypatch, tmp_path) -> None:
    source = tmp_path / "server" / "share" / "dir" / "video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.setenv("COURSEWORK_SMB_MOUNT_ROOT", str(tmp_path))

    resolved = api.prepare_pipeline_source("job-a", "smb://server/share/dir/video.mp4")

    assert resolved == str(source)


def test_download_remote_source_uses_job_directory(monkeypatch, tmp_path) -> None:
    class DummyResponse:
        headers = {"Content-Length": "5"}

        def __init__(self) -> None:
            self._chunks = [b"abc", b"de", b""]

        def __enter__(self) -> DummyResponse:
            return self

        def __exit__(self, *_) -> None:
            return None

        def read(self, _) -> bytes:
            return self._chunks.pop(0)

    monkeypatch.setattr(api, "job_dir", lambda _: tmp_path)
    monkeypatch.setattr(api, "urlopen", lambda *_args, **_kwargs: DummyResponse())

    target = api.download_remote_source("job-a", "https://example.test/path/video.mp4")

    assert target == tmp_path / "source.mp4"
    assert target.read_bytes() == b"abcde"


def test_profile_is_case_insensitive() -> None:
    request = api.JobCreateRequest.model_validate(
        {
            "jobId": "job-a",
            "source": "video.mp4",
            "profile": "preview",
            "detectionClasses": [{"class": "DRUGS"}],
        }
    )

    assert request.profile == "PREVIEW"


def test_create_job_response_sets_location_header(monkeypatch) -> None:
    request = api.JobCreateRequest.model_validate(
        {
            "jobId": "job-a",
            "source": "video.mp4",
            "profile": "FULL",
            "detectionClasses": [{"class": "DRUGS"}],
        }
    )

    monkeypatch.setattr(api, "submit_job", lambda _: {"jobId": "job-a"})

    response = api.create_api_job(request)

    assert response.status_code == 201
    assert response.headers["location"] == "/api/jobs/job-a"


def test_handover_access_runs_configured_command(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    monkeypatch.setenv("COURSEWORK_HANDOVER_COMMAND", "true")
    monkeypatch.setattr(api.subprocess, "run", fake_run)

    response = api.handover_access()

    assert response.status_code == 204
    assert calls[0][0][0] == ["true"]
    assert calls[0][1]["check"] is True


def test_load_jobs_marks_unfinished_jobs_as_error(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "job-a"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "jobId": "job-a",
                "status": "IN_PROGRESS",
                "createdAt": "2026-05-25T00:00:00Z",
                "startedAt": "2026-05-25T00:01:00Z",
                "request": {
                    "source": "video.mp4",
                    "profile": "FULL",
                    "detectionClasses": [{"class": "DRUGS"}],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path)
    with api._jobs_lock:
        api._jobs.clear()

    api.load_jobs_from_disk()

    record = api.get_record("job-a")
    assert record["status"] == "ERROR"
    assert record["error"]["message"] == "сервер был перезапущен до завершения задачи"


def test_openapi_documents_create_job_location_header() -> None:
    api.app.openapi_schema = None

    response_schema = api.app.openapi()["paths"]["/api/jobs"]["post"]["responses"]["201"]

    assert "Location" in response_schema["headers"]


def test_openapi_contract_paths_match_bundled_spec() -> None:
    api.app.openapi_schema = None
    bundled = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs" / "linza.detector-rest-api.yml").read_text(
            encoding="utf-8"
        )
    )
    actual = api.app.openapi()

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

    assert actual_paths == expected_paths


def test_openapi_contract_components_match_bundled_spec() -> None:
    api.app.openapi_schema = None
    bundled = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs" / "linza.detector-rest-api.yml").read_text(
            encoding="utf-8"
        )
    )
    actual = api.app.openapi()

    assert actual["components"]["schemas"] == bundled["components"]["schemas"]
    assert actual["components"]["responses"] == bundled["components"]["responses"]


def test_dockerfile_copies_docs_for_canonical_openapi() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY docs/ /app/docs/" in dockerfile
    assert "COPY scripts/ /app/scripts/" in dockerfile


def test_dockerignore_excludes_private_and_heavy_artifacts() -> None:
    dockerignore = (Path(__file__).resolve().parents[1] / ".dockerignore").read_text(
        encoding="utf-8"
    ).splitlines()

    for ignored in ("data/", "output/", "models/", ".venv/", ".env", "smb/"):
        assert ignored in dockerignore


def test_compose_exposes_contract_runtime_options() -> None:
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    service = compose["services"]["api"]

    assert "8000:8000" in service["ports"]
    assert "COURSEWORK_SMB_MOUNT_ROOT" in service["environment"]
    assert "COURSEWORK_HANDOVER_COMMAND" in service["environment"]
    assert "COURSEWORK_VLM_EVERY_N" in service["environment"]
    assert "OPENROUTER_TIMEOUT_SECONDS" in service["environment"]
    assert "OPENROUTER_RETRIES" in service["environment"]
    assert "./smb:/app/smb:ro" in service["volumes"]
