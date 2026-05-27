# Контракт и проверка требований

Документ фиксирует, какие требования закрыты проектом и где находится
реализация. Он нужен для защиты и для технической проверки перед публикацией.

## Источники требований

- перечень практических заданий ПЗ 1-8;
- формулировка курсовой про единый сценарий обработки видео;
- OpenAPI-спецификация `REST API Линза.Детектор`;
- пример отчёта `TIME_BASED_REPORT`;
- требование развернуть рабочий сервис на VM.

Каноническая OpenAPI-спецификация хранится в:

```text
docs/linza.detector-rest-api.yml
```

FastAPI отдаёт эти схемы для контрактных маршрутов `/api/*`.

## Практические задания

| Требование | Реализация | Основной результат |
|---|---|---|
| ПЗ 1: OpenCV и OCR изображений | `src/pz1_counters.py` | `output/pz1/recognized.xlsx` |
| ПЗ 2: нарезка видео | `src/pz2_slicer.py` | `output/pz2/<video>/frame_*.jpg` |
| ПЗ 3: OCR титров | `src/pz3_subtitles.py` | `output/pz3/<video>/subtitles.srt` |
| ПЗ 4: Whisper | `src/pz4_whisper.py` | `output/pz4/<video>/transcript.txt` |
| ПЗ 5: YOLO | `src/pz5_yolo.py` | `output/pz5/<video>/detections.jsonl` |
| ПЗ 6: ResNet | `src/pz6_resnet.py` | `output/pz6/<video>/predictions.jsonl` |
| ПЗ 7: LLM/VLM | `src/pz7_llm.py`, `src/pz7_openrouter.py`, `src/pz7_vlm_gemini.py` | `output/pz7/<video>/summary.json` |
| ПЗ 8: постобработка | `src/pz8_postprocess.py` | `output/pz8/<video>/tracks.json` |
| Курсовая | `src/coursework_pipeline.py` | `output/coursework/<video>/report.md` |

## REST API

| Метод | Маршрут | Статус |
|---|---|---|
| `GET` | `/api/jobs` | реализовано |
| `POST` | `/api/jobs` | реализовано |
| `GET` | `/api/jobs/{jobId}` | реализовано |
| `DELETE` | `/api/jobs/{jobId}` | реализовано |
| `GET` | `/api/billing/{customerId}` | реализовано как локальная учебная информация |
| `POST` | `/api/admin/handover-access` | реализовано, выключено без явной настройки |

Поддержанные статусы задач:

```text
PENDING, IN_PROGRESS, DONE, ERROR
```

## Выходные артефакты

| Файл | Назначение |
|---|---|
| `report.md` | текстовый отчёт для защиты |
| `findings.json` | внутренние признаки и статистика |
| `job_result.json` | результат в формате `JobResult` |
| `time_based_report.json` | временные интервалы в формате примера |

`JobResult` содержит:

- `processingDurationSeconds`;
- `sourceInfo`;
- `totalDetections`;
- `detectionClassStatistics`;
- `detections`.

## Ограничения по безопасности

- `sourceCredentials` принимаются по контракту, но не сохраняются в `state.json`.
- `smb://` работает только через заранее смонтированный каталог
  `COURSEWORK_SMB_MOUNT_ROOT`.
- Внешние LLM/VLM API выключены по умолчанию.
- `.gitignore` и `.dockerignore` исключают `data/`, `models/`, `output/`,
  `.venv/`, `.env*`.
- `COURSEWORK_HANDOVER_COMMAND` должен указывать только на заранее проверенный
  локальный скрипт на VM.

## Проверки

Перед публикацией должны проходить:

```bash
python3 -m ruff check src tests
python3 -m pytest -q
python3 scripts/verify_contract.py
python3 scripts/build_release.py --check
```

Быстрая проверка работающего сервера:

```bash
python3 scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

Проверенный пользовательский сценарий:

1. создать задачу через `POST /api/jobs`;
2. дождаться `DONE`;
3. открыть `job_result.json`;
4. открыть `time_based_report.json`;
5. удалить задачу через `DELETE /api/jobs/{jobId}`, если она больше не нужна.

## Статус

Код, документация, Docker-сборка и проверки подготовлены. Публикация на GitHub и
доступность конкретной VM проверяются отдельно, потому что зависят от внешней
инфраструктуры.
