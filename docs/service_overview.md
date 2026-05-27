# Сервис анализа видео

Документ описывает серверную часть проекта: REST API, жизненный цикл задачи,
формат результатов и порядок демонстрации.

## Назначение

Сервис принимает видео, запускает pipeline анализа и сохраняет результат в
виде отчётов. Одна задача соответствует одному видеофайлу.

Основной результат доступен в двух формах:

- человекочитаемый отчёт `report.md`;
- JSON по контракту `JobResult`.

## Состав

| Компонент | Файл | Ответственность |
|---|---|---|
| HTTP API | `src/api.py` | маршруты, валидация запросов, очередь задач |
| DTO | `src/dto.py` | Pydantic-модели запроса и ответа |
| Pipeline | `src/coursework_pipeline.py` | последовательный запуск ПЗ и сбор отчёта |
| Контракт | `src/contract.py` | сборка `JobResult` и `TIME_BASED_REPORT` |
| OpenAPI | `docs/linza.detector-rest-api.yml` | исходная спецификация контрактных маршрутов |
| Docker | `Dockerfile`, `docker-compose.yml` | серверная сборка и запуск |

Состояние задач хранится на диске в `output/jobs/<jobId>/`. После перезапуска
сервиса незавершённые задачи помечаются как ошибочные, чтобы не показывать
устаревший статус `IN_PROGRESS`.

## Обработка задачи

```text
POST /api/jobs
  │
  ├─ проверка source и jobId
  ├─ сохранение request.json
  ├─ запуск pipeline в отдельном worker
  │
  ├─ ПЗ 2: нарезка видео
  ├─ ПЗ 3: OCR титров
  ├─ ПЗ 4: распознавание речи
  ├─ ПЗ 5: YOLO-детекции
  ├─ ПЗ 6: ResNet-классификация
  ├─ ПЗ 7: LLM/VLM-анализ
  ├─ ПЗ 8: постобработка
  │
  └─ DONE или ERROR
```

По умолчанию сервер не отправляет данные во внешние LLM API. Для этого
используется:

```bash
COURSEWORK_LLM_BACKEND=none
```

Внешний OpenRouter/Gemini включается только явной настройкой переменных
окружения.

## Контрактные маршруты

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | проверка доступности |
| `GET` | `/api/jobs` | список задач |
| `POST` | `/api/jobs` | создание задачи |
| `GET` | `/api/jobs/{jobId}` | состояние и результат |
| `DELETE` | `/api/jobs/{jobId}` | остановка или удаление задачи из списка |
| `GET` | `/api/billing/{customerId}` | локальная информация по учебному тарифу |
| `POST` | `/api/admin/handover-access` | административное действие, выключено без настройки |

Дополнительные учебные маршруты:

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/process` | упрощённое создание задачи |
| `POST` | `/upload` | загрузка видео через multipart-form |
| `GET` | `/jobs` | список задач в упрощённом формате |
| `GET` | `/jobs/{job_id}` | состояние задачи |
| `GET` | `/jobs/{job_id}/log` | лог выполнения |
| `GET` | `/jobs/{job_id}/report` | Markdown-отчёт |
| `GET` | `/jobs/{job_id}/artifacts/{name}` | отдельный артефакт |

## Пример запроса

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "jobId": "coursework-001",
    "source": "/app/data/videos/example.mp4",
    "customerId": "coursework",
    "profile": "FULL",
    "detectionClasses": [
      {"class": "DRUGS"},
      {"class": "DEVIANT"},
      {"class": "TERRORISM"}
    ]
  }'
```

Проверка состояния:

```bash
curl http://127.0.0.1:8000/api/jobs/coursework-001
```

Пример успешного ответа:

```json
{
  "jobId": "coursework-001",
  "status": "DONE",
  "createdAt": "2026-05-26T18:55:01Z",
  "startedAt": "2026-05-26T18:55:01Z",
  "finishedAt": "2026-05-26T19:09:40Z",
  "result": {
    "processingDurationSeconds": 877.948223,
    "sourceInfo": {
      "frameCount": 3848,
      "fps": 29.969999,
      "durationSeconds": 128.395067
    },
    "totalDetections": 25,
    "detectionClassStatistics": [],
    "detections": []
  }
}
```

## Артефакты

Рабочий каталог задачи:

```text
output/jobs/<jobId>/
├── request.json
├── log.txt
├── pipeline_state.json
└── state.json
```

Итоги по видео:

```text
output/coursework/<video>/
├── report.md
├── findings.json
├── job_result.json
└── time_based_report.json
```

Артефакты ПЗ:

- `output/pz2/<video>/frame_*.jpg`;
- `output/pz3/<video>/subtitles.json`, `subtitles.srt`;
- `output/pz4/<video>/transcript.json`, `transcript.srt`, `transcript.txt`;
- `output/pz5/<video>/detections.jsonl`, `summary.json`;
- `output/pz6/<video>/predictions.jsonl`, `summary.json`;
- `output/pz7/<video>/classified.jsonl`, `summary.json`;
- `output/pz8/<video>/tracks.json`, `subs_dedup.json`.

## Развёртывание

Рекомендуемый режим для учебной VM:

- Ubuntu 24.04 LTS;
- Docker и Docker Compose;
- 2-4 vCPU, 4-8 GB RAM;
- публичный порт `8000` или nginx-прокси на `80/443`.

Проверка на сервере:

```bash
docker ps
docker logs --tail=80 destructive-api
curl -i http://127.0.0.1:8000/health
```

Перезапуск:

```bash
cd /opt/coursework/app
docker compose restart
```

Полная пересборка:

```bash
cd /opt/coursework/app
docker compose up -d --build
```

## Что показывать

1. Swagger UI: `/docs`.
2. `GET /health`.
3. `GET /api/jobs`.
4. Создание задачи через `/api/jobs` или `/upload`.
5. Статус задачи и поле `result`.
6. `report.md`, `job_result.json`, `time_based_report.json`.
7. Лог задачи через `/jobs/{job_id}/log`, если нужно показать ход pipeline.

## Ограничения

- серверная сборка рассчитана на CPU и будет медленнее локального GPU;
- OCR и YOLO занимают основное время на длинных видео;
- внешние LLM/VLM API выключены до явного разрешения и настройки ключа;
- `sourceCredentials` не сохраняются в состоянии задачи;
- `smb://` работает только при заранее смонтированном каталоге;
- публичный порт без nginx и ограничения доступа годится только для учебного
  показа.
