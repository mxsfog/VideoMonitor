# Анализ видеоданных: ПЗ 1-8 и курсовая работа

Проект подготовлен для курса «Анализ данных» МАИ.

Автор: Дуров Максим, группа М3О-314Б-23.

Тема курсовой работы: выявление признаков деструктивного контента в видео. В
репозитории собраны восемь практических заданий, общий pipeline, REST API,
документация и проверки, необходимые для воспроизводимого запуска.

## Что реализовано

Система принимает видеофайл или ссылку на видео, выполняет обработку по шагам
ПЗ 2-8 и формирует итоговый отчёт. ПЗ 1 вынесено отдельным скриптом, потому что
работает с набором изображений счётчиков.

| Раздел | Документ | Содержание |
|---|---|---|
| Сервис | [docs/service_overview.md](docs/service_overview.md) | устройство REST API, сценарий работы, артефакты |
| ПЗ 1 | [docs/pz1.md](docs/pz1.md) | OpenCV и EasyOCR для распознавания показаний счётчиков |
| ПЗ 2 | [docs/pz2.md](docs/pz2.md) | загрузка и нарезка видео на кадры |
| ПЗ 3 | [docs/pz3.md](docs/pz3.md) | OCR титров и подготовка субтитров |
| ПЗ 4 | [docs/pz4.md](docs/pz4.md) | распознавание речи через faster-whisper |
| ПЗ 5 | [docs/pz5.md](docs/pz5.md) | детекция объектов YOLOv8 |
| ПЗ 6 | [docs/pz6.md](docs/pz6.md) | классификация кадров ResNet50 |
| ПЗ 7 | [docs/pz7.md](docs/pz7.md) | текстовая и визуальная LLM/VLM-классификация |
| ПЗ 8 | [docs/pz8.md](docs/pz8.md) | дедупликация текста и склейка детекций в треки |
| Курсовая | [docs/coursework.md](docs/coursework.md) | общий pipeline и итоговая логика скоринга |
| Контракт | [docs/contract.md](docs/contract.md) | соответствие REST API контракту «Линза.Детектор» |

## Структура

```text
coursework/
├── src/
│   ├── api.py                   # REST API и управление задачами
│   ├── common.py                # рабочие каталоги и логирование
│   ├── contract.py              # сборка JobResult и TIME_BASED_REPORT
│   ├── coursework_pipeline.py   # общий pipeline курсовой работы
│   ├── dto.py                   # Pydantic-модели REST-контракта
│   ├── pz1_counters.py          # ПЗ 1
│   ├── pz2_slicer.py            # ПЗ 2
│   ├── pz3_subtitles.py         # ПЗ 3
│   ├── pz4_whisper.py           # ПЗ 4
│   ├── pz5_yolo.py              # ПЗ 5
│   ├── pz6_resnet.py            # ПЗ 6
│   ├── pz7_llm.py               # ПЗ 7, локальная LLM
│   ├── pz7_openrouter.py        # ПЗ 7, текстовая LLM через OpenRouter
│   ├── pz7_vlm_gemini.py        # ПЗ 7, VLM через OpenRouter
│   └── pz8_postprocess.py       # ПЗ 8
├── docs/                        # описание ПЗ, курсовой, деплоя и контракта
├── tests/                       # проверки контракта, API и VLM-парсинга
├── scripts/                     # сборка релиза, smoke-тест, contract verifier
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── requirements-server.txt
```

Каталоги `data/`, `models/`, `output/`, `dist/`, `.venv/` и локальные `.env`
файлы не входят в публикацию. Они создаются при запуске и могут содержать
приватные данные, модели или тяжёлые артефакты.

Исходные видео, изображения и результаты выполнения намеренно не публикуются.
Для проверки достаточно передать любой локальный файл `video.mp4` через CLI или
REST API.

## Основной сценарий

```text
видео
  │
  ├─ ПЗ 2: кадры
  ├─ ПЗ 3: OCR титров
  ├─ ПЗ 4: транскрипт речи
  ├─ ПЗ 5: объекты YOLO
  ├─ ПЗ 6: классы сцен ResNet
  ├─ ПЗ 7: LLM/VLM-классификация
  └─ ПЗ 8: постобработка
        │
        └─ курсовой агрегатор:
           report.md, findings.json, job_result.json, time_based_report.json
```

Итоговый результат сохраняется в:

```text
output/coursework/<video>/
```

Ключевые файлы:

- `report.md` — человекочитаемый отчёт;
- `findings.json` — внутренние признаки и сводки;
- `job_result.json` — результат в формате REST-контракта;
- `time_based_report.json` — отчёт по временным интервалам.

## Установка для локального запуска

Команды ниже рассчитаны на Windows PowerShell из каталога проекта.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pip install --index-url `
  https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
```

Для серверной Docker-сборки используется `requirements-server.txt`. Она
рассчитана на CPU и не требует GPU.

## Переменные окружения

Безопасный шаблон находится в `.env.example`. В репозиторий нельзя добавлять
реальные ключи, пароли и токены.

Основные параметры:

| Переменная | Назначение |
|---|---|
| `COURSEWORK_LLM_BACKEND` | `none`, `ollama`, `openrouter` или `vlm` |
| `COURSEWORK_LLM_MODEL` | модель для выбранного backend |
| `OPENROUTER_API_KEY` | ключ OpenRouter, нужен только для внешней LLM/VLM |
| `COURSEWORK_PIPELINE_FPS` | частота нарезки видео на сервере |
| `COURSEWORK_VLM_EVERY_N` | прореживание кадров перед VLM |
| `COURSEWORK_SMB_MOUNT_ROOT` | корень заранее смонтированного SMB-каталога |

По умолчанию сервер не отправляет данные во внешние LLM API.

## Запуск отдельных ПЗ

```bash
python src/pz1_counters.py
python src/pz2_slicer.py data/videos/example.mp4 --fps 1
python src/pz3_subtitles.py --frames output/pz2/example --src-fps 1
python src/pz4_whisper.py data/videos/example.mp4 --model tiny --lang ru
python src/pz5_yolo.py data/videos/example.mp4 --every-n 5
python src/pz6_resnet.py output/pz2/example
python run_pz7_openrouter.py
python src/pz8_postprocess.py --out-name example --subs output/pz3/example/subtitles.json --detections output/pz5/example/detections.jsonl
```

## Запуск курсовой

```bash
python src/coursework_pipeline.py data/videos/example.mp4 \
  --fps 1 \
  --whisper-model tiny \
  --lang ru \
  --llm-backend none
```

Полный запуск с VLM:

```bash
python src/coursework_pipeline.py data/videos/example.mp4 \
  --fps 1 \
  --whisper-model tiny \
  --lang ru \
  --llm-backend vlm \
  --llm-model google/gemini-2.5-flash \
  --vlm-every-n 10
```

Если шаги уже рассчитаны, отчёт можно пересобрать без повторной обработки:

```bash
python src/coursework_pipeline.py data/videos/example.mp4 \
  --skip pz2,pz3,pz4,pz5,pz6,pz7,pz8
```

## REST API

API реализован на FastAPI. Контрактные маршруты:

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | проверка доступности сервиса |
| `GET` | `/api/jobs` | список задач |
| `POST` | `/api/jobs` | создание задачи |
| `GET` | `/api/jobs/{jobId}` | состояние и результат задачи |
| `DELETE` | `/api/jobs/{jobId}` | остановка или удаление задачи из списка |
| `GET` | `/api/billing/{customerId}` | локальная информация по учебному тарифу |
| `POST` | `/api/admin/handover-access` | административный endpoint, выключен без явной настройки |

Совместимые учебные маршруты `/process`, `/upload`, `/jobs/*` оставлены для
быстрой проверки и просмотра отчётов.

Локальный запуск API:

```bash
COURSEWORK_LLM_BACKEND=none \
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

## Docker

```bash
docker compose up -d --build
docker compose logs -f api
```

Сервис слушает порт `8000`.

## Проверки

```bash
python3 -m ruff check src tests
python3 -m pytest -q
python3 scripts/verify_contract.py
python3 scripts/build_release.py --check
```

Быстрая проверка запущенного API:

```bash
python3 scripts/smoke_api.py --base-url http://127.0.0.1:8000
```
