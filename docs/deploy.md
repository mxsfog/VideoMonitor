# Развёртывание на VPS

Документ описывает рабочий порядок запуска REST API на учебной виртуальной
машине, в том числе на VPS Timeweb. Инструкция рассчитана на Ubuntu 24.04 LTS
и Docker Compose.

## Требования

Минимальная конфигурация:

- 2 vCPU;
- 4 GB RAM;
- 30 GB SSD;
- публичный IPv4;
- открытый порт `8000` или nginx-прокси на `80/443`.

Рекомендуемая конфигурация для длинных видео:

- 4 vCPU;
- 8 GB RAM;
- 50 GB SSD.

GPU на сервере не требуется. Docker-сборка рассчитана на CPU.

## Подготовка сервера

```bash
ssh root@<IP>

apt update
apt install -y curl git docker.io docker-compose-plugin

systemctl enable --now docker
docker --version
docker compose version
```

## Доставка проекта

Вариант 1: клонировать опубликованный репозиторий.

```bash
cd /opt
git clone <REPO_URL> coursework
cd /opt/coursework
```

Вариант 2: передать подготовленный release-архив.

Локально:

```bash
python3 scripts/build_release.py --check
python3 scripts/build_release.py
```

На сервере:

```bash
mkdir -p /opt/coursework/app
cd /opt/coursework/app
unzip /path/to/coursework_release.zip
```

В архив не включаются `data/`, `models/`, `output/`, `.venv/`, `.env` и другие
локальные артефакты.

## Настройка окружения

Создать `.env` рядом с `docker-compose.yml`.

Базовый безопасный режим:

```bash
cat > .env <<'EOF'
COURSEWORK_LLM_BACKEND=none
COURSEWORK_LLM_MODEL=qwen3.5:9b
COURSEWORK_PIPELINE_FPS=1.0
COURSEWORK_VLM_EVERY_N=10
EOF

chmod 600 .env
```

Режим с VLM через OpenRouter:

```bash
cat >> .env <<'EOF'
COURSEWORK_LLM_BACKEND=vlm
COURSEWORK_LLM_MODEL=google/gemini-2.5-flash
OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
OPENROUTER_TIMEOUT_SECONDS=60
OPENROUTER_RETRIES=1
EOF
```

Ключи и токены нельзя коммитить в репозиторий.

## Запуск

```bash
docker compose up -d --build
docker ps
docker logs --tail=80 destructive-api
```

Проверка изнутри сервера:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/docs
```

Проверка снаружи:

```bash
curl -i http://<IP>:8000/health
```

## Создание задачи

```bash
curl -X POST http://<IP>:8000/api/jobs \
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

Проверка:

```bash
curl http://<IP>:8000/api/jobs/coursework-001
```

Отчёт:

```bash
curl http://<IP>:8000/jobs/coursework-001/report
curl http://<IP>:8000/jobs/coursework-001/artifacts/job_result.json
curl http://<IP>:8000/jobs/coursework-001/artifacts/time_based_report.json
```

## Загрузка файла через API

```bash
curl -X POST http://<IP>:8000/upload \
  -F "file=@video.mp4" \
  -F "fps=1.0" \
  -F "whisper_model=tiny" \
  -F "llm_backend=none"
```

Ответ содержит `jobId`. Далее статус проверяется через `/api/jobs/{jobId}` или
`/jobs/{jobId}`.

## Nginx

Для показа через обычный HTTP-порт лучше поставить reverse proxy.

```bash
apt install -y nginx
```

Пример `/etc/nginx/sites-available/coursework`:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/coursework /etc/nginx/sites-enabled/coursework
nginx -t
systemctl reload nginx
```

После этого Swagger UI будет доступен по адресу:

```text
http://<IP>/docs
```

## Обновление

```bash
cd /opt/coursework/app
git pull
docker compose up -d --build
```

Если проект передан архивом, распаковать новую версию в тот же каталог и снова
выполнить:

```bash
docker compose up -d --build
```

## Логи

```bash
docker logs --tail=120 destructive-api
docker logs -f destructive-api
```

Лог конкретной задачи:

```bash
curl http://<IP>:8000/jobs/<jobId>/log
```

## Остановка

```bash
docker compose down
```

Удаление volumes использовать только если точно не нужны локальные данные:

```bash
docker compose down -v
```

## Производительность

На CPU основное время занимают OCR и YOLO. Для короткого ролика в 2-4 минуты
ожидаемое время обработки может быть от нескольких минут до 15 минут в
зависимости от частоты кадров и выбранных моделей.

Что ускоряет обработку:

- `COURSEWORK_PIPELINE_FPS=0.5` или `1.0`;
- Whisper `tiny`;
- увеличение `COURSEWORK_VLM_EVERY_N`;
- пропуск уже рассчитанных шагов через `--skip` при локальной пересборке
  отчёта.

## Безопасность

Для учебной VM допустим упрощённый режим, но базовые ограничения нужно
соблюдать:

- не публиковать `.env`;
- не печатать ключи в отчётах и логах;
- не отправлять приватные видео во внешние API без разрешения;
- не оставлять публичный порт без контроля дольше, чем нужно для показа;
- периодически чистить `data/uploads/` и `output/jobs/`;
- не включать `COURSEWORK_HANDOVER_COMMAND` без ручной проверки скрипта.

## Чек-лист перед показом

- контейнер `destructive-api` запущен;
- `/health` отвечает `200`;
- `/docs` открывается;
- создана хотя бы одна задача со статусом `DONE`;
- доступны `report.md`, `job_result.json`, `time_based_report.json`;
- выбранный режим LLM указан явно;
- секреты лежат только в `.env`;
- тяжёлые `data/`, `models/`, `output/` не попали в репозиторий.
