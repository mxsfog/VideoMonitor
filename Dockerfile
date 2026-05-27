# CPU-only сборка для запуска на любом VPS.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/hf \
    TORCH_HOME=/app/models/torch \
    EASYOCR_MODULE_PATH=/app/models/easyocr \
    YOLO_CONFIG_DIR=/app/models/ultralytics

# ffmpeg для извлечения аудио и yt-dlp + curl для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch для CPU ставим отдельно
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

COPY requirements-server.txt /app/
# torch уже стоит — убираем дубли
RUN grep -vE '^torch(vision)?==' requirements-server.txt > /tmp/req-rest.txt \
    && pip install -r /tmp/req-rest.txt

# Код
COPY src/ /app/src/
COPY docs/ /app/docs/
COPY scripts/ /app/scripts/
COPY run_pz7.py run_pz7_openrouter.py run_pz7_vlm.py /app/

# Каталоги под данные и артефакты
RUN mkdir -p /app/data /app/output /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
