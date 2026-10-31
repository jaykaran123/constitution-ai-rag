# syntax=docker/dockerfile:1
# Build from the repository root:  docker compose build   (context: ., file: backend/Dockerfile)
FROM python:3.12-slim


ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=120
# Tesseract is only needed for OCR of image uploads (PDF text extraction does not use it).
ARG INSTALL_OCR=true
RUN if [ "$INSTALL_OCR" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
      && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app/backend
COPY backend/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 -r requirements.txt

COPY backend/app ./app
COPY backend/scripts ./scripts
COPY backend/eval ./eval
COPY backend/data/constitution ./data/constitution
COPY web /app/web

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data/uploads /data/backups ./data/statutes \
 && chown -R appuser /data /app/backend/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
