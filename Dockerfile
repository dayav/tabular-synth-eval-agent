# syntax=docker/dockerfile:1
# Multi-stage build: dependencies compile in the builder, the runtime image ships only the venv and src/.

FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.lock

FROM python:3.12-slim AS runtime
ARG APP_VERSION=0.3.0
# libgomp1: OpenMP runtime that the xgboost wheel links against
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUNS_DIR=/data/runs \
    APP_VERSION=${APP_VERSION} \
    PORT=8000
RUN mkdir -p /data/runs && chown -R app:app /data /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4).status == 200 else 1)"
# ANTHROPIC_API_KEY is read from the environment at request time; never bake it into the image.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
