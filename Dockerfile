FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY app/ ./app/

USER appuser

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
