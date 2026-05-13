FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser

WORKDIR /app

# Install Python deps before copying app code so layer is cached on dep changes
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Model weights and config are always volume-mounted — never baked into the image
# docker run -v /opt/models:/models:ro -v /etc/inference-server:/etc/inference-server:ro

USER appuser

EXPOSE 8000

ENTRYPOINT ["python3.12", "-m", "uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
