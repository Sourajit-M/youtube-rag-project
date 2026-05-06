FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.11-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy uv from official image
COPY --from=uv_bin /uv /uvx /bin/

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv 
COPY app/ ./app/
COPY eval/ ./eval/

# Make venv the active Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Render.com sets PORT dynamically — default to 8000 locally
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]