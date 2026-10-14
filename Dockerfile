# One image for both services; docker-compose.yml picks the command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel bundles CUDA (~2 GB) we don't need.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged runtime user
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser data/samples ./data/samples
COPY --chown=appuser:appuser data/eval ./data/eval
COPY --chown=appuser:appuser pyproject.toml ./

RUN mkdir -p /app/storage /app/.cache/huggingface \
    && chown -R appuser:appuser /app/storage /app/.cache

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
