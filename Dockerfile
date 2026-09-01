# ollama-rag-kit API container.
#
# python:3.12-slim because the image is the contract: the host can run
# 3.13 or anything else, the service always ships on a pinned, tested
# interpreter. slim keeps the pull small; the app has no system-level
# build dependencies.

FROM python:3.14-slim

# No .pyc litter, unbuffered logs so docker logs streams in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Requirements first: this layer only rebuilds when dependencies change,
# so day-to-day code edits reuse the cached pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user. The service needs no privileges, so it gets
# none; a container escape from uid 1000 is a far smaller event than one
# from root.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
