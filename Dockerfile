# CodeQuest judge worker — consumes submission jobs and runs them in Docker.
#
# Build:  docker build -t codequest-worker judge
#
# Runs as root on purpose: it needs the host's Docker socket to start the
# language containers that execute untrusted user code. The language images
# (codequest-python/cpp/java/js) must exist on the daemon the socket points at;
# infra/docker-compose.prod.yml builds them from judge/languages/*.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD ["python", "-c", "import docker; docker.from_env().ping()"]

# Root: the Docker socket is group-controlled by the host (typically gid 999
# or 998); container group mappings for it differ per host, which root avoids.
CMD ["python", "worker.py"]
