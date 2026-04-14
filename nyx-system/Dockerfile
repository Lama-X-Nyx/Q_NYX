FROM python:3.11-slim

# OS deps minimal (curl for HEALTHCHECK)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer)
COPY nyx-system/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App source
COPY nyx-system /app/nyx-system
ENV PYTHONPATH=/app/nyx-system \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

WORKDIR /app/nyx-system

# Persistent volumes for state + logs + decision DB
VOLUME ["/app/nyx-system/storage"]

# Liveness: strategy heartbeat must be < 5 min old
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "from src.paper_live.heartbeat import Heartbeat; \
                 import sys; \
                 sys.exit(0 if Heartbeat('/app/nyx-system/storage/heartbeat.json').is_alive(300) else 1)"

# Default command: run paper-live driver
CMD ["python", "-m", "src.paper_live.main"]
