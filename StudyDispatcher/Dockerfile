# StudyDispatcher — balanced cell assignment + redirect to V2/V3.
#
# Build from the repository root so the shared admin-session helper is present:
#   docker build -f StudyDispatcher/Dockerfile -t study-dispatcher .
# or via the top-level docker-compose.yml.

FROM python:3.12-slim

WORKDIR /app

COPY SharedSchema /app/SharedSchema
COPY StudyDispatcher/requirements.lock /app/StudyDispatcher/requirements.lock

WORKDIR /app/StudyDispatcher
RUN pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir --no-deps -e ../SharedSchema

COPY StudyDispatcher/app ./app

# Persistent assignment ledger (SQLite). Keep the balance across restarts.
VOLUME ["/app/data"]
ENV DISPATCHER_DB_PATH=/app/data/dispatcher.db

EXPOSE 8000

# Single worker so the in-process assignment lock keeps cells balanced.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
