# DemoSiteV2 — Contextual Bandit shop (Clickworker experiment, condition V2)
#
# IMPORTANT: build from the repository root, not from DemoSiteV2/, because
# requirements.txt installs the shared package via `-e ../SharedSchema`, which
# lives outside this directory.
#
#   docker build -f DemoSiteV2/Dockerfile -t demosite-v2 .
#   (or simply `make docker-build-v2`)

FROM python:3.12-slim

# ── OS dependencies ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Shared editable dependency (referenced as `-e ../SharedSchema`) ────────────
COPY SharedSchema /app/SharedSchema

# ── Application code ───────────────────────────────────────────────────────────
COPY DemoSiteV2 /app/DemoSiteV2

# ── Python dependencies ────────────────────────────────────────────────────────
# Run from the app directory so the relative `-e ../SharedSchema` resolves to
# /app/SharedSchema.
WORKDIR /app/DemoSiteV2
RUN pip install --no-cache-dir -r requirements.txt

# ── Persistent volume for SQLite database ─────────────────────────────────────
# Mount a pre-trained demosite.db here for the frozen-policy experiment so the
# bandit serves trained arm statistics; otherwise an empty DB is seeded on start.
VOLUME ["/app/data"]
ENV DATABASE_PATH=/app/data/demosite.db

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
