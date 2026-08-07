# DemoSiteV3 — Sequential RL / PPO shop (Clickworker experiment, condition V3)
#
# IMPORTANT: build from the repository root, not from DemoSiteV3/, because:
#   1. requirements.txt installs the shared package via `-e ../SharedSchema`.
#   2. config.py resolves the frozen policy artifacts at
#      PROJECT_ROOT/OfflineTraining/outputs (ppo_policy.pt, trained_policy.json),
#      which must be baked into the image for the frozen-policy A/B test.
# Both live outside DemoSiteV3/.
#
#   docker build -f DemoSiteV3/Dockerfile -t demosite-v3 .
#   (or simply `make docker-build-v3`)

FROM python:3.12-slim

# ── OS dependencies ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Shared editable dependency (referenced as `-e ../SharedSchema`) ────────────
COPY SharedSchema /app/SharedSchema

# ── Frozen pre-trained policies ────────────────────────────────────────────────
# config.py uses PROJECT_ROOT=/app, so it looks for the PPO checkpoint and the
# offline tabular policy under /app/OfflineTraining/outputs.
COPY OfflineTraining/outputs /app/OfflineTraining/outputs

# ── Application code ───────────────────────────────────────────────────────────
COPY DemoSiteV3 /app/DemoSiteV3

# ── Python dependencies ────────────────────────────────────────────────────────
# Run from the app directory so the relative `-e ../SharedSchema` resolves to
# /app/SharedSchema.
WORKDIR /app/DemoSiteV3
RUN pip install --no-cache-dir -r requirements.txt

# ── Persistent volume for SQLite database ─────────────────────────────────────
VOLUME ["/app/data"]
ENV DATABASE_PATH=/app/data/demosite.db

# ── Runtime ────────────────────────────────────────────────────────────────────
# Frozen-policy experiment env (FREEZE_POLICY / LEARNER_ENABLED / POLICY_MODE)
# is supplied at run time — see the docker-run-v3 target in the Makefile.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
