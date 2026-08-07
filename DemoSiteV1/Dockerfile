FROM python:3.12-slim

# ── OS dependencies ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Persistent volume for SQLite database ─────────────────────────────────────
# demosite.db is created at /app/demosite.db (BASE_DIR = project root inside container)
VOLUME ["/app/data"]

ENV DATABASE_PATH=/app/data/demosite.db

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
