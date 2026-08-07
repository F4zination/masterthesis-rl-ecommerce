"""Application configuration.

Reads settings from environment variables with sensible defaults for
local development.  Override ``DATABASE_PATH`` and ``SECRET_KEY`` in
production via environment variables or a Docker secrets manager.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "demosite_test.db"))
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
APP_NAME = "DemoSite - Phase 2 Contextual Bandits"
DEBUG = True
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
LEGAL_BASE_URL = os.environ.get("LEGAL_BASE_URL", "").rstrip("/")

# Shared researcher authentication. The dispatcher issues the signed cookie;
# this service only verifies it before serving analytics data.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip()
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
ADMIN_SESSION_SECRET = (
    os.environ.get("ADMIN_SESSION_SECRET", "").strip() or ADMIN_TOKEN
)
ADMIN_COOKIE_NAME = os.environ.get(
    "ADMIN_COOKIE_NAME", "study_admin_session"
).strip()
ADMIN_LOGIN_URL = os.environ.get("ADMIN_LOGIN_URL", "").strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# When FREEZE_POLICY=true the bandit arm statistics are NOT updated during the
# Clickworker experiment so that both V2 and V3 run as frozen policies.
FREEZE_POLICY = _env_bool("FREEZE_POLICY", False)

# Study deployments fail before schema creation/seed if the mounted database
# does not already contain a trained contextual-bandit policy. Local
# development keeps the historical create-and-seed behavior by default.
REQUIRE_BANDIT_POLICY = _env_bool("REQUIRE_BANDIT_POLICY", False)
