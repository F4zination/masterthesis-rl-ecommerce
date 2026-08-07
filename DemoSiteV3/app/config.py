"""Application configuration.

Reads settings from environment variables with sensible defaults for
local development.  Override ``DATABASE_PATH`` and ``SECRET_KEY`` in
production via environment variables or a Docker secrets manager.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_db_path = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "demosite_test.db"))
DATABASE_URL = f"sqlite:///{_db_path}"
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


PROJECT_ROOT = os.path.dirname(BASE_DIR)
OFFLINE_TRAINING_DIR = os.path.join(PROJECT_ROOT, "OfflineTraining")
OFFLINE_OUTPUT_DIR = os.path.join(OFFLINE_TRAINING_DIR, "outputs")
SIMULATION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "CustomerSimulation", "output")

POLICY_MODE = os.environ.get("POLICY_MODE", "ppo_first").strip().lower()
# Study deployments can require the PPO artifact to load successfully instead
# of silently changing the treatment to an offline-policy or bandit fallback.
REQUIRE_PPO_CHECKPOINT = _env_bool("REQUIRE_PPO_CHECKPOINT", False)
OFFLINE_POLICY_PATH = os.environ.get(
	"OFFLINE_POLICY_PATH",
	os.path.join(OFFLINE_OUTPUT_DIR, "trained_policy.json"),
)
ACTIVE_POLICY_POINTER_PATH = os.environ.get(
	"ACTIVE_POLICY_POINTER_PATH",
	os.path.join(OFFLINE_OUTPUT_DIR, "active_version.txt"),
)

TIMING_ENABLED = _env_bool("TIMING_ENABLED", True)
MAX_OPPORTUNITIES_PER_SESSION = int(os.environ.get("MAX_OPPORTUNITIES_PER_SESSION", "30"))
MIN_DECISION_COOLDOWN_MS = int(os.environ.get("MIN_DECISION_COOLDOWN_MS", "0"))
OFFLINE_POLICY_DEFER_MS_DEFAULT = int(os.environ.get("OFFLINE_POLICY_DEFER_MS_DEFAULT", "4000"))
ANALYTICS_TIMELINE_LOOKBACK_HOURS = max(
	1,
	int(os.environ.get("ANALYTICS_TIMELINE_LOOKBACK_HOURS", "24")),
)

LEARNER_ENABLED = _env_bool("LEARNER_ENABLED", True)
LEARNER_INTERVAL_SECONDS = int(os.environ.get("LEARNER_INTERVAL_SECONDS", "21600"))
LEARNER_MIN_DECISIONS = int(os.environ.get("LEARNER_MIN_DECISIONS", "50"))
LEARNER_LOOKBACK_HOURS = int(os.environ.get("LEARNER_LOOKBACK_HOURS", "24"))
LEARNER_MAX_TRAIN_MINUTES = int(os.environ.get("LEARNER_MAX_TRAIN_MINUTES", "20"))
LEARNER_TIMING_MODE = os.environ.get("LEARNER_TIMING_MODE", "opportunity").strip().lower()
LEARNER_OUTPUT_DIR = os.environ.get("LEARNER_OUTPUT_DIR", OFFLINE_OUTPUT_DIR)
LEARNER_ALGO = os.environ.get("LEARNER_ALGO", "ppo").strip().lower()

PPO_CHECKPOINT_PATH = os.environ.get(
	"PPO_CHECKPOINT_PATH",
	os.path.join(OFFLINE_OUTPUT_DIR, "ppo_policy.pt"),
)
PPO_PRETRAIN_DATASET_PATH = os.environ.get(
	"PPO_PRETRAIN_DATASET_PATH",
	os.path.join(SIMULATION_OUTPUT_DIR, "offline_transitions.jsonl"),
)
PPO_HIDDEN_SIZES = os.environ.get("PPO_HIDDEN_SIZES", "128,64")
PPO_GAMMA = float(os.environ.get("PPO_GAMMA", "0.95"))
PPO_CLIP_EPS = float(os.environ.get("PPO_CLIP_EPS", "0.2"))
PPO_ENTROPY_COEF = float(os.environ.get("PPO_ENTROPY_COEF", "0.01"))
PPO_VALUE_COEF = float(os.environ.get("PPO_VALUE_COEF", "0.5"))
PPO_LR = float(os.environ.get("PPO_LR", "3e-4"))
PPO_GAE_LAMBDA = float(os.environ.get("PPO_GAE_LAMBDA", "0.95"))
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", "6"))
PPO_MINIBATCH_SIZE = int(os.environ.get("PPO_MINIBATCH_SIZE", "64"))
PPO_MAX_GRAD_NORM = float(os.environ.get("PPO_MAX_GRAD_NORM", "0.5"))

# When FREEZE_POLICY=true the bandit fallback arm statistics are NOT updated
# during the Clickworker experiment so that V3 runs as a frozen policy.
# Also set LEARNER_ENABLED=false to prevent background retraining.
FREEZE_POLICY = _env_bool("FREEZE_POLICY", False)
