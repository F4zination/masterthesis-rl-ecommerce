from contextlib import asynccontextmanager
import logging
from pathlib import Path
import secrets
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from shared_schema.admin_session import verify_admin_session

from . import config
from .config import (
    APP_NAME,
    DEBUG,
    LEARNER_ENABLED,
    LEARNER_INTERVAL_SECONDS,
    POLICY_MODE,
    REQUIRE_PPO_CHECKPOINT,
)
from .database import init_db
from .seed import seed_database, reconcile_product_images
from .routers import shop, api, events, decision, analytics
from .services.decision import get_policy_health_status
from .services.learner import BackgroundLearner

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("demosite.startup")


def _enforce_policy_requirements(health: dict) -> None:
    """Fail closed when a study deployment requires the PPO treatment."""
    if not REQUIRE_PPO_CHECKPOINT:
        return
    if POLICY_MODE not in {"ppo_first", "ppo_only"}:
        raise RuntimeError(
            "REQUIRE_PPO_CHECKPOINT=true requires POLICY_MODE=ppo_first or ppo_only"
        )
    if not health.get("ppo_checkpoint_valid"):
        raise RuntimeError(
            "Required PPO checkpoint is missing or invalid: "
            f"{health.get('ppo_checkpoint_path') or '<unset>'}"
        )
    if not health.get("ppo_checkpoint_study_contract_valid"):
        raise RuntimeError(
            "Required PPO checkpoint violates the online sequential study "
            "contract: "
            f"{health.get('ppo_checkpoint_study_contract_error') or '<unspecified>'}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup, initialises the database schema and seeds it with default
    categories and products if the database is empty.  The ``yield``
    hands control back to FastAPI to serve requests; any teardown logic
    would go after it.

    Args:
        app: The FastAPI application instance (provided by the framework).
    """
    # Validate the frozen treatment before making even routine database writes.
    health = get_policy_health_status()
    _enforce_policy_requirements(health)

    init_db()
    seeded = seed_database()
    if seeded:
        logger.info("Database seeded with categories and products.")
    else:
        logger.info("Database already populated, skipping seed.")
        # Seeding is skipped for an existing database, so realign stored
        # image paths with the repository in case an asset was renamed.
        repointed = reconcile_product_images()
        if repointed:
            logger.info("Re-pointed %d product image path(s).", repointed)

    learner: BackgroundLearner | None = None

    logger.info(
        "Policy runtime: mode=%s timing_enabled=%s cooldown_ms=%s cap_per_session=%s",
        health.get("policy_mode"),
        health.get("timing_enabled"),
        health.get("min_decision_cooldown_ms"),
        health.get("max_opportunities_per_session"),
    )

    if health.get("offline_policy_valid"):
        logger.info(
            "Offline policy loaded: path=%s states=%s defaults=%s",
            health.get("policy_path"),
            health.get("policy_states", 0),
            health.get("default_points", 0),
        )
    else:
        logger.warning(
            "Offline policy unavailable or invalid: path=%s reason=%s",
            health.get("policy_path"),
            health.get("reason"),
        )

    if health.get("ppo_checkpoint_valid"):
        logger.info(
            "PPO checkpoint loaded: path=%s states=%s",
            health.get("ppo_checkpoint_path"),
            health.get("ppo_summary", {}).get("state_vocab_size", 0),
        )
    else:
        logger.info(
            "PPO checkpoint unavailable or invalid: path=%s",
            health.get("ppo_checkpoint_path"),
        )

    if health.get("policy_mode") == "offline_first" and not health.get("offline_policy_valid"):
        logger.warning("Fallback active: offline_first will use contextual bandit until policy is valid.")
    if health.get("policy_mode") == "offline_only" and not health.get("offline_policy_valid"):
        logger.warning("Fallback active: offline_only will return no-op until policy is valid.")

    if LEARNER_ENABLED:
        learner = BackgroundLearner(interval_seconds=LEARNER_INTERVAL_SECONDS)
        learner.start()
        logger.info("Background learner started: interval_seconds=%s", LEARNER_INTERVAL_SECONDS)
    else:
        logger.info("Background learner disabled")

    yield

    if learner is not None:
        learner.stop()
        logger.info("Background learner stopped")


app = FastAPI(title=APP_NAME, debug=DEBUG, lifespan=lifespan)


def _analytics_authorized(request: Request) -> bool:
    cookie_ok = verify_admin_session(
        request.cookies.get(config.ADMIN_COOKIE_NAME, ""),
        config.ADMIN_SESSION_SECRET,
        expected_username=config.ADMIN_USERNAME,
    )
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    bearer_ok = bool(
        config.ADMIN_TOKEN
        and scheme.lower() == "bearer"
        and secrets.compare_digest(token, config.ADMIN_TOKEN)
    )
    return cookie_ok or bearer_ok


def _original_request_url(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0]
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    path = request.url.path
    if request.url.query:
        path += "?" + request.url.query
    return f"{scheme}://{host}{path}"


@app.middleware("http")
async def protect_analytics(request: Request, call_next):
    """Require the shared researcher session for analytics UI and APIs."""
    path = request.url.path.rstrip("/") or "/"
    is_dashboard = path == "/analytics"
    is_api = path == "/api/analytics" or path.startswith("/api/analytics/")
    if (is_dashboard or is_api) and not _analytics_authorized(request):
        if is_dashboard and config.ADMIN_LOGIN_URL:
            return RedirectResponse(
                config.ADMIN_LOGIN_URL
                + "?"
                + urlencode({"next": _original_request_url(request)}),
                status_code=303,
            )
        if is_dashboard:
            return HTMLResponse("Admin access is not configured.", status_code=503)
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    response = await call_next(request)
    if is_dashboard or is_api:
        response.headers["Cache-Control"] = "no-store"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(shop.router)
app.include_router(api.router)
app.include_router(events.router)
app.include_router(decision.router)
app.include_router(analytics.router)
