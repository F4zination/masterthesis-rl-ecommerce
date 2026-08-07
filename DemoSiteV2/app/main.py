from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from shared_schema.admin_session import verify_admin_session
from shared_schema.policy_contract import (
    PolicyContractError,
    validate_bandit_policy_connection,
)

from . import config
from .config import APP_NAME, DEBUG
from .database import init_db
from .seed import seed_database, reconcile_product_images
from .routers import shop, api, events, decision, analytics

BASE_DIR = Path(__file__).resolve().parent


def validate_required_bandit_policy() -> None:
    """Fail closed unless the frozen policy covers the full serving domain."""
    if not config.REQUIRE_BANDIT_POLICY:
        return
    if not config.FREEZE_POLICY:
        raise RuntimeError("REQUIRE_BANDIT_POLICY=true requires FREEZE_POLICY=true")
    db_path = Path(config.DATABASE_PATH)
    if not db_path.is_file():
        raise RuntimeError(f"required bandit database is missing: {db_path}")
    try:
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            validate_bandit_policy_connection(connection)
        finally:
            connection.close()
    except (sqlite3.Error, PolicyContractError) as exc:
        raise RuntimeError(
            f"required bandit database failed the serving-domain contract: {exc}"
        ) from exc


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
    # Validate the mounted frozen artifact before init_db can create an empty
    # schema or seed a database that merely looks deployable.
    validate_required_bandit_policy()
    init_db()
    seeded = seed_database()
    if seeded:
        print("Database seeded with categories and products.")
    else:
        print("Database already populated, skipping seed.")
        # Seeding is skipped for an existing database, so realign stored
        # image paths with the repository in case an asset was renamed.
        repointed = reconcile_product_images()
        if repointed:
            print(f"Re-pointed {repointed} product image path(s).")
    yield


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
