"""Study dispatcher configuration.

All values come from environment variables so the service can be configured
entirely from ``docker-compose`` / ``.env``. The two base URLs are the only
required settings; everything else has a sensible default for local testing.
"""
import os

# Public base URLs of the two experiment conditions. These must be the
# *browser-facing* URLs (e.g. ``https://v2.example.com``) because the
# dispatcher 302-redirects the participant's browser to them — internal Docker
# hostnames would not resolve client-side.
V2_BASE_URL = os.environ.get("V2_BASE_URL", "").rstrip("/")
V3_BASE_URL = os.environ.get("V3_BASE_URL", "").rstrip("/")

# Condition label -> destination base URL.
CONDITIONS = {"v2": V2_BASE_URL, "v3": V3_BASE_URL}

# The five persona scenarios the demo sites know how to render. These strings
# MUST match the values the sites validate against in ``shop.py``
# (``_VALID_PERSONAS``).
KNOWN_PERSONAS = [
    "explorer",
    "fastbuyer",
    "detailedcomparator",
    "discounthunter",
    "windowshopper",
]


def _personas_from_env() -> list[str]:
    """Return the personas this deployment actually recruits.

    ``STUDY_PERSONAS`` (comma-separated) restricts assignment to a subset of
    the known scenarios without touching the sites: the unlisted scenario
    wording stays deployed and digest-locked, it is simply never assigned.
    The August 2026 redesign recruits three of the five cells this way.

    Fails closed: a name outside ``KNOWN_PERSONAS`` raises rather than
    silently recruiting the wrong cells, because the sites would 404 the
    persona and every affected assignment would be lost.
    """
    raw = os.environ.get("STUDY_PERSONAS", "").strip()
    if not raw:
        return list(KNOWN_PERSONAS)
    personas = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(personas) - set(KNOWN_PERSONAS))
    if unknown:
        raise ValueError(
            f"STUDY_PERSONAS contains unknown persona(s) {unknown}; "
            f"known: {KNOWN_PERSONAS}"
        )
    if len(personas) != len(set(personas)):
        raise ValueError(f"STUDY_PERSONAS contains duplicates: {raw!r}")
    return personas


PERSONAS = _personas_from_env()

# Where the SQLite assignment ledger lives (bind-mounted volume in Docker).
DB_PATH = os.environ.get("DISPATCHER_DB_PATH", "/app/data/dispatcher.db")

# Per-cell recruitment target. Once every one of the 2x5 = 10 cells has at
# least this many assignments the study is considered full.
TARGET_PER_CELL = int(os.environ.get("TARGET_PER_CELL", "30"))

# Optional URL to send participants to once the study is full (e.g. a Prolific
# "study complete" page). If empty, a plain "study is full" message is shown.
STUDY_FULL_URL = os.environ.get("STUDY_FULL_URL", "").strip()

# Bearer token guarding the /admin/counts endpoint. If empty the endpoint is
# disabled (returns 403) so counts are never exposed unauthenticated.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

# Researcher login shared by the dispatcher and both condition dashboards.
# ADMIN_PASSWORD/ADMIN_SESSION_SECRET fall back to the legacy ADMIN_TOKEN so an
# existing secure deployment can be upgraded without briefly exposing the
# analytics routes. New deployments should use three independent random values.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip() or ADMIN_TOKEN
ADMIN_SESSION_SECRET = (
    os.environ.get("ADMIN_SESSION_SECRET", "").strip() or ADMIN_TOKEN
)
ADMIN_COOKIE_NAME = os.environ.get(
    "ADMIN_COOKIE_NAME", "study_admin_session"
).strip()
ADMIN_COOKIE_DOMAIN = os.environ.get("ADMIN_COOKIE_DOMAIN", "").strip()
ADMIN_COOKIE_SECURE = os.environ.get("ADMIN_COOKIE_SECURE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADMIN_SESSION_TTL_SECONDS = max(
    300, int(os.environ.get("ADMIN_SESSION_TTL_SECONDS", "28800"))
)

# Docker-internal endpoints used by the combined dispatcher dashboard. They
# default to the browser-facing URLs for non-Docker/local deployments.
V2_ANALYTICS_INTERNAL_URL = os.environ.get(
    "V2_ANALYTICS_INTERNAL_URL", V2_BASE_URL
).rstrip("/")
V3_ANALYTICS_INTERNAL_URL = os.environ.get(
    "V3_ANALYTICS_INTERNAL_URL", V3_BASE_URL
).rstrip("/")

# Legal and study-information fields. The defaults identify the thesis project,
# but the controller's service address and contact details must be supplied by
# the deployer; they cannot safely be inferred from the source repository.
STUDY_RESEARCHER_NAME = os.environ.get("STUDY_RESEARCHER_NAME", "Finn Rehnert").strip()
STUDY_INSTITUTION = os.environ.get(
    "STUDY_INSTITUTION", "Technische Hochschule Ulm"
).strip()
LEGAL_CONTROLLER_NAME = os.environ.get("LEGAL_CONTROLLER_NAME", "").strip()
LEGAL_CONTROLLER_ADDRESS = os.environ.get("LEGAL_CONTROLLER_ADDRESS", "").strip()
LEGAL_CONTROLLER_EMAIL = os.environ.get("LEGAL_CONTROLLER_EMAIL", "").strip()
LEGAL_CONTROLLER_PHONE = os.environ.get("LEGAL_CONTROLLER_PHONE", "").strip()
LEGAL_CONTENT_RESPONSIBLE = os.environ.get("LEGAL_CONTENT_RESPONSIBLE", "").strip()
LEGAL_ADDITIONAL_IMPRINT_DETAILS = os.environ.get(
    "LEGAL_ADDITIONAL_IMPRINT_DETAILS", ""
).strip()
LEGAL_DATA_PROTECTION_CONTACT = os.environ.get(
    "LEGAL_DATA_PROTECTION_CONTACT", ""
).strip()
LEGAL_HOSTING_PROVIDER = os.environ.get("LEGAL_HOSTING_PROVIDER", "").strip()
STUDY_RETENTION_PERIOD = os.environ.get("STUDY_RETENTION_PERIOD", "").strip()
SERVER_LOG_RETENTION_PERIOD = os.environ.get("SERVER_LOG_RETENTION_PERIOD", "").strip()
LEGAL_THIRD_COUNTRY_INFORMATION = os.environ.get(
    "LEGAL_THIRD_COUNTRY_INFORMATION", ""
).strip()

# This changes whenever the consent/privacy wording or processing purpose
# changes materially. It is stored with the participant assignment.
CONSENT_VERSION = "2026-07-18"

# Query-parameter names that may carry the platform's participant id, in
# priority order. Prolific sends ``PROLIFIC_PID``; Clickworker/MTurk vary.
PID_PARAMS = ["pid", "PROLIFIC_PID", "participant_id", "worker_id", "wid"]


def is_configured() -> bool:
    """Return ``True`` once both condition base URLs are set."""
    return bool(V2_BASE_URL and V3_BASE_URL)


def shop_hostnames() -> set[str]:
    """Return the bare hostnames of the two condition sites.

    Used to detect when a legal page is being served through the gateway on a
    demo-shop domain rather than on the dispatcher itself.
    """
    from urllib.parse import urlsplit

    hosts = set()
    for base in (V2_BASE_URL, V3_BASE_URL):
        if base:
            hostname = urlsplit(base).hostname
            if hostname:
                hosts.add(hostname.lower())
    return hosts


def admin_login_configured() -> bool:
    """Return whether the researcher login has all required credentials."""
    return bool(ADMIN_USERNAME and ADMIN_PASSWORD and ADMIN_SESSION_SECRET)


def legal_information_complete() -> bool:
    """Return whether the minimum controller information is configured."""
    return all(
        (
            LEGAL_CONTROLLER_NAME,
            LEGAL_CONTROLLER_ADDRESS,
            LEGAL_CONTROLLER_EMAIL,
            LEGAL_HOSTING_PROVIDER,
            STUDY_RETENTION_PERIOD,
            SERVER_LOG_RETENTION_PERIOD,
            LEGAL_THIRD_COUNTRY_INFORMATION,
        )
    )
