"""Study dispatcher service.

A single public entry point (``GET /go``) that randomly but *balancedly*
assigns each participant to one experiment cell (V2/V3 x one of five personas)
and 302-redirects their browser to the matching demo site with the
``wid``/``persona`` query params the sites already understand.

The demo sites need no changes: they read ``wid``/``persona`` from the URL on
their landing page and persist them into cookies.
"""
from __future__ import annotations

import re
import secrets
import time
from contextlib import asynccontextmanager
from html import escape
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from . import admin, config, legal, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    yield


app = FastAPI(title="Study Dispatcher", lifespan=lifespan)
app.include_router(admin.router)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shopping Study</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: #172033;
      background: #f2f5fb;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 24px;
      background:
        radial-gradient(circle at top left, rgba(79, 70, 229, .18), transparent 34rem),
        #f2f5fb;
    }
    main {
      width: min(100%, 460px);
      margin: auto;
      padding: 42px;
      border: 1px solid rgba(148, 163, 184, .25);
      border-radius: 22px;
      background: rgba(255, 255, 255, .96);
      box-shadow: 0 24px 65px rgba(30, 41, 59, .14);
    }
    .eyebrow {
      margin: 0 0 10px;
      color: #4f46e5;
      font-size: .78rem;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.9rem, 7vw, 2.55rem);
      line-height: 1.1;
      letter-spacing: -.035em;
    }
    .intro {
      margin: 16px 0 30px;
      color: #64748b;
      line-height: 1.65;
    }
    label[for="nickname"] {
      display: block;
      margin-bottom: 9px;
      font-size: .92rem;
      font-weight: 700;
    }
    input[type="text"] {
      width: 100%;
      padding: 14px 16px;
      border: 1px solid #cbd5e1;
      border-radius: 11px;
      background: #fff;
      color: #172033;
      font: inherit;
      outline: none;
      transition: border-color .18s ease, box-shadow .18s ease;
    }
    input[type="text"]:focus {
      border-color: #6366f1;
      box-shadow: 0 0 0 4px rgba(99, 102, 241, .14);
    }
    button {
      width: 100%;
      margin-top: 16px;
      padding: 14px 18px;
      border: 0;
      border-radius: 11px;
      background: #4f46e5;
      color: #fff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      transition: background .18s ease, transform .18s ease;
    }
    button:hover { background: #4338ca; }
    button:active { transform: translateY(1px); }
    .consent {
      display: grid;
      grid-template-columns: 20px 1fr;
      gap: 10px;
      margin-top: 18px;
      color: #475569;
      font-size: .84rem;
      line-height: 1.5;
    }
    .consent input {
      width: 18px;
      height: 18px;
      margin: 2px 0 0;
      accent-color: #4f46e5;
    }
    a { color: #4338ca; }
    .note {
      margin: 18px 0 0;
      color: #94a3b8;
      font-size: .8rem;
      line-height: 1.5;
      text-align: center;
    }
    footer {
      width: min(100%, 540px);
      margin: 28px auto 0;
      color: #64748b;
      font-size: .8rem;
      line-height: 1.6;
      text-align: center;
    }
    footer a { margin: 0 7px; }
    @media (max-width: 520px) {
      main { padding: 30px 24px; }
    }
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Participant access</p>
    <h1>Welcome to the shopping study</h1>
    <p class="intro">Choose a nickname to begin. We will create a unique participant ID and take you to the study.</p>
    <form action="/start" method="post">
      <label for="nickname">Nickname</label>
      <input id="nickname" name="nickname" type="text" minlength="2" maxlength="32"
             autocomplete="off" placeholder="e.g. blue-otter" required autofocus>
      <div class="consent">
        <input id="consent" name="consent" type="checkbox" value="yes" required>
        <label for="consent">I have read the
          <a href="/participant-information" target="_blank">participant information</a>
          and <a href="/privacy" target="_blank">privacy notice</a>. I voluntarily consent
          to participate and to the described processing of my pseudonymized study data.
        </label>
      </div>
      <button type="submit">Start study*</button>
    </form>
    <p class="note">Your nickname is only used to create your participant ID. Do not use
      your real name.<br><br>*By checking the box and clicking “Start study”, you consent
      to participate and allow your pseudonymized data to be used for this master thesis.
      Only anonymous, aggregated results will be published.</p>
  </main>
  <footer>
    <div>
      <a href="/imprint">Legal notice</a>
      <a href="/privacy">Privacy</a>
      <a href="/participant-information">Participant information</a>
      <a href="/admin/login">Researcher login</a>
    </div>
    <p>&copy; 2026 __STUDY_RESEARCHER__ &middot; Master thesis study</p>
  </footer>
</body>
</html>
"""


def _resolve_pid(request: Request, pid: str) -> str | None:
    """Find the participant id from the explicit param or known aliases."""
    if pid:
        return pid
    for key in config.PID_PARAMS:
        value = request.query_params.get(key)
        if value:
            return value
    return None


def _participant_id(nickname: str) -> str | None:
    """Create a unique, URL-safe participant id from a submitted nickname."""
    nickname = nickname.strip()
    if not 2 <= len(nickname) <= 32:
        return None

    # Keep the id recognisable while avoiding whitespace and URL punctuation.
    slug = re.sub(r"[^a-z0-9]+", "-", nickname.lower()).strip("-")[:24]
    if not slug:
        slug = "participant"
    return f"{slug}-{secrets.token_hex(6)}"


def _form_value(form: dict[str, list[str]], name: str) -> str:
    """Return the first value for an URL-encoded form field."""
    values = form.get(name, [])
    return values[0].strip() if values else ""


def _dispatch(
    resolved_pid: str,
    *,
    consent_version: str = "",
    consented_at: float | None = None,
    redirect_status: int = 302,
):
    """Assign a participant and redirect them to their experiment cell."""
    if not config.is_configured():
        raise HTTPException(
            status_code=500,
            detail="Dispatcher not configured: set V2_BASE_URL and V3_BASE_URL.",
        )

    existing = store.get(resolved_pid)

    # Only block genuinely new participants once the study is full; workers who
    # already have an assignment must always be able to resume their session.
    if existing is None and store.is_full():
        if config.STUDY_FULL_URL:
            return RedirectResponse(config.STUDY_FULL_URL, status_code=302)
        return HTMLResponse(
            legal.message_page(
                "Thanks!",
                "This study has reached its participant target and is no longer "
                "accepting new sessions.",
            ),
            status_code=200,
        )

    record, _created = store.assign(
        resolved_pid,
        consent_version=consent_version,
        consented_at=consented_at,
    )
    base = config.CONDITIONS[record.condition]
    target = f"{base}/?{urlencode({'wid': record.wid, 'persona': record.persona})}"
    return RedirectResponse(target, status_code=redirect_status)


@app.get("/healthz")
def healthz():
    """Liveness probe."""
    return {
        "ok": True,
        "configured": config.is_configured(),
        "legal_information_complete": config.legal_information_complete(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Show the participant entry page."""
    researcher = escape(config.STUDY_RESEARCHER_NAME or "Study project")
    return HTMLResponse(_INDEX_HTML.replace("__STUDY_RESEARCHER__", researcher))


@app.get("/imprint", response_class=HTMLResponse)
def imprint(request: Request):
    """Show provider information for the study website."""
    return HTMLResponse(legal.imprint_page(on_shop_host=legal.is_shop_host(request)))


@app.get("/privacy", response_class=HTMLResponse)
def privacy(request: Request):
    """Show the study's Article 13 GDPR privacy information."""
    return HTMLResponse(legal.privacy_page(on_shop_host=legal.is_shop_host(request)))


@app.get("/participant-information", response_class=HTMLResponse)
def participant_information(request: Request):
    """Show the informed-consent information for participants."""
    return HTMLResponse(
        legal.participant_information_page(on_shop_host=legal.is_shop_host(request))
    )


@app.post("/start")
async def start(request: Request):
    """Validate explicit study consent, mint an id, and dispatch the participant."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=415, detail="unsupported form encoding")

    try:
        body = (await request.body()).decode("utf-8")
        form = parse_qs(body, keep_blank_values=True, max_num_fields=4)
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid form submission") from None

    nickname = _form_value(form, "nickname")
    participant_id = _participant_id(nickname)
    if participant_id is None:
        return HTMLResponse(
            legal.message_page(
                "Invalid nickname",
                "Please return to the previous page and enter 2 to 32 characters.",
            ),
            status_code=400,
        )
    if _form_value(form, "consent") != "yes":
        return HTMLResponse(
            legal.message_page(
                "Consent required",
                "Please return to the previous page and review the participant and "
                "privacy information before deciding whether to take part.",
            ),
            status_code=400,
        )

    return _dispatch(
        participant_id,
        consent_version=config.CONSENT_VERSION,
        consented_at=time.time(),
        redirect_status=303,
    )


@app.get("/go")
def go(request: Request, pid: str = ""):
    """Assign a cell and redirect the participant to their demo site.

    Query params:
        pid: the platform participant id (also accepted via ``PROLIFIC_PID``,
            ``participant_id``, ``worker_id``, or ``wid``).
        If no id is present, an anonymous id is minted so direct links still
        work for manual testing. The participant entry form uses ``POST /start``
        instead, so consent can be validated and recorded first.
    """
    resolved_pid = _resolve_pid(request, pid)
    if resolved_pid is None:
        resolved_pid = "anon-" + secrets.token_hex(6)
    return _dispatch(resolved_pid)
