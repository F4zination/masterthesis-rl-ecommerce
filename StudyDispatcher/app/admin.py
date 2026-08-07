"""Researcher login and combined study dashboard."""
from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from shared_schema.admin_session import create_admin_session, verify_admin_session

from . import config, store


router = APIRouter()


_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Researcher login</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #172033; background: #eef2f7; }
    * { box-sizing: border-box; }
    body { min-height: 100vh; margin: 0; display: grid; place-items: center; padding: 24px;
      background: radial-gradient(circle at top, rgba(37,99,235,.18), transparent 34rem), #eef2f7; }
    main { width: min(100%, 420px); padding: 38px; background: #fff; border: 1px solid #dbe3ef;
      border-radius: 18px; box-shadow: 0 24px 60px rgba(15,23,42,.13); }
    .eyebrow { margin: 0 0 8px; color: #2563eb; font-size: .76rem; font-weight: 800;
      letter-spacing: .12em; text-transform: uppercase; }
    h1 { margin: 0 0 10px; font-size: 2rem; }
    p { color: #64748b; line-height: 1.55; }
    label { display: block; margin: 18px 0 7px; font-size: .9rem; font-weight: 750; }
    input { width: 100%; padding: 13px 14px; border: 1px solid #cbd5e1; border-radius: 10px; font: inherit; }
    input:focus { border-color: #2563eb; outline: 3px solid rgba(37,99,235,.13); }
    button { width: 100%; margin-top: 22px; padding: 13px; border: 0; border-radius: 10px;
      background: #1d4ed8; color: white; font: inherit; font-weight: 800; cursor: pointer; }
    .error { padding: 11px 13px; border-radius: 9px; background: #fef2f2; color: #b91c1c; font-size: .9rem; }
    a { color: #1d4ed8; }
    .back { margin-top: 22px; text-align: center; font-size: .85rem; }
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Restricted area</p>
    <h1>Researcher login</h1>
    <p>Sign in once to access the combined dispatcher dashboard and both condition dashboards.</p>
    __ERROR__
    <form action="/admin/login" method="post">
      <input type="hidden" name="next" value="__NEXT__">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required autofocus>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Sign in</button>
    </form>
    <p class="back"><a href="/">Return to participant entry</a></p>
  </main>
</body>
</html>"""


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Study administration</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #172033; background: #f1f5f9; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    header { padding: 24px max(24px, calc((100vw - 1180px)/2)); color: #fff;
      background: linear-gradient(120deg, #172554, #1d4ed8); display: flex; gap: 20px;
      align-items: center; justify-content: space-between; }
    header p { margin: 5px 0 0; color: #bfdbfe; }
    h1 { margin: 0; font-size: clamp(1.45rem, 4vw, 2.1rem); }
    main { width: min(1180px, calc(100% - 32px)); margin: 26px auto 50px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, .button { padding: 10px 14px; border: 1px solid #cbd5e1; border-radius: 9px;
      background: #fff; color: #172033; font: inherit; font-weight: 700; cursor: pointer; text-decoration: none; }
    header button { color: #fff; border-color: rgba(255,255,255,.45); background: rgba(255,255,255,.1); }
    .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 13px; }
    .card { padding: 18px; border: 1px solid #dbe3ef; border-radius: 14px; background: #fff;
      box-shadow: 0 5px 18px rgba(15,23,42,.045); }
    .kpi { grid-column: span 2; }
    .kpi .value { margin-top: 6px; font-size: 1.75rem; font-weight: 850; letter-spacing: -.03em; }
    .label { color: #64748b; font-size: .78rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
    section { margin-top: 24px; }
    section h2 { margin: 0 0 12px; font-size: 1.12rem; }
    .conditions { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .condition-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 16px; }
    .condition-head h3 { margin: 0; }
    .condition-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .mini strong { display: block; margin-top: 4px; font-size: 1.24rem; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
    th { color: #475569; background: #f8fafc; }
    .status { margin: 0 0 18px; padding: 12px 14px; border-radius: 10px; background: #e0f2fe; color: #075985; }
    .status.error { background: #fef2f2; color: #b91c1c; }
    .muted { color: #64748b; }
    .api-error { color: #b91c1c; }
    @media (max-width: 820px) { .kpi { grid-column: span 3; } .conditions { grid-template-columns: 1fr; } }
    @media (max-width: 520px) { header { align-items: flex-start; flex-direction: column; } .kpi { grid-column: span 6; }
      .condition-grid { grid-template-columns: 1fr 1fr; } .table-wrap { overflow-x: auto; } }
  </style>
</head>
<body>
  <header>
    <div><h1>Study administration</h1><p>Combined V2 contextual-bandit and V3 PPO overview</p></div>
    <form action="/admin/logout" method="post"><button type="submit">Sign out</button></form>
  </header>
  <main>
    <div class="toolbar">
      <button id="refresh" type="button">Refresh data</button>
      <span id="updated" class="muted"></span>
    </div>
    <section><div id="status" class="status">Loading study statistics…</div></section>
    <section class="grid">
      <div class="card kpi"><div class="label">Assignments</div><div class="value" id="assignments">—</div></div>
      <div class="card kpi"><div class="label">Study participants logged</div><div class="value" id="participants">—</div></div>
      <div class="card kpi"><div class="label">Completed sessions</div><div class="value" id="completed">—</div></div>
      <div class="card kpi"><div class="label">Decisions</div><div class="value" id="decisions">—</div></div>
      <div class="card kpi"><div class="label">Purchases</div><div class="value" id="purchases">—</div></div>
      <div class="card kpi"><div class="label">Simulated revenue</div><div class="value" id="revenue">—</div></div>
    </section>
    <section>
      <h2>Conditions</h2>
      <div class="conditions" id="conditions"></div>
    </section>
    <section>
      <h2>Assignment cells</h2>
      <div class="card table-wrap">
        <table><thead><tr><th>Condition</th><th>Persona</th><th>Assigned</th><th>Target</th><th>Progress</th></tr></thead>
        <tbody id="cells"></tbody></table>
      </div>
    </section>
  </main>
  <script>
    const n = value => Number(value || 0).toLocaleString();
    const eur = value => Number(value || 0).toLocaleString(undefined, {style:'currency', currency:'EUR'});
    const metric = (label, value) => `<div class="mini"><span class="label">${label}</span><strong>${value}</strong></div>`;
    async function loadData() {
      const status = document.getElementById('status');
      status.className = 'status'; status.textContent = 'Loading study statistics…';
      try {
        const response = await fetch('/admin/api/overview', {credentials: 'same-origin', cache: 'no-store'});
        if (response.status === 401) { window.location.href = '/admin/login?next=%2Fadmin'; return; }
        if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
        const data = await response.json();
        document.getElementById('assignments').textContent = n(data.assignments.total);
        document.getElementById('participants').textContent = n(data.combined.study_participants);
        document.getElementById('completed').textContent = n(data.combined.completed_sessions);
        document.getElementById('decisions').textContent = n(data.combined.total_decisions);
        document.getElementById('purchases').textContent = n(data.combined.purchases);
        document.getElementById('revenue').textContent = eur(data.combined.total_revenue);
        const conditions = document.getElementById('conditions'); conditions.replaceChildren();
        for (const [label, condition] of Object.entries(data.conditions)) {
          const card = document.createElement('article'); card.className = 'card';
          const title = label.toUpperCase();
          if (!condition.ok) {
            card.innerHTML = `<div class="condition-head"><h3>${title}</h3><a class="button" href="${condition.dashboard_url}">Open dashboard</a></div><p class="api-error"></p>`;
            card.querySelector('.api-error').textContent = condition.error;
          } else {
            const s = condition.summary;
            card.innerHTML = `<div class="condition-head"><h3>${title}</h3><a class="button" href="${condition.dashboard_url}">Open dashboard</a></div><div class="condition-grid">${metric('Participants', n(s.study_participants))}${metric('Completed', n(s.completed_sessions))}${metric('Attention passed', n(s.attention_passed))}${metric('Decisions', n(s.total_decisions))}${metric('Events', n(s.total_events))}${metric('Purchases', n(s.purchases))}</div>`;
          }
          conditions.appendChild(card);
        }
        const cells = document.getElementById('cells'); cells.replaceChildren();
        for (const cell of data.assignments.cells) {
          const row = document.createElement('tr');
          const pct = data.assignments.target_per_cell ? Math.min(100, cell.n / data.assignments.target_per_cell * 100) : 0;
          for (const value of [cell.condition.toUpperCase(), cell.persona, n(cell.n), n(data.assignments.target_per_cell), `${pct.toFixed(0)}%`]) {
            const td = document.createElement('td'); td.textContent = value; row.appendChild(td);
          }
          cells.appendChild(row);
        }
        const unavailable = Object.values(data.conditions).filter(c => !c.ok).length;
        status.textContent = unavailable ? `${unavailable} condition API could not be reached; available values are still shown.` : (data.assignments.full ? 'Recruitment target reached in every cell.' : 'All condition APIs are reachable. Recruitment is still open.');
        if (unavailable) status.className = 'status error';
        document.getElementById('updated').textContent = `Updated ${new Date(data.generated_at).toLocaleString()}`;
      } catch (error) {
        status.className = 'status error'; status.textContent = `Could not load dashboard: ${error.message}`;
      }
    }
    document.getElementById('refresh').addEventListener('click', loadData);
    loadData(); setInterval(loadData, 30000);
  </script>
</body>
</html>"""


def _session_is_valid(request: Request) -> bool:
    token = request.cookies.get(config.ADMIN_COOKIE_NAME, "")
    return verify_admin_session(
        token,
        config.ADMIN_SESSION_SECRET,
        expected_username=config.ADMIN_USERNAME,
    )


def _bearer_is_valid(request: Request) -> bool:
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    return bool(
        config.ADMIN_TOKEN
        and scheme.lower() == "bearer"
        and secrets.compare_digest(supplied, config.ADMIN_TOKEN)
    )


def _request_is_admin(request: Request, legacy_query_token: str = "") -> bool:
    if _session_is_valid(request) or _bearer_is_valid(request):
        return True
    return bool(
        config.ADMIN_TOKEN
        and legacy_query_token
        and secrets.compare_digest(legacy_query_token, config.ADMIN_TOKEN)
    )


def _safe_next(raw_next: str) -> str:
    """Allow local admin paths or the configured V2/V3 public hosts only."""
    raw_next = raw_next.strip()
    if "\\" in raw_next or any(char in raw_next for char in ("\r", "\n", "\x00")):
        return "/admin"
    if raw_next.startswith("/") and not raw_next.startswith("//"):
        return raw_next

    parsed = urlsplit(raw_next)
    allowed_origins = {
        (urlsplit(value).scheme, urlsplit(value).netloc)
        for value in (config.V2_BASE_URL, config.V3_BASE_URL)
        if value
    }
    if (parsed.scheme, parsed.netloc) in allowed_origins:
        return raw_next
    return "/admin"


def _login_page(next_url: str, error: str = "") -> HTMLResponse:
    error_html = f'<p class="error">{escape(error)}</p>' if error else ""
    html = (
        _LOGIN_HTML.replace("__ERROR__", error_html)
        .replace("__NEXT__", escape(_safe_next(next_url), quote=True))
    )
    response = HTMLResponse(html)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/admin/login", response_class=HTMLResponse)
def admin_login(request: Request, next: str = "/admin"):
    if _session_is_valid(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _login_page(next)


@router.post("/admin/login")
async def admin_login_submit(request: Request):
    if not config.admin_login_configured():
        return _login_page(
            "/admin",
            "Admin login is not configured. Set ADMIN_PASSWORD and ADMIN_SESSION_SECRET.",
        )

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=415, detail="unsupported form encoding")
    try:
        form = parse_qs(
            (await request.body()).decode("utf-8"),
            keep_blank_values=True,
            max_num_fields=3,
        )
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid form submission") from None

    username = (form.get("username") or [""])[0]
    password = (form.get("password") or [""])[0]
    next_url = (form.get("next") or ["/admin"])[0]
    username_valid = secrets.compare_digest(username, config.ADMIN_USERNAME)
    password_valid = secrets.compare_digest(password, config.ADMIN_PASSWORD)
    valid = username_valid and password_valid
    if not valid:
        await asyncio.sleep(0.35)
        response = _login_page(next_url, "Invalid username or password.")
        response.status_code = 401
        return response

    session_token = create_admin_session(
        config.ADMIN_SESSION_SECRET,
        config.ADMIN_USERNAME,
        ttl_seconds=config.ADMIN_SESSION_TTL_SECONDS,
    )
    response = RedirectResponse(_safe_next(next_url), status_code=303)
    response.set_cookie(
        config.ADMIN_COOKIE_NAME,
        session_token,
        max_age=config.ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=config.ADMIN_COOKIE_SECURE,
        samesite="lax",
        path="/",
        domain=config.ADMIN_COOKIE_DOMAIN or None,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/admin/logout")
def admin_logout():
    response = RedirectResponse("/admin/login", status_code=303)
    response.delete_cookie(
        config.ADMIN_COOKIE_NAME,
        path="/",
        domain=config.ADMIN_COOKIE_DOMAIN or None,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    if not _session_is_valid(request):
        return RedirectResponse(
            "/admin/login?" + urlencode({"next": "/admin"}), status_code=303
        )
    response = HTMLResponse(_DASHBOARD_HTML)
    response.headers["Cache-Control"] = "no-store"
    return response


def _fetch_json(base_url: str, path: str, session_token: str) -> dict:
    if not base_url:
        raise RuntimeError("analytics URL is not configured")
    request = URLRequest(
        f"{base_url}{path}",
        headers={
            "Accept": "application/json",
            "Cookie": f"{config.ADMIN_COOKIE_NAME}={session_token}",
        },
    )
    with urlopen(request, timeout=4) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise RuntimeError("analytics response is too large")
    result = json.loads(body.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("analytics response is not an object")
    return result


def _fetch_condition(label: str, base_url: str, public_url: str, session_token: str) -> dict:
    dashboard_url = f"{public_url}/analytics"
    try:
        summary = _fetch_json(base_url, "/api/analytics/summary", session_token)
        return {"ok": True, "summary": summary, "dashboard_url": dashboard_url}
    except HTTPError as exc:
        error = f"Analytics API returned HTTP {exc.code}."
    except (URLError, TimeoutError):
        error = "Analytics API is currently unreachable."
    except (json.JSONDecodeError, UnicodeDecodeError, RuntimeError, ValueError):
        error = "Analytics API returned an invalid response."
    return {"ok": False, "error": error, "dashboard_url": dashboard_url, "label": label}


def _assignments_payload() -> dict:
    tally = store.counts()
    return {
        "target_per_cell": config.TARGET_PER_CELL,
        "total": sum(tally.values()),
        "full": store.is_full(),
        "cells": [
            {"condition": condition, "persona": persona, "n": count}
            for (condition, persona), count in sorted(tally.items())
        ],
    }


@router.get("/admin/api/overview")
def admin_overview(request: Request):
    if not _request_is_admin(request):
        raise HTTPException(status_code=401, detail="authentication required")

    session_token = request.cookies.get(config.ADMIN_COOKIE_NAME, "")
    if not verify_admin_session(
        session_token,
        config.ADMIN_SESSION_SECRET,
        expected_username=config.ADMIN_USERNAME,
    ):
        session_token = create_admin_session(
            config.ADMIN_SESSION_SECRET,
            config.ADMIN_USERNAME,
            ttl_seconds=300,
        )

    condition_specs = {
        "v2": (
            config.V2_ANALYTICS_INTERNAL_URL,
            config.V2_BASE_URL,
        ),
        "v3": (
            config.V3_ANALYTICS_INTERNAL_URL,
            config.V3_BASE_URL,
        ),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            label: pool.submit(_fetch_condition, label, internal, public, session_token)
            for label, (internal, public) in condition_specs.items()
        }
        conditions = {label: future.result() for label, future in futures.items()}

    combined_keys = (
        "study_participants",
        "completed_sessions",
        "attention_passed",
        "total_decisions",
        "unique_sessions",
        "total_events",
        "widget_clicks",
        "widget_dismissals",
        "add_to_cart_count",
        "purchases",
        "total_revenue",
    )
    combined = {key: 0 for key in combined_keys}
    for condition in conditions.values():
        if not condition["ok"]:
            continue
        summary = condition["summary"]
        for key in combined_keys:
            value = summary.get(key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                combined[key] += value
    combined["total_revenue"] = round(float(combined["total_revenue"]), 2)

    response = JSONResponse(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assignments": _assignments_payload(),
            "conditions": conditions,
            "combined": combined,
        }
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/admin/counts")
def admin_counts(request: Request, token: str = ""):
    """Return per-cell counts for a session, bearer token, or legacy query token."""
    if not _request_is_admin(request, token):
        raise HTTPException(status_code=403, detail="forbidden")
    response = JSONResponse(_assignments_payload())
    response.headers["Cache-Control"] = "no-store"
    return response
