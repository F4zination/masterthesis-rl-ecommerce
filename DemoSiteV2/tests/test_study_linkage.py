"""Phase 6 verification: study attribution linkage + frozen-policy invariance.

Runnable standalone (no pytest / httpx required):

    .venv/Scripts/python DemoSiteV2/tests/test_study_linkage.py
    FREEZE_POLICY=true .venv/Scripts/python DemoSiteV2/tests/test_study_linkage.py

The plain run exercises ``worker_id``/``persona`` linkage across ``Event``,
``Order`` and ``DecisionLog`` (Phase 1) plus the attention-check completion
flow (Phase 2). With ``FREEZE_POLICY=true`` it additionally asserts that no
``BanditArmStat`` row mutates during a full session.

The functions are named ``test_*`` so they can also be collected by pytest
when it (and httpx are not needed here) is available; the module bootstraps
its own ``sys.path`` and a throwaway database on import.
"""
from __future__ import annotations

import os
import re
import sys
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent


def _bootstrap() -> None:
    # SharedSchema + the app package must be importable. The repo's editable
    # shared_schema install points at a stale path, so add it explicitly.
    for p in (str(REPO_ROOT / "SharedSchema"), str(APP_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Always use a throwaway DB so the test never touches a real one, even if
    # DATABASE_PATH is set in the environment.
    os.environ["DATABASE_PATH"] = os.path.join(
        tempfile.mkdtemp(prefix="study_test_"), "study_test.db"
    )


_bootstrap()

from starlette.requests import Request  # noqa: E402
from starlette.responses import RedirectResponse  # noqa: E402

from app import config  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import (  # noqa: E402
    BanditArmStat, CartItem, Category, DecisionLog, Event, Order, Product,
)
from app.routers import shop as shop_router  # noqa: E402
from app.routers.decision import DecisionRequest, request_decision  # noqa: E402
from app.routers.events import EventRequest, track_event  # noqa: E402
from shared_schema.features import _eligible_actions, _normalize_context  # noqa: E402

WID = "W123"
PERSONA = "explorer"
COOKIES = {"session_id": "s-test", "wid": WID, "study_persona": PERSONA}


def _request(cookies: dict, form: dict | None = None) -> Request:
    """Build a minimal Starlette ``Request`` carrying cookies (and form body)."""
    headers: list[tuple[bytes, bytes]] = []
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
    if cookie_str:
        headers.append((b"cookie", cookie_str.encode()))
    body = b""
    method = "GET"
    if form is not None:
        method = "POST"
        body = urlencode(form).encode()
        headers.append((b"content-type", b"application/x-www-form-urlencoded"))
    scope = {
        "type": "http", "http_version": "1.1", "method": method,
        "path": "/", "raw_path": b"/", "query_string": b"", "headers": headers,
    }
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _fresh_db():
    """Initialise schema (incl. migrations) and seed two products."""
    init_db()
    db = SessionLocal()
    for model in (CartItem, Event, Order, DecisionLog, BanditArmStat, Product, Category):
        db.query(model).delete()
    db.commit()
    cat = Category(name="Electronics", slug="electronics", description="")
    db.add(cat)
    db.flush()
    db.add_all([
        Product(name="Wireless Headphones", slug="wireless-headphones", price=89.99, category_id=cat.id),
        Product(name="Mechanical Keyboard", slug="mechanical-keyboard", price=129.99, category_id=cat.id),
    ])
    db.commit()
    return db


def _decision_request(**kw) -> DecisionRequest:
    """Build a DecisionRequest using only fields the model declares (V2 vs V3)."""
    fields = set(DecisionRequest.model_fields)
    return DecisionRequest(**{k: v for k, v in kw.items() if k in fields})


def test_linkage() -> None:
    db = _fresh_db()
    sid = COOKIES["session_id"]
    db.add(CartItem(session_id=sid, product_id=db.query(Product).first().id, quantity=1))
    db.commit()

    # 1) decision -> DecisionLog.context_json carries attribution
    request_decision(
        _decision_request(session_id=sid, decision_point="landing", context={"screen_width": 1280}),
        _request(COOKIES), db=db,
    )
    log = db.query(DecisionLog).order_by(DecisionLog.id.desc()).first()
    assert log is not None, "no DecisionLog written"
    assert log.context_json.get("worker_id") == WID, log.context_json
    assert log.context_json.get("persona") == PERSONA, log.context_json

    # 2) event -> Event row carries attribution
    track_event(
        EventRequest(session_id=sid, event_type="page_view", page="/"),
        _request(COOKIES), db=db,
    )
    ev = db.query(Event).filter(Event.event_type == "page_view").first()
    assert ev.worker_id == WID and ev.persona == PERSONA

    # 3) checkout -> Order + purchase event carry attribution
    resp = shop_router.process_checkout(_request(COOKIES), db=db)
    assert isinstance(resp, RedirectResponse)
    order = db.query(Order).order_by(Order.id.desc()).first()
    assert order is not None and order.worker_id == WID and order.persona == PERSONA
    purchase = db.query(Event).filter(Event.event_type == "purchase").first()
    assert purchase.worker_id == WID and purchase.persona == PERSONA

    # 4) attention check -> attention_check event + completion code reveal
    # Graded on category recall, not product recall: the questions mix one real
    # shop category with decoy departments, so a product name scores zero.
    real_name = db.query(Category).first().name
    out = asyncio.run(shop_router.submit_attention(
        _request(COOKIES, {"q1": real_name, "q2": real_name, "order_id": str(order.id)}), db=db,
    ))
    att = db.query(Event).filter(Event.event_type == "attention_check").first()
    assert att is not None and att.worker_id == WID and att.persona == PERSONA
    assert att.metadata_json.get("passed") is True, att.metadata_json
    saved_code = str(att.metadata_json.get("completion_code", ""))
    assert re.fullmatch(r"\d{6}", saved_code), "completion code not persisted"
    rendered = out.body.decode()
    assert saved_code in rendered, "persisted completion code not rendered"
    assert "saved automatically" in rendered
    assert "job you accepted" not in rendered

    # 5) cell recovery: distinct (worker_id, persona) reconstructs the assignment
    cells = db.query(Event.worker_id, Event.persona).distinct().all()
    assert (WID, PERSONA) in cells
    db.close()
    print("[linkage] OK — worker_id/persona on Event, Order, DecisionLog, "
          "attention_check; completion code persisted before rendering")


def test_new_worker_gets_fresh_session_without_breaking_resume() -> None:
    db = _fresh_db()
    response = shop_router.home(
        _request({"session_id": "old-session", "wid": "OLD-WORKER"}),
        db=db,
        wid=WID,
        persona=PERSONA,
    )
    session_headers = [
        value for value in response.headers.getlist("set-cookie")
        if value.startswith("session_id=")
    ]
    assert len(session_headers) == 1, response.headers.getlist("set-cookie")
    assert "old-session" not in session_headers[0]

    resumed = shop_router.home(
        _request({"session_id": "current-session", "wid": WID}),
        db=db,
        wid=WID,
        persona=PERSONA,
    )
    assert not any(
        value.startswith("session_id=")
        for value in resumed.headers.getlist("set-cookie")
    ), resumed.headers.getlist("set-cookie")
    db.close()
    print("[session]  OK - new worker resets session; same worker resumes")


def test_freeze_no_mutation() -> None:
    if not config.FREEZE_POLICY:
        print("[freeze]  skipped (set FREEZE_POLICY=true to run)")
        return

    db = _fresh_db()
    sid = COOKIES["session_id"]
    db.add(CartItem(session_id=sid, product_id=db.query(Product).first().id, quantity=1))
    db.commit()

    decision_point = "landing"
    context = {"screen_width": 1280}
    request_decision(
        _decision_request(session_id=sid, decision_point=decision_point, context=context),
        _request(COOKIES), db=db,
    )
    assert db.query(BanditArmStat).count() == 0, \
        "frozen serving inserted rows for an unseen context"
    context_key, _ = _normalize_context(decision_point, context)
    sentinel = datetime(2020, 1, 1, 0, 0, 0)
    for action in _eligible_actions(decision_point):
        db.add(BanditArmStat(
            decision_point=decision_point, context_key=context_key, action=action,
            impressions=5, reward_sum=2.5, updated_at=sentinel,
        ))
    db.commit()
    before = {r.id: (r.impressions, r.reward_sum, r.updated_at) for r in db.query(BanditArmStat).all()}

    # Full frozen session over existing rows: decision + reward-bearing event + purchase.
    res = request_decision(
        _decision_request(session_id=sid, decision_point=decision_point, context=context),
        _request(COOKIES), db=db,
    )
    track_event(
        EventRequest(session_id=sid, event_type="widget_click", page="/",
                     metadata={"decision_id": res.get("decision_id"), "decision_point": decision_point}),
        _request(COOKIES), db=db,
    )
    shop_router.process_checkout(_request(COOKIES), db=db)

    for r in db.query(BanditArmStat).all():
        if r.id in before:
            assert (r.impressions, r.reward_sum, r.updated_at) == before[r.id], \
                f"BanditArmStat row {r.id} mutated under FREEZE_POLICY"
        else:
            assert r.impressions == 0 and r.reward_sum == 0.0, \
                f"new arm {r.id} gained reward under FREEZE_POLICY"
    db.close()
    print("[freeze]  OK — no BanditArmStat row mutated; new arms remain zeroed")


if __name__ == "__main__":
    test_linkage()
    test_new_worker_gets_fresh_session_without_breaking_resume()
    test_freeze_no_mutation()
    print("ALL CHECKS PASSED")
