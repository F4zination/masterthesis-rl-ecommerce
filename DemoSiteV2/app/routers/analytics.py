from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from ..database import get_db
from ..models import DecisionLog, BanditArmStat, Event, Order
from ..services.decision import PRIOR_COUNT, PRIOR_MEAN

DISPLAY_TZ = ZoneInfo("Europe/Berlin")


def _fmt_berlin(naive_utc: datetime | None) -> str | None:
    """Convert a naive UTC datetime to an ISO-8601 string in Europe/Berlin local time.

    Args:
        naive_utc: A timezone-naive datetime assumed to be UTC, or ``None``.

    Returns:
        A local-time ISO string without timezone suffix (e.g. ``"2026-04-30T14:30:00"``),
        or ``None`` if the input is ``None``.
    """
    if naive_utc is None:
        return None
    return naive_utc.replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ).strftime("%Y-%m-%dT%H:%M:%S")


router = APIRouter(tags=["analytics"])
templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)


@router.get("/analytics", response_class=HTMLResponse)
def analytics_dashboard(request: Request):
    """Render the analytics dashboard SPA shell.

    The dashboard itself is a client-side single-page application that
    fetches data from the ``/api/analytics/*`` endpoints; this handler
    only serves the HTML skeleton.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        An HTML response rendering ``analytics.html``.
    """
    return templates.TemplateResponse(request, name="analytics.html", context={})


@router.get("/api/analytics/summary")
def api_summary(db: Session = Depends(get_db)):
    """Return aggregate KPIs for the analytics dashboard.

    Queries counts for decisions, sessions, events, widget interactions,
    add-to-carts, purchases, total revenue, and explore vs. exploit splits.

    Args:
        db: Injected database session.

    Returns:
        A dict with keys: ``total_decisions``, ``unique_sessions``,
        ``total_events``, ``widget_clicks``, ``widget_dismissals``,
        ``add_to_cart_count``, ``purchases``, ``total_revenue``,
        ``explore_count``, and ``exploit_count``.
    """
    total_decisions = db.query(func.count(DecisionLog.id)).scalar() or 0
    unique_sessions = (
        db.query(func.count(func.distinct(DecisionLog.session_id))).scalar() or 0
    )
    total_events = db.query(func.count(Event.id)).scalar() or 0
    widget_clicks = (
        db.query(func.count(Event.id))
        .filter(Event.event_type == "widget_click")
        .scalar()
        or 0
    )
    widget_dismissals = (
        db.query(func.count(Event.id))
        .filter(Event.event_type == "widget_dismiss")
        .scalar()
        or 0
    )
    add_to_cart_count = (
        db.query(func.count(Event.id))
        .filter(Event.event_type == "add_to_cart")
        .scalar()
        or 0
    )
    purchases = (
        db.query(func.count(Event.id))
        .filter(Event.event_type == "purchase")
        .scalar()
        or 0
    )
    total_revenue = db.query(func.sum(Order.total)).scalar() or 0.0
    study_participants = (
        db.query(func.count(func.distinct(Event.worker_id)))
        .filter(Event.worker_id != "")
        .scalar()
        or 0
    )
    attention_rows = (
        db.query(Event.session_id, Event.metadata_json)
        .filter(Event.event_type == "attention_check")
        .all()
    )
    completed_sessions = {session_id for session_id, _metadata in attention_rows}
    attention_passed = {
        session_id
        for session_id, metadata in attention_rows
        if isinstance(metadata, dict) and metadata.get("passed") is True
    }

    # Explore = propensity ≤ 0.10 (epsilon/n_actions = 0.2/3 ≈ 0.067, exploit > 0.8)
    explore_count = (
        db.query(func.count(DecisionLog.id))
        .filter(DecisionLog.propensity <= 0.10)
        .scalar()
        or 0
    )
    exploit_count = total_decisions - explore_count

    return {
        "total_decisions": total_decisions,
        "unique_sessions": unique_sessions,
        "total_events": total_events,
        "widget_clicks": widget_clicks,
        "widget_dismissals": widget_dismissals,
        "add_to_cart_count": add_to_cart_count,
        "purchases": purchases,
        "total_revenue": round(float(total_revenue), 2),
        "study_participants": study_participants,
        "completed_sessions": len(completed_sessions),
        "attention_passed": len(attention_passed),
        "explore_count": explore_count,
        "exploit_count": exploit_count,
    }


@router.get("/api/analytics/arm_stats")
def api_arm_stats(db: Session = Depends(get_db)):
    """Return per-arm statistics for all tracked bandit arms.

    Includes impressions, cumulative reward, and the Bayesian-smoothed mean
    estimate for each ``(decision_point, context_key, action)`` triplet.

    Args:
        db: Injected database session.

    Returns:
        A list of dicts each containing ``decision_point``, ``context_key``,
        ``action``, ``impressions``, ``reward_sum``, ``mean_estimate``,
        and ``updated_at``.
    """
    stats = (
        db.query(BanditArmStat)
        .order_by(
            BanditArmStat.decision_point,
            BanditArmStat.action,
            BanditArmStat.context_key,
        )
        .all()
    )
    return [
        {
            "decision_point": s.decision_point,
            "context_key": s.context_key,
            "action": s.action,
            "impressions": s.impressions,
            "reward_sum": round(s.reward_sum, 4),
            "mean_estimate": round(
                (s.reward_sum + PRIOR_COUNT * PRIOR_MEAN) / (s.impressions + PRIOR_COUNT),
                4,
            ),
            "updated_at": _fmt_berlin(s.updated_at),
        }
        for s in stats
    ]


@router.get("/api/analytics/action_distribution")
def api_action_distribution(db: Session = Depends(get_db)):
    """Return the count of decisions grouped by decision point and action.

    Used to visualise how often each widget action has been served at each
    funnel step.

    Args:
        db: Injected database session.

    Returns:
        A list of dicts each containing ``decision_point``, ``action``,
        and ``count``.
    """
    rows = (
        db.query(
            DecisionLog.decision_point,
            DecisionLog.action,
            func.count(DecisionLog.id).label("count"),
        )
        .group_by(DecisionLog.decision_point, DecisionLog.action)
        .all()
    )
    return [
        {"decision_point": r.decision_point, "action": r.action, "count": r.count}
        for r in rows
    ]


@router.get("/api/analytics/timeline")
def api_timeline(db: Session = Depends(get_db)):
    """Return a time-bucketed series of decision counts for charting.

    Automatically selects the appropriate time granularity:
    minute buckets for spans under 2 hours, hourly for under 3 days,
    and daily for longer spans.  Returns an empty series if no decisions
    have been recorded yet.

    Args:
        db: Injected database session.

    Returns:
        A dict with ``granularity`` (``"minute"``, ``"hour"``, or ``"day"``),
        ``min`` and ``max`` ISO-8601 timestamps, and ``rows`` — a list of
        dicts with ``bucket``, ``decision_point``, and ``count``.
    """
    span = db.execute(
        text("SELECT MIN(timestamp), MAX(timestamp) FROM decision_logs")
    ).fetchone()

    if not span or not span[0]:
        return {"granularity": "hour", "rows": []}

    def _parse(v):
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace(" ", "T"))

    # Timestamps in the DB are naive UTC; convert to Berlin local time.
    def _to_berlin(naive_utc: datetime) -> datetime:
        return naive_utc.replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ)

    min_dt, max_dt = _parse(span[0]), _parse(span[1])
    delta = (max_dt - min_dt).total_seconds()

    # Determine the Berlin UTC offset (handles CET/CEST DST automatically).
    berlin_offset_s = _to_berlin(min_dt).utcoffset().total_seconds()
    offset_hours = int(berlin_offset_s / 3600)
    sqlite_modifier = f"'{'+' if offset_hours >= 0 else ''}{offset_hours} hours'"

    if delta < 7200:          # < 2 h  → minute buckets
        bucket_sql = f"strftime('%Y-%m-%dT%H:%M', timestamp, {sqlite_modifier})"
        granularity = "minute"
    elif delta < 259200:      # < 3 d  → hourly
        bucket_sql = f"strftime('%Y-%m-%dT%H:00', timestamp, {sqlite_modifier})"
        granularity = "hour"
    else:                     # ≥ 3 d  → daily
        bucket_sql = f"strftime('%Y-%m-%d', timestamp, {sqlite_modifier})"
        granularity = "day"

    rows = db.execute(
        text(
            f"""
            SELECT
                {bucket_sql} AS bucket,
                decision_point,
                COUNT(*) AS cnt
            FROM decision_logs
            GROUP BY bucket, decision_point
            ORDER BY bucket
            """
        )
    ).fetchall()

    min_berlin = _to_berlin(min_dt)
    max_berlin = _to_berlin(max_dt)

    return {
        "granularity": granularity,
        # No "Z" suffix → JS parses as local time, matching the bucketed strings
        "min": min_berlin.strftime("%Y-%m-%dT%H:%M"),
        "max": max_berlin.strftime("%Y-%m-%dT%H:%M"),
        "rows": [{"bucket": r[0], "decision_point": r[1], "count": r[2]} for r in rows],
    }


@router.get("/api/analytics/recent")
def api_recent(db: Session = Depends(get_db)):
    """Return the 100 most recent bandit decisions for the live feed.

    Session IDs are truncated to the first 8 characters for display.

    Args:
        db: Injected database session.

    Returns:
        A list of dicts each containing ``id``, ``session_id``,
        ``decision_point``, ``action``, ``propensity``, ``timestamp``,
        and ``estimates`` (per-arm value estimates at decision time).
    """
    rows = (
        db.query(DecisionLog)
        .order_by(DecisionLog.timestamp.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": r.id,
            "session_id": r.session_id[:8] + "...",
            "decision_point": r.decision_point,
            "action": r.action,
            "propensity": round(r.propensity, 4),
            "timestamp": _fmt_berlin(r.timestamp),
            "estimates": (r.context_json or {}).get("estimates", {}),
        }
        for r in rows
    ]
