from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from ..config import ANALYTICS_TIMELINE_LOOKBACK_HOURS
from ..database import get_db
from ..models import DecisionLog, BanditArmStat, Event, Order
from ..services.decision import PRIOR_COUNT, PRIOR_MEAN
from ..services.learner import get_learner_status
from ..services.ppo import get_ppo_health_status

try:
    DISPLAY_TZ = ZoneInfo("Europe/Berlin")
except ZoneInfoNotFoundError:
    # Fallback for environments where IANA timezone data is missing.
    DISPLAY_TZ = timezone.utc


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


def _policy_source_label(context: dict | None) -> str:
    ctx = context or {}
    explicit = str(ctx.get("policy_source") or "").strip()
    if explicit and explicit != "unknown":
        return explicit

    model = str(ctx.get("model") or "").strip().lower()
    if model.startswith("ppo_") or model == "ppo_clip_actor_critic":
        return "ppo_policy"
    if model.startswith("epsilon_greedy"):
        return "contextual_bandit"
    if model == "offline_unavailable_noop":
        return "offline_policy_noop"
    if model.startswith("tabular_") or "offline" in model:
        return "offline_policy"
    if model == "timing_gate_v1":
        return "timing_gate"
    return "unknown"


def _context_label(context_key: str) -> str:
    if not context_key:
        return "unknown"
    parts = context_key.split("|")
    if len(parts) <= 1:
        return context_key
    values = parts[1:]
    preview = " • ".join(values[:4])
    return preview + (" …" if len(values) > 4 else "")


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


@router.get("/api/analytics/visitor_overview")
def api_visitor_overview(db: Session = Depends(get_db)):
    """Return session-based visitor and funnel metrics for the V3 dashboard.

    Uses the anonymous session identifier as the best available visitor proxy.
    Aggregates recent site activity into a compact overview suitable for
    thesis-facing dashboard KPIs and the funnel visualization.
    """
    decision_sessions = {
        session_id
        for (session_id,) in db.query(DecisionLog.session_id).distinct().all()
        if session_id
    }
    event_sessions = {
        session_id
        for (session_id,) in db.query(Event.session_id).distinct().all()
        if session_id
    }
    order_sessions = {
        session_id
        for (session_id,) in db.query(Order.session_id).distinct().all()
        if session_id
    }

    visitor_sessions = event_sessions | decision_sessions | order_sessions

    add_to_cart_sessions = {
        session_id
        for (session_id,) in db.query(Event.session_id)
        .filter(Event.event_type == "add_to_cart")
        .distinct()
        .all()
        if session_id
    }
    checkout_sessions = {
        session_id
        for (session_id,) in db.query(Event.session_id)
        .filter(Event.event_type == "checkout_submit")
        .distinct()
        .all()
        if session_id
    }
    purchase_sessions = order_sessions or {
        session_id
        for (session_id,) in db.query(Event.session_id)
        .filter(Event.event_type == "purchase")
        .distinct()
        .all()
        if session_id
    }

    visitor_count = len(visitor_sessions)
    decision_count = db.query(func.count(DecisionLog.id)).scalar() or 0
    purchase_count = db.query(func.count(Order.id)).scalar() or 0
    revenue_total = db.query(func.sum(Order.total)).scalar() or 0.0

    conversion_rate = (len(purchase_sessions) / visitor_count * 100.0) if visitor_count else 0.0
    decision_coverage = (len(decision_sessions) / visitor_count * 100.0) if visitor_count else 0.0
    avg_decisions_per_visitor = (decision_count / visitor_count) if visitor_count else 0.0

    return {
        "visitors": visitor_count,
        "decisioned_sessions": len(decision_sessions),
        "add_to_cart_sessions": len(add_to_cart_sessions),
        "checkout_sessions": len(checkout_sessions),
        "purchase_sessions": len(purchase_sessions),
        "purchases": purchase_count,
        "conversion_rate": round(conversion_rate, 2),
        "decision_coverage_rate": round(decision_coverage, 2),
        "avg_decisions_per_visitor": round(avg_decisions_per_visitor, 2),
        "revenue_total": round(float(revenue_total), 2),
        "funnel": [
            {"stage": "Visitors", "count": visitor_count},
            {"stage": "Decisioned", "count": len(decision_sessions)},
            {"stage": "Add to cart", "count": len(add_to_cart_sessions)},
            {"stage": "Checkout", "count": len(checkout_sessions)},
            {"stage": "Purchase", "count": len(purchase_sessions)},
        ],
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


@router.get("/api/analytics/decision_distribution")
def api_decision_distribution(db: Session = Depends(get_db)):
    """Return V3 decision distribution across point, action, and policy source."""
    rows = db.query(DecisionLog).all()

    by_point = Counter()
    by_action = Counter()
    by_source = Counter()
    by_point_and_source = Counter()

    for row in rows:
        ctx = row.context_json or {}
        source = _policy_source_label(ctx)
        by_point[row.decision_point] += 1
        by_action[row.action] += 1
        by_source[source] += 1
        by_point_and_source[(row.decision_point, source)] += 1

    return {
        "total_decisions": len(rows),
        "by_point": [
            {"decision_point": key, "count": value}
            for key, value in sorted(by_point.items())
        ],
        "by_action": [
            {"action": key, "count": value}
            for key, value in sorted(by_action.items())
        ],
        "by_source": [
            {"policy_source": key, "count": value}
            for key, value in sorted(by_source.items())
        ],
        "by_point_and_source": [
            {"decision_point": key[0], "policy_source": key[1], "count": value}
            for key, value in sorted(by_point_and_source.items())
        ],
    }


@router.get("/api/analytics/confidence_summary")
def api_confidence_summary(db: Session = Depends(get_db)):
    """Return propensity-based confidence summaries for V3 decisions."""
    rows = db.query(DecisionLog.decision_point, DecisionLog.propensity).all()

    buckets = [
        ("<0.10", lambda value: value < 0.10),
        ("0.10-0.24", lambda value: 0.10 <= value < 0.25),
        ("0.25-0.49", lambda value: 0.25 <= value < 0.50),
        ("0.50-0.74", lambda value: 0.50 <= value < 0.75),
        ("0.75+", lambda value: value >= 0.75),
    ]

    bucket_counts = Counter()
    by_point_count = Counter()
    by_point_sum = Counter()
    by_point_low_conf = Counter()

    for decision_point, propensity in rows:
        value = float(propensity or 0.0)
        by_point_count[decision_point] += 1
        by_point_sum[decision_point] += value
        if value < 0.25:
            by_point_low_conf[decision_point] += 1
        for label, predicate in buckets:
            if predicate(value):
                bucket_counts[label] += 1
                break

    total = len(rows)
    avg_propensity = (sum(by_point_sum.values()) / total) if total else 0.0
    low_conf_share = (sum(by_point_low_conf.values()) / total * 100.0) if total else 0.0

    return {
        "total_decisions": total,
        "average_propensity": round(avg_propensity, 4),
        "low_confidence_share": round(low_conf_share, 2),
        "bucket_counts": [
            {"bucket": label, "count": bucket_counts.get(label, 0)}
            for label, _ in buckets
        ],
        "by_decision_point": [
            {
                "decision_point": point,
                "decisions": by_point_count[point],
                "average_propensity": round(by_point_sum[point] / by_point_count[point], 4),
                "low_confidence_share": round(
                    by_point_low_conf[point] / by_point_count[point] * 100.0,
                    2,
                ),
            }
            for point in sorted(by_point_count)
        ],
    }


@router.get("/api/analytics/context_insights")
def api_context_insights(db: Session = Depends(get_db)):
    """Return compact context-aware learning summaries for V3 operations detail."""
    stats = db.query(BanditArmStat).filter(BanditArmStat.impressions > 0).all()

    grouped: dict[tuple[str, str], dict] = {}
    for stat in stats:
        key = (stat.decision_point, stat.context_key)
        entry = grouped.setdefault(
            key,
            {
                "decision_point": stat.decision_point,
                "context_key": stat.context_key,
                "context_label": _context_label(stat.context_key),
                "schema_version": stat.schema_version,
                "total_impressions": 0,
                "total_reward": 0.0,
                "actions": [],
            },
        )
        mean_estimate = round(
            (stat.reward_sum + PRIOR_COUNT * PRIOR_MEAN) / (stat.impressions + PRIOR_COUNT),
            4,
        )
        entry["total_impressions"] += stat.impressions
        entry["total_reward"] += stat.reward_sum
        entry["actions"].append(
            {
                "action": stat.action,
                "impressions": stat.impressions,
                "reward_sum": round(stat.reward_sum, 4),
                "mean_estimate": mean_estimate,
            }
        )

    by_point: dict[str, list[dict]] = {}
    for entry in grouped.values():
        actions = sorted(entry["actions"], key=lambda item: item["mean_estimate"], reverse=True)
        best_action = actions[0] if actions else None
        by_point.setdefault(entry["decision_point"], []).append(
            {
                "context_key": entry["context_key"],
                "context_label": entry["context_label"],
                "schema_version": entry["schema_version"],
                "total_impressions": entry["total_impressions"],
                "total_reward": round(entry["total_reward"], 4),
                "best_action": best_action["action"] if best_action else "n/a",
                "best_mean_estimate": best_action["mean_estimate"] if best_action else 0.0,
            }
        )

    return {
        "points": [
            {
                "decision_point": point,
                "contexts": sorted(contexts, key=lambda item: item["total_impressions"], reverse=True)[:3],
            }
            for point, contexts in sorted(by_point.items())
        ]
    }


@router.get("/api/analytics/timeline")
def api_timeline(db: Session = Depends(get_db)):
    """Return a time-bucketed series of decision counts for charting.

    Shows a rolling recent window so the dashboard reflects the current
    state instead of collapsing into coarse all-time buckets. Uses minute
    buckets for short active windows and hourly buckets for longer recent
    windows. Returns an empty series if no decisions have been recorded yet.

    Args:
        db: Injected database session.

    Returns:
        A dict with ``granularity`` (``"minute"`` or ``"hour"``),
        ``lookback_hours``, ``min`` and ``max`` ISO-8601 timestamps, and
        ``rows`` — a list of dicts with ``bucket``, ``decision_point``,
        and ``count``.
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

    recent_window_hours = ANALYTICS_TIMELINE_LOOKBACK_HOURS
    window_span_seconds = recent_window_hours * 3600
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    # Anchor the chart on the current period when traffic is recent; otherwise,
    # fall back to the most recent activity window so the chart is still useful.
    window_end_dt = now_utc if (now_utc - max_dt).total_seconds() <= window_span_seconds else max_dt
    window_start_dt = max(min_dt, window_end_dt - timedelta(hours=recent_window_hours))
    delta = (window_end_dt - window_start_dt).total_seconds()

    # Determine the Berlin UTC offset (handles CET/CEST DST automatically).
    berlin_offset_s = _to_berlin(window_start_dt).utcoffset().total_seconds()
    offset_hours = int(berlin_offset_s / 3600)
    sqlite_modifier = f"'{'+' if offset_hours >= 0 else ''}{offset_hours} hours'"

    if delta < 7200:          # < 2 h  → minute buckets
        bucket_sql = f"strftime('%Y-%m-%dT%H:%M', timestamp, {sqlite_modifier})"
        granularity = "minute"
    else:                     # rolling recent window → hourly
        bucket_sql = f"strftime('%Y-%m-%dT%H:00', timestamp, {sqlite_modifier})"
        granularity = "hour"

    rows = db.execute(
        text(
            f"""
            SELECT
                {bucket_sql} AS bucket,
                decision_point,
                COUNT(*) AS cnt
            FROM decision_logs
            WHERE timestamp >= :window_start_dt AND timestamp <= :window_end_dt
            GROUP BY bucket, decision_point
            ORDER BY bucket
            """
        ),
        {
            "window_start_dt": window_start_dt,
            "window_end_dt": window_end_dt,
        },
    ).fetchall()

    min_berlin = _to_berlin(window_start_dt)
    max_berlin = _to_berlin(window_end_dt)

    return {
        "granularity": granularity,
        "lookback_hours": recent_window_hours,
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
            "policy_source": _policy_source_label(r.context_json or {}),
            "estimates": (r.context_json or {}).get("estimates", {}),
        }
        for r in rows
    ]


@router.get("/api/analytics/policy_breakdown")
def api_policy_breakdown(db: Session = Depends(get_db)):
    """Return decision distribution by policy source and model version.

    Uses ``DecisionLog.context_json`` fields written during serving:
    ``policy_source`` and ``model``.
    """
    rows = db.query(DecisionLog).all()

    by_source = Counter()
    by_model = Counter()
    by_source_and_point = Counter()

    for row in rows:
        ctx = row.context_json or {}
        source = _policy_source_label(ctx)
        model = str(ctx.get("model") or "unknown")

        by_source[source] += 1
        by_model[model] += 1
        by_source_and_point[(source, row.decision_point)] += 1

    return {
        "total_decisions": len(rows),
        "by_source": [{"policy_source": k, "count": v} for k, v in sorted(by_source.items())],
        "by_model": [{"model": k, "count": v} for k, v in sorted(by_model.items())],
        "by_source_and_point": [
            {"policy_source": k[0], "decision_point": k[1], "count": v}
            for k, v in sorted(by_source_and_point.items())
        ],
    }


@router.get("/api/analytics/timing_outcomes")
def api_timing_outcomes(db: Session = Depends(get_db)):
    """Return timing-related outcomes for the hybrid opportunity flow.

    Combines opportunity lifecycle events (checked/skipped/served) with
    decision-level timing metadata from ``DecisionLog.context_json``.
    """
    events = (
        db.query(Event)
        .filter(
            Event.event_type.in_(
                [
                    "opportunity_checked",
                    "opportunity_skipped",
                    "opportunity_served",
                ]
            )
        )
        .all()
    )

    event_counts = Counter()
    skip_reasons = Counter()
    by_decision_point = Counter()

    defer_values = []
    for event in events:
        md = event.metadata_json or {}
        event_counts[event.event_type] += 1

        point = str(md.get("decision_point") or "unknown")
        by_decision_point[(event.event_type, point)] += 1

        if event.event_type == "opportunity_skipped":
            reason = str(md.get("reason_code") or "unknown")
            skip_reasons[reason] += 1

            try:
                defer_ms = int(md.get("next_check_after_ms") or 0)
                if defer_ms > 0:
                    defer_values.append(defer_ms)
            except (TypeError, ValueError):
                pass

    decision_rows = db.query(DecisionLog).all()
    timing_gate_reasons = Counter()
    for row in decision_rows:
        ctx = row.context_json or {}
        gate = ctx.get("timing_gate") or {}
        if isinstance(gate, dict):
            reason = str(gate.get("reason") or "unknown")
            timing_gate_reasons[reason] += 1

    avg_defer_ms = 0.0
    if defer_values:
        avg_defer_ms = sum(defer_values) / len(defer_values)

    return {
        "opportunity_event_counts": [
            {"event_type": k, "count": v} for k, v in sorted(event_counts.items())
        ],
        "opportunity_skip_reasons": [
            {"reason_code": k, "count": v} for k, v in sorted(skip_reasons.items())
        ],
        "opportunity_by_decision_point": [
            {"event_type": k[0], "decision_point": k[1], "count": v}
            for k, v in sorted(by_decision_point.items())
        ],
        "decision_timing_gate_reasons": [
            {"reason": k, "count": v} for k, v in sorted(timing_gate_reasons.items())
        ],
        "avg_next_check_after_ms": round(avg_defer_ms, 2),
    }


@router.get("/api/analytics/learner_status")
def api_learner_status():
    """Return background learner health, last-run metadata, and PPO checkpoint status."""
    status = get_learner_status()
    status["ppo"] = get_ppo_health_status()
    return status
