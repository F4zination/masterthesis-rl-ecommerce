# ---------------------------------------------------------------------------
# Feature schema: decision-point feature lists, action availability, pure
# feature-extraction helpers and reward calculation.  Imports only from
# shared_schema.constants — no external dependencies.
# ---------------------------------------------------------------------------

import os
from collections.abc import Sequence

from .constants import (
    ALL_ACTIONS,
    CONTEXT_SCHEMA_VERSION,
    EVENT_REWARD,
    FEATURE_BUCKET_THRESHOLDS,
    HISTORY_PRIMED_CREDIT_DECAY,
    HISTORY_PRIMING_ACTIONS,
    HISTORY_PRIMING_AMOUNT,
)

# ---------------------------------------------------------------------------
# Per-decision-point feature schema (v2)
# ---------------------------------------------------------------------------
# Maps each decision point to the ordered list of feature names that are
# meaningful at that location.  Only listed features are included in the
# context_key, preventing spurious state-space explosion from irrelevant
# dimensions.
#
# To add a new feature:
#   1. Register an extractor in _FEATURE_EXTRACTORS below.
#   2. Add its name to the relevant entries in POINT_FEATURES.
#   3. Run: python -m shared_schema.migrations run --db-path /path/to/db.
#
POINT_FEATURES: dict[str, list[str]] = {
    "landing": ["device_type", "traffic_source", "page_depth_bucket"],
    "pdp": ["device_type", "traffic_source", "page_depth_bucket", "price_bucket"],
    "cart": ["device_type", "traffic_source", "cart_total_bucket", "item_count_bucket"],
    "checkout": ["device_type", "traffic_source", "cart_total_bucket", "item_count_bucket"],
    "scroll_engagement": ["device_type", "page_depth_bucket"],
}

POINT_ACTIONS: dict[str, list[str]] = {
    "landing": ALL_ACTIONS.copy(),
    "pdp": ALL_ACTIONS.copy(),
    "cart": ALL_ACTIONS.copy(),
    "checkout": ALL_ACTIONS.copy(),
    "scroll_engagement": ALL_ACTIONS.copy(),
}

# ---------------------------------------------------------------------------
# Sequential / trajectory history features (V3 sequential mode only)
# ---------------------------------------------------------------------------
# These bucketed summaries of the accumulated trajectory are emitted by
# ``SimState.to_state_dict(sequential=True)`` but are deliberately *absent*
# from ``POINT_FEATURES``.  Consequence (see V3_sequential_environment_plan §3):
#   * the V2 bandit keys its context only on ``POINT_FEATURES`` → it ignores
#     them, staying myopic;
#   * V3 PPO encodes every key in the emitted state dict → it conditions on
#     the trajectory automatically.
# Both policies therefore see identical per-decision raw context and identical
# reward; the only difference is the *temporal* information available to the
# sequential learner — the legitimate asymmetry the experiment isolates.
HISTORY_FEATURES: list[str] = [
    "interventions_shown_bucket",
    "steps_since_widget_bucket",
    "session_step_bucket",
    "primed_credit_bucket",
]


def _history_from_actions(actions: list[str]) -> dict:
    """Fold a session's served-action history into the raw history values.

    Replays the same per-step recurrences ``SimState`` applies in
    ``CustomerSimulation/simulation/simulator.py::_make_next_state``, so a
    production service reconstructing history from its decision log produces
    values with identical semantics to the simulator states the sequential
    policy was trained on:

    * ``session_step``             — one increment per prior decision;
    * ``interventions_shown``      — count of prior non-``no-op`` actions;
    * ``steps_since_intervention`` — 0 right after a widget, else +1 per step
      (equals ``session_step`` when no widget was ever served);
    * ``primed_credit``            — decays by ``HISTORY_PRIMED_CREDIT_DECAY``
      each step and gains ``HISTORY_PRIMING_AMOUNT`` when a priming action
      (``HISTORY_PRIMING_ACTIONS``) was served.
    """
    interventions_shown = 0
    steps_since_intervention = 0
    primed_credit = 0.0
    for action in actions:
        if action != "no-op":
            interventions_shown += 1
            steps_since_intervention = 0
        else:
            steps_since_intervention += 1
        primed_credit *= HISTORY_PRIMED_CREDIT_DECAY
        if action in HISTORY_PRIMING_ACTIONS:
            primed_credit += HISTORY_PRIMING_AMOUNT

    return {
        "session_step": len(actions),
        "interventions_shown": interventions_shown,
        "steps_since_intervention": steps_since_intervention,
        "primed_credit": primed_credit,
    }


def _device_type(context: dict) -> str:
    """Classify the visitor's device type based on reported screen width.

    Args:
        context: Request context dict; may contain ``screen_width`` (int or str).

    Returns:
        One of ``"mobile"`` (< 768 px), ``"tablet"`` (< 1024 px),
        ``"desktop"`` (≥ 1024 px), or ``"unknown"`` if width is absent.
    """
    width = int(context.get("screen_width") or 0)
    if width and width < 768:
        return "mobile"
    if width and width < 1024:
        return "tablet"
    if width:
        return "desktop"
    return "unknown"


def _traffic_source(context: dict) -> str:
    """Infer the traffic source from the HTTP referrer stored in the context.

    Args:
        context: Request context dict; may contain ``referrer`` (str).

    Returns:
        One of ``"direct"`` (no referrer), ``"search"`` (major search engines),
        ``"social"`` (major social platforms), or ``"referral"`` (any other URL).
    """
    ref = str(context.get("referrer") or "").lower()
    if not ref:
        return "direct"
    if "google" in ref or "bing" in ref or "duckduckgo" in ref:
        return "search"
    if "facebook" in ref or "instagram" in ref or "linkedin" in ref or "tiktok" in ref:
        return "social"
    return "referral"


def _bucket(value: float, thresholds: Sequence[float]) -> int:
    """Map a continuous value to a discrete bucket index.

    Returns the index of the first threshold that exceeds ``value``, or
    ``len(thresholds)`` if the value is larger than all thresholds.

    Args:
        value: The numeric value to categorise.
        thresholds: Ascending list of boundary values.

    Returns:
        Zero-based bucket index in the range ``[0, len(thresholds)]``.
    """
    for idx, threshold in enumerate(thresholds):
        if value < threshold:
            return idx
    return len(thresholds)


# ---------------------------------------------------------------------------
# Feature extractor registry
# ---------------------------------------------------------------------------
# Maps feature names to callables ``(ctx: dict) -> str | int``.  To add a
# new feature, register it here and reference it in POINT_FEATURES above.
#
_FEATURE_EXTRACTORS: dict[str, callable] = {
    "device_type": _device_type,
    "traffic_source": _traffic_source,
    "page_depth_bucket": lambda ctx: _bucket(
        float(ctx.get("page_depth") or 0),
        FEATURE_BUCKET_THRESHOLDS["page_depth_bucket"],
    ),
    "cart_total_bucket": lambda ctx: _bucket(
        float(ctx.get("cart_total") or 0),
        FEATURE_BUCKET_THRESHOLDS["cart_total_bucket"],
    ),
    "price_bucket": lambda ctx: _bucket(
        float(ctx.get("price") or 0),
        FEATURE_BUCKET_THRESHOLDS["price_bucket"],
    ),
    "item_count_bucket": lambda ctx: _bucket(
        float(ctx.get("item_count") or 0),
        FEATURE_BUCKET_THRESHOLDS["item_count_bucket"],
    ),
    # Sequential history extractors (used only when the bucket name is present
    # in POINT_FEATURES — see SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT below).
    "interventions_shown_bucket": lambda ctx: _bucket(
        float(ctx.get("interventions_shown") or 0),
        FEATURE_BUCKET_THRESHOLDS["interventions_shown_bucket"],
    ),
    "steps_since_widget_bucket": lambda ctx: _bucket(
        float(ctx.get("steps_since_intervention") or 0),
        FEATURE_BUCKET_THRESHOLDS["steps_since_widget_bucket"],
    ),
    "session_step_bucket": lambda ctx: _bucket(
        float(ctx.get("session_step") or 0),
        FEATURE_BUCKET_THRESHOLDS["session_step_bucket"],
    ),
    "primed_credit_bucket": lambda ctx: _bucket(
        float(ctx.get("primed_credit") or 0),
        FEATURE_BUCKET_THRESHOLDS["primed_credit_bucket"],
    ),
}


# ---------------------------------------------------------------------------
# Robustness check (V3_sequential_environment_plan §7): when this env var is
# truthy, expose the cumulative-fatigue counter ``interventions_shown_bucket``
# to the *bandit* by appending it to every POINT_FEATURES entry.  If PPO still
# beats the bandit with this on, the win is genuine temporal credit assignment
# rather than mere information asymmetry.  Off by default so the headline runs
# preserve the legitimate asymmetry and reproduce the legacy artifacts exactly.
def _expose_history_to_bandit() -> bool:
    return os.environ.get("SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


if _expose_history_to_bandit():
    for _dp in POINT_FEATURES:
        if "interventions_shown_bucket" not in POINT_FEATURES[_dp]:
            POINT_FEATURES[_dp].append("interventions_shown_bucket")


def _normalize_context(
    decision_point: str,
    context: dict | None,
    include_history: bool = False,
) -> tuple[str, dict]:
    """Normalise a raw request context into a per-decision-point feature vector.

    Only the features listed in ``POINT_FEATURES`` for the given
    ``decision_point`` are extracted and included in the ``context_key``.
    This prevents irrelevant dimensions from inflating the arm state-space.

    Falls back to all registered features for unrecognised decision points.

    Args:
        decision_point: Identifier for the location in the user journey
            (e.g., ``"landing"``, ``"pdp"``, ``"cart"``).
        context: Raw context dictionary supplied by the client, or ``None``.
        include_history: When true, additionally extract the bucketed
            ``HISTORY_FEATURES`` into the normalized dict — but *not* into
            the ``context_key``.  This mirrors
            ``SimState.to_state_dict(sequential=True)``: the sequential
            policy (PPO/FQI) conditions on trajectory history while the
            bandit's arm key stays myopic.  The raw history values
            (``interventions_shown``, ``steps_since_intervention``,
            ``session_step``, ``primed_credit``) must be present in
            ``context`` for the buckets to be meaningful.

    Returns:
        A tuple ``(context_key, normalized_context)`` where ``context_key``
        is a pipe-delimited string (``"<point>|<f1>|<f2>|..."``) containing
        only the relevant features, and ``normalized_context`` is a dict of
        the extracted feature values plus ``"schema_version"``.
    """
    ctx = context or {}
    features = POINT_FEATURES.get(decision_point, list(_FEATURE_EXTRACTORS.keys()))

    normalized: dict = {"decision_point": decision_point, "schema_version": CONTEXT_SCHEMA_VERSION}
    for feature in features:
        normalized[feature] = _FEATURE_EXTRACTORS[feature](ctx)

    context_key = "|".join([decision_point] + [str(normalized[f]) for f in features])

    if include_history:
        for feature in HISTORY_FEATURES:
            if feature not in normalized:
                normalized[feature] = _FEATURE_EXTRACTORS[feature](ctx)

    return context_key, normalized


def _eligible_actions(decision_point: str) -> list[str]:
    """Return the list of valid widget actions for a given decision point.

    Args:
        decision_point: The step in the purchase funnel (e.g., ``"cart"``).

    Returns:
        List of action strings defined in ``POINT_ACTIONS``; falls back to
        ``ALL_ACTIONS`` for unrecognised decision points.
    """
    return POINT_ACTIONS.get(decision_point, ALL_ACTIONS)


def _decision_point_from_page(page: str) -> str | None:
    """Map a URL path to the corresponding bandit decision point.

    Used to infer which decision point should receive a reward when an event
    does not carry an explicit ``decision_point`` in its metadata.

    Args:
        page: The URL path of the current page (e.g., ``"/product/my-item"``).

    Returns:
        A decision point string such as ``"pdp"``, or ``None`` if the page
        does not correspond to any tracked decision point.
    """
    if page == "/":
        return "landing"
    if page.startswith("/product/"):
        return "pdp"
    if page == "/cart":
        return "cart"
    if page == "/checkout":
        return "checkout"
    return None


def _event_reward(event_type: str, metadata: dict | None) -> float:
    """Calculate the reward signal for a user interaction event.

    Base rewards are looked up from ``EVENT_REWARD``.  Bonus rewards are
    applied for meaningful scroll depth (≥ 50 %), long dwell time (≥ 30 s),
    and high-value purchases (scaled by order total, capped at +4.0).

    Args:
        event_type: Semantic label for the interaction (e.g., ``"add_to_cart"``).
        metadata: Optional dict of event-specific data; relevant keys include
            ``depth`` (int, percent), ``seconds`` (int), and ``order_total`` (float).

    Returns:
        Scalar reward value; returns ``0.0`` for unrecognised event types.
    """
    reward = EVENT_REWARD.get(event_type, 0.0)
    md = metadata or {}

    if event_type == "scroll_depth":
        depth = int(md.get("depth") or 0)
        if depth >= 50:
            return 0.2
    if event_type == "dwell_time":
        seconds = int(md.get("seconds") or 0)
        if seconds >= 30:
            return 0.2
    if event_type == "purchase":
        total = float(md.get("order_total") or 0)
        reward += min(total / 100.0, 4.0)

    return reward
