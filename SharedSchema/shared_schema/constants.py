# ---------------------------------------------------------------------------
# Scalar constants and reward/cost lookup tables shared across DemoSite
# versions and CustomerSimulation.  No external dependencies.
# ---------------------------------------------------------------------------

from typing import Literal, TypeAlias, get_args


EPSILON: float = 0.2
PRIOR_COUNT: float = 5.0
PRIOR_MEAN: float = 0.0
LOOKBACK_MINUTES: int = 30
CONTEXT_SCHEMA_VERSION: int = 2

# ---------------------------------------------------------------------------
# Finite serving-state domains
# ---------------------------------------------------------------------------
# These values are the complete set of categorical tokens that production can
# emit. Training and auditing code should consume this contract rather than
# infer a vocabulary only from values that happened to occur in one dataset.
DecisionPointName: TypeAlias = Literal[
    "landing",
    "pdp",
    "cart",
    "checkout",
    "scroll_engagement",
]
DECISION_POINTS: tuple[str, ...] = get_args(DecisionPointName)
DEVICE_TYPES: tuple[str, ...] = ("unknown", "mobile", "tablet", "desktop")
TRAFFIC_SOURCES: tuple[str, ...] = ("direct", "search", "social", "referral")

# A bucket feature with N thresholds has N + 1 possible integer values.
FEATURE_BUCKET_THRESHOLDS: dict[str, tuple[float, ...]] = {
    "page_depth_bucket": (1.0, 3.0, 6.0),
    "cart_total_bucket": (25.0, 80.0, 150.0),
    "price_bucket": (20.0, 60.0, 120.0),
    "item_count_bucket": (1.0, 3.0, 6.0),
    "interventions_shown_bucket": (1.0, 3.0, 6.0),
    "steps_since_widget_bucket": (1.0, 3.0, 6.0),
    "session_step_bucket": (3.0, 6.0, 12.0),
    "primed_credit_bucket": (0.5, 1.5, 3.0),
}

FEATURE_VALUE_DOMAINS: dict[str, tuple[str | int, ...]] = {
    "schema_version": (CONTEXT_SCHEMA_VERSION,),
    "decision_point": DECISION_POINTS,
    "device_type": DEVICE_TYPES,
    "traffic_source": TRAFFIC_SOURCES,
    **{
        feature: tuple(range(len(thresholds) + 1))
        for feature, thresholds in FEATURE_BUCKET_THRESHOLDS.items()
    },
}

# Action cost discourages over-serving intervention widgets.
ACTION_COST: dict[str, float] = {
    "no-op": 0.0,
    "trending_carousel": 0.05,
    "discount_banner": 0.1,
    "frequently_bought_together": 0.04,
    "trust_badge": 0.03,
    "help_popup": 0.02,
}

ALL_ACTIONS: list[str] = list(ACTION_COST.keys())

EVENT_REWARD: dict[str, float] = {
    "widget_click": 1.2,
    "widget_dismiss": -0.8,
    "add_to_cart": 1.5,
    "remove_from_cart": -0.5,
    "checkout_submit": 2.0,
    "purchase": 8.0,
    "exit_intent": -0.6,
    # Legacy: neither the frontend nor the simulator ever emits this event
    # type — help-popup interactions are rewarded as generic "widget_click".
    # Kept only so historical logs containing it would still score.
    "help_popup_click": 1.0,
}

# ---------------------------------------------------------------------------
# History-feature computation constants (V3 sequential state).
# Single source of truth for how the ``primed_credit`` observation is derived
# from the served-action history.  ``SessionSimulator`` (training) and the
# DemoSiteV3 decision service (serving) must use identical values so the
# feature means the same thing in both places.
# ---------------------------------------------------------------------------
HISTORY_PRIMING_ACTIONS: list[str] = ["help_popup", "trust_badge"]
HISTORY_PRIMING_AMOUNT: float = 1.0
HISTORY_PRIMED_CREDIT_DECAY: float = 0.9
