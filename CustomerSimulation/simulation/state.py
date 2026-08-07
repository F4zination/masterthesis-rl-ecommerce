import sys
from pathlib import Path

# Make SharedSchema importable without installation when running locally.
_repo_root = Path(__file__).resolve().parent.parent.parent
_shared_schema_path = _repo_root / "SharedSchema"
if str(_shared_schema_path) not in sys.path:
    sys.path.insert(0, str(_shared_schema_path))

from dataclasses import dataclass

from shared_schema.constants import CONTEXT_SCHEMA_VERSION, FEATURE_BUCKET_THRESHOLDS
from shared_schema.features import HISTORY_FEATURES, POINT_FEATURES, _bucket


@dataclass
class SimState:
    """Mutable session state tracked by the simulator."""

    decision_point: str
    device_type: str
    traffic_source: str
    # The browser increments sessionStorage from 0 before its first decision,
    # so a live landing-page decision observes page_depth=1.
    page_depth: int = 1
    item_count: int = 0
    cart_total: float = 0.0
    price: float = 0.0
    session_step: int = 0

    # --- Trajectory / history fields (V3 sequential dynamics) ---------------
    # These accumulate over the session and only surface in the emitted state
    # dict when ``to_state_dict(sequential=True)``.  In legacy mode they are
    # tracked but never observed, so adding them changes no emitted artifact.
    interventions_shown: int = 0      # cumulative non-no-op actions served
    steps_since_intervention: int = 0  # steps since the last widget was served
    recent_interventions: float = 0.0  # exponential-window intervention count
    primed_credit: float = 0.0         # latent assist/priming credit (mech. 2)

    def to_state_dict(self, sequential: bool = False) -> dict:
        """Return a state dict matching the production _normalize_context output.

        Only features listed in POINT_FEATURES for the current decision_point
        are included, so adding a new feature to POINT_FEATURES propagates here
        automatically — as long as the corresponding raw field exists on SimState.

        When ``sequential`` is true, bucketed trajectory-history features (the
        names in ``HISTORY_FEATURES``) are appended to the emitted dict but are
        *not* added to POINT_FEATURES.  PPO's encoder picks them up; the
        bandit's context_key ignores them (unless explicitly exposed via
        SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT).  This is the legitimate
        algorithm-class asymmetry the experiment isolates.
        """
        feature_values: dict = {
            "device_type": self.device_type,
            "traffic_source": self.traffic_source,
            "page_depth_bucket": _bucket(
                float(self.page_depth),
                FEATURE_BUCKET_THRESHOLDS["page_depth_bucket"],
            ),
            "cart_total_bucket": _bucket(
                self.cart_total,
                FEATURE_BUCKET_THRESHOLDS["cart_total_bucket"],
            ),
            "price_bucket": _bucket(
                self.price,
                FEATURE_BUCKET_THRESHOLDS["price_bucket"],
            ),
            "item_count_bucket": _bucket(
                float(self.item_count),
                FEATURE_BUCKET_THRESHOLDS["item_count_bucket"],
            ),
            "interventions_shown_bucket": _bucket(
                float(self.interventions_shown),
                FEATURE_BUCKET_THRESHOLDS["interventions_shown_bucket"],
            ),
            "steps_since_widget_bucket": _bucket(
                float(self.steps_since_intervention),
                FEATURE_BUCKET_THRESHOLDS["steps_since_widget_bucket"],
            ),
            "session_step_bucket": _bucket(
                float(self.session_step),
                FEATURE_BUCKET_THRESHOLDS["session_step_bucket"],
            ),
            "primed_credit_bucket": _bucket(
                float(self.primed_credit),
                FEATURE_BUCKET_THRESHOLDS["primed_credit_bucket"],
            ),
        }

        features = list(POINT_FEATURES.get(self.decision_point, []))
        state: dict = {
            "decision_point": self.decision_point,
            "schema_version": CONTEXT_SCHEMA_VERSION,
        }
        for f in features:
            state[f] = feature_values[f]

        if sequential:
            for f in HISTORY_FEATURES:
                # Avoid duplicating a feature already surfaced via POINT_FEATURES
                # (e.g. when the robustness env var exposes it to the bandit).
                if f not in state:
                    state[f] = feature_values[f]

        return state
