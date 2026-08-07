import copy
import math
import random as _random_module
import uuid
from datetime import datetime, timedelta

from .archetype import Archetype
from .behavior_policy import BehaviorPolicy
from .state import SimState

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_shared_schema_path = _repo_root / "SharedSchema"
if str(_shared_schema_path) not in sys.path:
    sys.path.insert(0, str(_shared_schema_path))

from shared_schema.constants import (
    ACTION_COST,
    DEVICE_TYPES,
    HISTORY_PRIMED_CREDIT_DECAY,
    HISTORY_PRIMING_ACTIONS,
    HISTORY_PRIMING_AMOUNT,
    TRAFFIC_SOURCES,
)
from shared_schema.features import _eligible_actions, _event_reward


# ---------------------------------------------------------------------------
# Sequential-dynamics configuration
# ---------------------------------------------------------------------------
# Every knob defaults to off so the baseline reproduces today's near-tie
# (the pre-registration anchor, H0).  When a knob is off it must not consume
# any extra RNG draws relative to the legacy simulator, so legacy runs remain
# byte-for-byte reproducible.  See V3_sequential_environment_plan.md §4.
DEFAULT_DYNAMICS: dict = {
    # Mechanism 1 — intervention fatigue
    "fatigue_rate": 0.0,        # 0 = off; widget response decays as exp(-rate*k)
    "fatigue_window": 0,        # 0 = cumulative count k; N = exponential window
    "p_dismiss_max": 0.6,       # ceiling that dismiss probability rises toward
    # Mechanism 2 — delayed / assist reward.  The credit bookkeeping constants
    # come from shared_schema so DemoSiteV3's serving-time primed_credit
    # computation uses identical values (single source of truth).
    "delayed_reward_strength": 0.0,   # 0 = off; purchase prob += strength*credit
    "delayed_reward_decay": HISTORY_PRIMED_CREDIT_DECAY,  # per-step decay of latent primed credit
    "priming_actions": list(HISTORY_PRIMING_ACTIONS),
    "priming_amount": HISTORY_PRIMING_AMOUNT,  # credit accrued each time a priming action serves
    "cash_out_point": "checkout",     # decision point where credit cashes out
    # Mechanism 3 — action-dependent transitions
    "transition_coupling_strength": 0.0,  # 0 = off; scales action_stage_lifts
    # Robustness check (handled in shared_schema.features via env var)
    "expose_history_to_bandit": False,
}


def merge_dynamics(dynamics: dict | None) -> dict:
    """Merge a partial dynamics dict over DEFAULT_DYNAMICS."""
    merged = dict(DEFAULT_DYNAMICS)
    if dynamics:
        for key, value in dynamics.items():
            if value is not None:
                merged[key] = value
    return merged


class SessionSimulator:
    """Generates one simulated session as a list of transition dicts.

    Args:
        dynamics: Optional sequential-dynamics knob dict (merged over
            ``DEFAULT_DYNAMICS``).  All knobs default off → legacy behaviour.
        sequential: When true, ``SimState.to_state_dict`` appends bucketed
            trajectory-history features (read by PPO, ignored by the bandit).
    """

    def __init__(self, dynamics: dict | None = None, sequential: bool = False) -> None:
        self.dynamics = merge_dynamics(dynamics)
        self.sequential = sequential

    def simulate_session(
        self,
        archetype: Archetype,
        policy: BehaviorPolicy,
        t_max: int = 20,
        rng: _random_module.Random | None = None,
    ) -> list[dict]:
        rand = rng or _random_module

        session_id = f"sim_{uuid.uuid4().hex}"
        # Deliberately stress every finite serving value uniformly, including
        # the missing-width fallback ``unknown``. Device type has no causal
        # effect in the simulator, so this is support coverage rather than an
        # estimate of production device prevalence.
        device_type = rand.choice(DEVICE_TYPES)
        traffic_source = rand.choice(TRAFFIC_SOURCES)

        state = SimState(
            decision_point="landing",
            device_type=device_type,
            traffic_source=traffic_source,
            price=rand.uniform(5.0, 200.0),
        )

        transitions: list[dict] = []
        base_ts = datetime.utcnow()
        has_set_state = hasattr(policy, "set_state")

        for t in range(t_max):
            dp = state.decision_point
            eligible = _eligible_actions(dp)

            if has_set_state:
                policy.set_state(state.to_state_dict(sequential=self.sequential))

            action, propensity = policy.select_action(eligible, archetype.name, rand)

            events, order_total = self._sample_events(state, action, archetype, rand)
            next_stage = self._sample_next_stage(dp, archetype, events, rand, action)
            events = self._ensure_cart_reachability_events(state, next_stage, events)

            reward_without_cost = self._compute_reward(events, order_total)
            action_cost = ACTION_COST.get(action, 0.0)
            reward = reward_without_cost - action_cost

            done = next_stage == "done" or t == t_max - 1
            next_state = self._make_next_state(state, next_stage, events, archetype, rand, action)

            transitions.append({
                "trajectory_id": session_id,
                "t": t,
                "decision_id": t,
                "timestamp": (base_ts + timedelta(seconds=t * 30)).isoformat(),
                "session_id": session_id,
                "decision_point": dp,
                "state": state.to_state_dict(sequential=self.sequential),
                "action": action,
                "propensity": propensity,
                "eligible_actions": eligible,
                "reward": reward,
                "reward_without_cost": reward_without_cost,
                "action_cost": action_cost,
                "next_state": next_state.to_state_dict(sequential=self.sequential),
                "done": done,
                "metadata": {
                    "user_type": archetype.name,
                    "generated_events": events,
                    "order_total_if_any": order_total if "purchase" in events else None,
                },
            })

            if done:
                break

            state = next_state

        return transitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_events(
        self,
        state: SimState,
        action: str,
        archetype: Archetype,
        rand: _random_module.Random,
    ) -> tuple[list[str], float]:
        events: list[str] = []
        dp = state.decision_point

        fatigue_rate = self.dynamics["fatigue_rate"]
        # Effective over-exposure count (cumulative or exponential window).
        if self.dynamics["fatigue_window"] > 0:
            k = state.recent_interventions
        else:
            k = float(state.interventions_shown)
        # decay == 1.0 when fatigue is off (rate==0) → identical to legacy.
        decay = math.exp(-fatigue_rate * k) if fatigue_rate > 0 else 1.0

        # Widget interaction — mutually exclusive click / dismiss
        if action != "no-op":
            widget_probs = archetype.widget_events.get(action, {})
            p_click = widget_probs.get("widget_click", 0.0)
            p_dismiss = widget_probs.get("widget_dismiss", 0.0)
            if decay < 1.0:
                p_dismiss_max = self.dynamics["p_dismiss_max"]
                p_click = p_click * decay
                # Annoyance: dismiss probability rises toward its ceiling.
                p_dismiss = p_dismiss + (p_dismiss_max - p_dismiss) * (1.0 - decay)
            r = rand.random()
            if r < p_click:
                events.append("widget_click")
            elif r < p_click + p_dismiss:
                events.append("widget_dismiss")

        # Base events with additive action lifts, clamped to [0, 1].
        # A lift may only *modulate* an event that is actually possible at the
        # current decision point (i.e. already present in base_events[dp]); it
        # must not manufacture an impossible event — e.g. trust_badge's
        # checkout_submit lift must not fire a +2.0 "checkout submit" on the
        # landing page.  Funnel-progression effects of an intervention belong to
        # the action-dependent transition mechanism, not to immediate reward.
        base = dict(archetype.base_events.get(dp, {}))
        for evt, lift in archetype.action_event_lifts.get(action, {}).items():
            if evt not in base:
                continue
            # Lifts also fade with fatigue (decay == 1.0 → unchanged).
            base[evt] = max(0.0, min(1.0, base[evt] + lift * decay))

        # Mechanism 2 — delayed/assist reward cashes out at the cash-out point.
        # Only the purchase *probability* shifts; the reward table is untouched,
        # so the delayed payoff is endogenous (earlier priming caused it).
        delayed_strength = self.dynamics["delayed_reward_strength"]
        if delayed_strength > 0 and dp == self.dynamics["cash_out_point"] and "purchase" in base:
            boost = delayed_strength * state.primed_credit
            base["purchase"] = max(0.0, min(1.0, base["purchase"] + boost))

        order_total = 0.0
        for event_type, prob in base.items():
            if rand.random() < prob:
                events.append(event_type)
                if event_type == "purchase":
                    order_total = rand.uniform(
                        archetype.order_total_range[0],
                        archetype.order_total_range[1],
                    )

        return events, order_total

    def _compute_reward(self, events: list[str], order_total: float) -> float:
        total = 0.0
        for evt in events:
            if evt == "purchase":
                total += _event_reward(evt, {"order_total": order_total})
            elif evt == "scroll_depth":
                total += _event_reward(evt, {"depth": 75})
            elif evt == "dwell_time":
                total += _event_reward(evt, {"seconds": 45})
            else:
                total += _event_reward(evt, {})
        return total

    @staticmethod
    def _ensure_cart_reachability_events(
        state: SimState,
        next_stage: str,
        events: list[str],
    ) -> list[str]:
        """Make an empty-cart transition agree with production reachability.

        Production emits ``cart`` and ``checkout`` decision points only when
        the cart contains an item. A sampled transition to either point from
        an empty simulated cart therefore represents an implicit add-to-cart.
        Materialise that event before reward calculation so both the resulting
        state and the transition reward describe the same user action.
        """
        if (
            next_stage in {"cart", "checkout"}
            and state.item_count <= 0
            and "add_to_cart" not in events
        ):
            return [*events, "add_to_cart"]
        return events

    def _sample_next_stage(
        self,
        dp: str,
        archetype: Archetype,
        events: list[str],
        rand: _random_module.Random,
        action: str = "no-op",
    ) -> str:
        if "purchase" in events:
            return "done"

        transitions = archetype.stage_transitions.get(dp, {"done": 1.0})
        stages = list(transitions.keys())

        # Mechanism 3 — actions steer the funnel.  Additive lifts on the
        # next-stage weights, scaled by the global coupling strength, clamped
        # to >= 0.  rand.choices renormalises internally.
        strength = self.dynamics["transition_coupling_strength"]
        if strength > 0:
            lifts = archetype.action_stage_lifts.get(action, {})
            if lifts:
                weights = [max(0.0, transitions[s] + strength * lifts.get(s, 0.0)) for s in stages]
                if sum(weights) <= 0:
                    weights = [transitions[s] for s in stages]
            else:
                weights = [transitions[s] for s in stages]
        else:
            weights = [transitions[s] for s in stages]

        return rand.choices(stages, weights=weights, k=1)[0]

    def _make_next_state(
        self,
        state: SimState,
        next_stage: str,
        events: list[str],
        archetype: Archetype,
        rand: _random_module.Random,
        action: str = "no-op",
    ) -> SimState:
        if (
            next_stage in {"cart", "checkout"}
            and state.item_count <= 0
            and "add_to_cart" not in events
        ):
            raise ValueError(
                "Entering cart/checkout with an empty cart requires an "
                "add_to_cart event; call _ensure_cart_reachability_events "
                "before computing reward and next state."
            )

        next_state = copy.copy(state)

        if "add_to_cart" in events:
            next_state.item_count += 1
            next_state.cart_total += state.price if state.price > 0 else rand.uniform(15.0, 80.0)

        # A scroll-engagement opportunity is emitted on the current page and
        # does not increment the browser's page_depth. Funnel navigation does;
        # terminating a session does not load a new page.
        if next_stage not in {"scroll_engagement", "done"}:
            next_state.page_depth += 1
        next_state.session_step += 1

        # --- Trajectory bookkeeping (Mechanism 1 counters) ------------------
        served = 1 if action != "no-op" else 0
        next_state.interventions_shown += served
        next_state.steps_since_intervention = 0 if served else state.steps_since_intervention + 1
        window = self.dynamics["fatigue_window"]
        if window > 0:
            alpha = 1.0 / float(window)
            next_state.recent_interventions = state.recent_interventions * (1.0 - alpha) + served

        # --- Latent assist credit / observed action history -----------------
        # Production reconstructs this history from served actions regardless
        # of whether delayed reward is enabled. The strength knob above controls
        # only whether the credit affects purchase probability.
        next_state.primed_credit = state.primed_credit * self.dynamics["delayed_reward_decay"]
        if action in self.dynamics["priming_actions"]:
            next_state.primed_credit += self.dynamics["priming_amount"]

        if next_stage != "done":
            next_state.decision_point = next_stage

        # Refresh product price when navigating to a PDP
        if next_stage == "pdp":
            next_state.price = rand.uniform(5.0, 200.0)

        return next_state
