import random
import sys
import unittest
from pathlib import Path


CUSTOMER_SIMULATION = Path(__file__).resolve().parents[1]
SHARED_SCHEMA = CUSTOMER_SIMULATION.parent / "SharedSchema"
sys.path.insert(0, str(CUSTOMER_SIMULATION))
sys.path.insert(0, str(SHARED_SCHEMA))

from shared_schema.constants import (  # noqa: E402
    DEVICE_TYPES,
    FEATURE_BUCKET_THRESHOLDS,
    FEATURE_VALUE_DOMAINS,
)
from shared_schema.features import _history_from_actions  # noqa: E402
from simulation.archetype import Archetype  # noqa: E402
from simulation.simulator import SessionSimulator  # noqa: E402
from simulation.state import SimState  # noqa: E402


class _AlwaysNoOpPolicy:
    def select_action(self, eligible_actions, archetype_name, rng):
        return "no-op", 1.0


class _UnknownDeviceRandom(random.Random):
    def choice(self, sequence):
        if "unknown" in sequence:
            return "unknown"
        return super().choice(sequence)


class _FixedRandom(random.Random):
    def random(self):
        return 0.5

    def uniform(self, lower, upper):
        return lower


def _archetype(
    *,
    stage_transitions: dict[str, dict[str, float]] | None = None,
    base_events: dict[str, dict[str, float]] | None = None,
) -> Archetype:
    return Archetype(
        name="Test",
        stage_transitions=stage_transitions or {"landing": {"done": 1.0}},
        base_events=base_events or {},
        widget_events={},
        action_event_lifts={},
        order_total_range=[40.0, 80.0],
    )


class StateSemanticAlignmentTests(unittest.TestCase):
    def test_shared_domain_covers_every_finite_serving_token(self) -> None:
        self.assertEqual(
            FEATURE_VALUE_DOMAINS["device_type"],
            ("unknown", "mobile", "tablet", "desktop"),
        )
        self.assertEqual(sum(map(len, FEATURE_VALUE_DOMAINS.values())), 46)
        for feature, thresholds in FEATURE_BUCKET_THRESHOLDS.items():
            self.assertEqual(
                FEATURE_VALUE_DOMAINS[feature],
                tuple(range(len(thresholds) + 1)),
            )

    def test_simulator_can_sample_unknown_and_starts_at_browser_page_depth(self) -> None:
        transitions = SessionSimulator().simulate_session(
            _archetype(),
            _AlwaysNoOpPolicy(),
            t_max=1,
            rng=_UnknownDeviceRandom(7),
        )

        self.assertIn("unknown", DEVICE_TYPES)
        self.assertEqual(transitions[0]["state"]["device_type"], "unknown")
        # Browser page_depth starts at 1, which falls in bucket 1.
        self.assertEqual(transitions[0]["state"]["page_depth_bucket"], 1)

    def test_scroll_opportunity_does_not_increment_page_depth(self) -> None:
        simulator = SessionSimulator()
        archetype = _archetype()
        rng = random.Random(1)

        for decision_point, depth in (("landing", 1), ("pdp", 5)):
            with self.subTest(decision_point=decision_point):
                state = SimState(
                    decision_point=decision_point,
                    device_type="desktop",
                    traffic_source="direct",
                    page_depth=depth,
                )
                next_state = simulator._make_next_state(
                    state,
                    "scroll_engagement",
                    [],
                    archetype,
                    rng,
                )
                self.assertEqual(next_state.page_depth, depth)

    def test_navigation_after_scroll_increments_page_depth(self) -> None:
        state = SimState(
            decision_point="scroll_engagement",
            device_type="desktop",
            traffic_source="direct",
            page_depth=2,
        )
        next_state = SessionSimulator()._make_next_state(
            state,
            "pdp",
            [],
            _archetype(),
            random.Random(1),
        )
        self.assertEqual(next_state.page_depth, 3)

    def test_primed_credit_history_updates_when_reward_effect_is_off(self) -> None:
        simulator = SessionSimulator(dynamics={"delayed_reward_strength": 0.0})
        archetype = _archetype()
        state = SimState("landing", "desktop", "direct")

        primed = simulator._make_next_state(
            state,
            "pdp",
            [],
            archetype,
            random.Random(1),
            "help_popup",
        )
        decayed = simulator._make_next_state(
            primed,
            "pdp",
            [],
            archetype,
            random.Random(2),
            "no-op",
        )

        serving_history = _history_from_actions(["help_popup", "no-op"])
        self.assertAlmostEqual(primed.primed_credit, 1.0)
        self.assertAlmostEqual(decayed.primed_credit, 0.9)
        self.assertAlmostEqual(decayed.primed_credit, serving_history["primed_credit"])

    def test_delayed_reward_strength_alone_controls_purchase_effect(self) -> None:
        archetype = _archetype(base_events={"checkout": {"purchase": 0.0}})
        state = SimState(
            decision_point="checkout",
            device_type="desktop",
            traffic_source="direct",
            item_count=1,
            cart_total=40.0,
            primed_credit=1.0,
        )

        off_events, _ = SessionSimulator(
            dynamics={"delayed_reward_strength": 0.0}
        )._sample_events(state, "no-op", archetype, _FixedRandom())
        on_events, _ = SessionSimulator(
            dynamics={"delayed_reward_strength": 1.0}
        )._sample_events(state, "no-op", archetype, _FixedRandom())

        self.assertNotIn("purchase", off_events)
        self.assertIn("purchase", on_events)

    def test_cart_and_checkout_transitions_materialize_add_event_before_reward(self) -> None:
        for target in ("cart", "checkout"):
            with self.subTest(target=target):
                archetype = _archetype(
                    stage_transitions={
                        "landing": {target: 1.0},
                        target: {"done": 1.0},
                    }
                )
                transitions = SessionSimulator().simulate_session(
                    archetype,
                    _AlwaysNoOpPolicy(),
                    t_max=2,
                    rng=random.Random(3),
                )

                first = transitions[0]
                self.assertIn("add_to_cart", first["metadata"]["generated_events"])
                self.assertEqual(first["reward_without_cost"], 1.5)
                self.assertGreater(first["next_state"]["item_count_bucket"], 0)

                target_states = [
                    transition["state"]
                    for transition in transitions
                    if transition["decision_point"] == target
                ]
                self.assertTrue(target_states)
                self.assertTrue(
                    all(state["item_count_bucket"] > 0 for state in target_states)
                )

    def test_empty_cart_transition_requires_reachability_alignment(self) -> None:
        state = SimState("pdp", "desktop", "direct")
        with self.assertRaisesRegex(ValueError, "_ensure_cart_reachability_events"):
            SessionSimulator()._make_next_state(
                state,
                "cart",
                [],
                _archetype(),
                random.Random(1),
            )


if __name__ == "__main__":
    unittest.main()
