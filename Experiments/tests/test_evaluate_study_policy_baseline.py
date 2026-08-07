import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "Experiments"
sys.path.insert(0, str(EXPERIMENTS))

from evaluate_study_policy_baseline import (  # noqa: E402
    DEVICE_STRATA,
    FrozenBanditPolicy,
    TRAFFIC_STRATA,
    _csv_rows,
    _funnel_depth_histogram,
    _funnel_progression,
    _production_history_buckets,
    _session_funnel_depth,
    derive_archetype_seeds,
    derive_context_seed,
    evaluate_cell,
    write_outputs,
)
from simulation.archetype import load_archetypes  # noqa: E402
from simulation.simulator import merge_dynamics  # noqa: E402
from shared_schema.constants import (  # noqa: E402
    ALL_ACTIONS,
    DEVICE_TYPES,
    TRAFFIC_SOURCES,
)


class _AlwaysNoOpPolicy:
    def reset_diagnostics(self) -> None:
        self.decisions = 0
        self.states = []

    def begin_session(self) -> None:
        pass

    def set_state(self, state: dict) -> None:
        self.state = state
        self.states.append(dict(state))

    def select_action(self, eligible_actions, archetype_name=None, rng=None):
        self.decisions += 1
        return "no-op", 1.0

    def diagnostics(self) -> dict:
        return {"decisions": self.decisions, "fallback_decisions": 0}


class StudyBaselineTests(unittest.TestCase):
    def test_context_strata_are_the_shared_serving_domains(self) -> None:
        self.assertEqual(DEVICE_STRATA, DEVICE_TYPES)
        self.assertEqual(TRAFFIC_STRATA, TRAFFIC_SOURCES)

    def test_archetype_seeds_are_distinct_and_policy_pairable(self) -> None:
        names = ["Explorer", "FastBuyer", "WindowShopper"]
        first = derive_archetype_seeds(names, 17)
        second = derive_archetype_seeds(names, 17)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), len(names))

    def test_context_seeds_are_independent_across_policy_cells(self) -> None:
        first = derive_context_seed(0, 0, 0, 0, 17)
        second = derive_context_seed(1, 0, 0, 0, 17)
        self.assertNotEqual(first, second)
        self.assertEqual(first, derive_context_seed(0, 0, 0, 0, 17))

    def test_serving_history_includes_primed_credit_when_dynamics_are_off(self) -> None:
        buckets = _production_history_buckets(["help_popup", "no-op"])
        self.assertEqual(buckets["interventions_shown_bucket"], 1)
        self.assertEqual(buckets["steps_since_widget_bucket"], 1)
        self.assertEqual(buckets["primed_credit_bucket"], 1)

    def test_cell_metrics_are_deterministic_for_a_fixed_seed(self) -> None:
        config = REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml"
        archetypes, _, full_config = load_archetypes(str(config))
        kwargs = {
            "policy_name": "no_op",
            "archetype": archetypes["Explorer"],
            "n_sessions": 30,
            "seed": 91,
            "t_max": 20,
            "dynamics": merge_dynamics(full_config.get("dynamics", {})),
        }
        first = evaluate_cell(policy=_AlwaysNoOpPolicy(), **kwargs)
        second = evaluate_cell(policy=_AlwaysNoOpPolicy(), **kwargs)
        self.assertEqual(first.summary, second.summary)
        self.assertIn("funnel_depth_mean", first.summary)

    def test_context_stratum_fixes_initial_device_and_traffic(self) -> None:
        config = REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml"
        archetypes, _, full_config = load_archetypes(str(config))
        policy = _AlwaysNoOpPolicy()
        result = evaluate_cell(
            policy_name="no_op",
            policy=policy,
            archetype=archetypes["Explorer"],
            n_sessions=5,
            seed=17,
            t_max=20,
            dynamics=merge_dynamics(full_config.get("dynamics", {})),
            device_type="unknown",
            traffic_source="referral",
        )
        self.assertEqual(result.summary["device_type"], "unknown")
        self.assertEqual(result.summary["traffic_source"], "referral")
        landing_states = [
            state for state in policy.states
            if state.get("decision_point") == "landing"
        ]
        self.assertTrue(landing_states)
        self.assertTrue(all(state["device_type"] == "unknown" for state in landing_states))
        self.assertTrue(all(state["traffic_source"] == "referral" for state in landing_states))

    def test_funnel_depth_uses_deepest_stage_and_purchase(self) -> None:
        transitions = [
            {"decision_point": "landing", "metadata": {"generated_events": []}},
            {"decision_point": "cart", "metadata": {"generated_events": []}},
        ]
        self.assertEqual(_session_funnel_depth(transitions), 3.0)
        transitions[-1]["metadata"]["generated_events"] = ["purchase"]
        self.assertEqual(_session_funnel_depth(transitions), 5.0)

    def test_funnel_progression_conditions_on_reaching_each_stage(self) -> None:
        depths = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0]
        self.assertEqual(
            _funnel_depth_histogram(depths),
            {"0": 1, "1": 2, "2": 3, "3": 1, "4": 1, "5": 2},
        )
        progression = _funnel_progression(depths)
        self.assertEqual(
            progression["landing_to_pdp"],
            {"reached": 9, "advanced": 7, "proportion": 7 / 9},
        )
        self.assertEqual(
            progression["checkout_to_purchase"],
            {"reached": 3, "advanced": 2, "proportion": 2 / 3},
        )
        # An empty conditioning set must report NaN, not a silent zero: a cell
        # in which nobody reached the cart says nothing about cart behaviour.
        import math

        no_carts = _funnel_progression([0.0, 1.0, 2.0])
        self.assertEqual(no_carts["cart_to_checkout"]["reached"], 0)
        self.assertTrue(math.isnan(no_carts["cart_to_checkout"]["proportion"]))

    def test_cell_summary_and_csv_carry_consistent_progression(self) -> None:
        config = REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml"
        archetypes, _, full_config = load_archetypes(str(config))
        result = evaluate_cell(
            policy_name="no_op",
            policy=_AlwaysNoOpPolicy(),
            archetype=archetypes["FastBuyer"],
            n_sessions=50,
            seed=23,
            t_max=20,
            dynamics=merge_dynamics(full_config.get("dynamics", {})),
        )
        summary = result.summary
        histogram = summary["funnel_depth_histogram"]
        self.assertEqual(sum(histogram.values()), summary["n_sessions"])
        progression = summary["funnel_progression"]
        # The progression must be exactly the histogram's tail ratios.
        reached_pdp = sum(histogram[str(d)] for d in range(2, 6))
        self.assertEqual(progression["landing_to_pdp"]["advanced"], reached_pdp)
        # Purchase (depth 5) must equal the conversion count: same observable.
        self.assertEqual(
            progression["checkout_to_purchase"]["advanced"],
            summary["conversion_count"],
        )
        row = _csv_rows("study", [summary])[0]
        self.assertEqual(row["funnel_depth_2_count"], histogram["2"])
        self.assertEqual(
            row["progression_pdp_to_cart_reached"],
            progression["pdp_to_cart"]["reached"],
        )

    def test_bandit_reports_partial_arm_coverage_and_serves_greedy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bandit.db"
            connection = sqlite3.connect(str(db_path))
            try:
                connection.execute(
                    "CREATE TABLE bandit_arm_stats ("
                    "decision_point TEXT, context_key TEXT, action TEXT, "
                    "impressions INTEGER, reward_sum REAL)"
                )
                connection.execute(
                    "INSERT INTO bandit_arm_stats VALUES (?, ?, ?, ?, ?)",
                    (
                        "landing",
                        "landing|mobile|direct|0",
                        "help_popup",
                        1,
                        10.0,
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            policy = FrozenBanditPolicy(db_path)
            policy.set_state(
                {
                    "decision_point": "landing",
                    "device_type": "mobile",
                    "traffic_source": "direct",
                    "page_depth_bucket": 0,
                }
            )
            action, propensity = policy.select_action(ALL_ACTIONS)
            self.assertEqual(action, "help_popup")
            self.assertEqual(propensity, 1.0)
            diagnostics = policy.diagnostics()
            self.assertEqual(diagnostics["partially_unseen_context_decisions"], 1)
            self.assertEqual(diagnostics["selected_unseen_arm_decisions"], 0)

    def test_writer_refuses_a_nonempty_study_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"
            output.mkdir()
            (output / "keep.txt").write_text("user artifact", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_outputs(output, {"study_id": "test"}, [])
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user artifact")


if __name__ == "__main__":
    unittest.main()
