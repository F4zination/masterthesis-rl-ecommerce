from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from Experiments import analyze_clickworker_study as analysis


def _shop_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY, session_id TEXT, event_type TEXT, page TEXT,
            timestamp TEXT, metadata_json TEXT, worker_id TEXT, persona TEXT
        );
        CREATE TABLE decision_logs (
            id INTEGER PRIMARY KEY, session_id TEXT, action TEXT,
            timestamp TEXT, context_json TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, session_id TEXT, total REAL,
            created_at TEXT, worker_id TEXT, persona TEXT
        );
        """
    )
    for row in rows:
        sid = row["session_id"]
        worker = row["worker_id"]
        persona = row["persona"]
        base = row.get("base", "2026-07-01 10:00:00")
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, "page_view", "/", base, "{}", worker, persona),
        )
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, "page_view", "/product/test", "2026-07-01 10:01:00", "{}", worker, persona),
        )
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, "page_view", "/cart", "2026-07-01 10:02:00", "{}", worker, persona),
        )
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, "dwell_time", "/product/test", "2026-07-01 10:01:30", json.dumps({"seconds": 30}), worker, persona),
        )
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, "attention_check", "/study/attention", "2026-07-01 10:02:10", json.dumps({"passed": row.get("passed", True)}), worker, persona),
        )
        conn.execute(
            "INSERT INTO decision_logs VALUES (NULL,?,?,?,?)",
            (sid, row.get("action", "trust_badge"), "2026-07-01 10:00:10", json.dumps({
                "worker_id": worker,
                "persona": persona,
                "policy_source": row.get("source", "ppo_policy"),
            })),
        )
        if row.get("purchase"):
            metadata = json.dumps({"order_total": 100.0})
            conn.execute(
                "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?)",
                (sid, "purchase", "/checkout", "2026-07-01 10:02:00", metadata, worker, persona),
            )
            conn.execute(
                "INSERT INTO orders VALUES (NULL,?,?,?,?,?)",
                (sid, 100.0, "2026-07-01 10:02:00", worker, persona),
            )
    conn.commit()
    conn.close()


def _dispatcher_db(path: Path, assignments: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE assignments (pid TEXT, wid TEXT, condition TEXT, persona TEXT, created_at REAL)"
    )
    for index, row in enumerate(assignments):
        wid, condition, persona = row[:3]
        created_at = float(row[3]) if len(row) > 3 else float(index)
        conn.execute(
            "INSERT INTO assignments VALUES (?,?,?,?,?)",
            (f"pid-{index}", wid, condition, persona, created_at),
        )
    conn.commit()
    conn.close()


class ClickworkerAnalysisTests(unittest.TestCase):
    def test_reward_quality_fallback_and_itt_no_show(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / "v2.db"
            v3 = root / "v3.db"
            dispatcher = root / "dispatcher.db"
            _shop_db(v2, [{
                "session_id": "s-v2", "worker_id": "w-v2", "persona": "explorer",
                "purchase": False, "source": "epsilon_greedy_v2_frozen_greedy",
            }])
            _shop_db(v3, [{
                "session_id": "s-v3", "worker_id": "w-v3", "persona": "explorer",
                "purchase": True, "source": "offline_policy",
            }])
            _dispatcher_db(dispatcher, [
                ("w-v2", "v2", "explorer"),
                ("w-v3", "v3", "explorer"),
                ("w-no-show", "v3", "fastbuyer"),
            ])

            sessions = analysis.load_shop_sessions("v2", v2) + analysis.load_shop_sessions("v3", v3)
            records, audit = analysis.build_participant_records(
                sessions, analysis.load_assignments(dispatcher)
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(audit, [])
            by_worker = {record.worker_id: record for record in records}
            self.assertTrue(by_worker["w-v2"].quality_ok)
            self.assertAlmostEqual(by_worker["w-v2"].session_reward, 0.2 - 0.03)
            self.assertEqual(by_worker["w-v3"].conversion, 1)
            self.assertAlmostEqual(by_worker["w-v3"].session_reward, 9.2 - 0.03)
            self.assertEqual(by_worker["w-v3"].fallback_decisions, 1)
            self.assertEqual(by_worker["w-v3"].fallback_rate, 1.0)
            self.assertFalse(by_worker["w-no-show"].observed_session)
            self.assertEqual(by_worker["w-no-show"].session_reward, 0.0)

    def test_first_session_is_prespecified_for_duplicate_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v2.db"
            _shop_db(db, [
                {"session_id": "first", "worker_id": "worker", "persona": "explorer", "base": "2026-07-01 09:00:00"},
                {"session_id": "second", "worker_id": "worker", "persona": "explorer", "base": "2026-07-01 11:00:00", "purchase": True},
            ])
            records, _ = analysis.build_participant_records(
                analysis.load_shop_sessions("v2", db),
                [analysis.Assignment("worker", "v2", "explorer")],
            )
            self.assertEqual(records[0].session_id, "first")
            self.assertEqual(records[0].duplicate_sessions, 1)
            self.assertEqual(records[0].conversion, 0)

    def test_adjusted_effect_uses_equal_persona_weights(self) -> None:
        records = []
        for persona, v2_value, v3_value in (
            ("explorer", 0.0, 2.0),
            ("fastbuyer", 10.0, 10.0),
        ):
            for condition, value in (("v2", v2_value), ("v3", v3_value)):
                records.append(
                    analysis.ParticipantRecord(
                        condition=condition, worker_id=f"{condition}-{persona}", persona=persona,
                        device_type="desktop", traffic_source="referral",
                        session_id="s", assigned=True, observed_session=True, recorded_landing=True,
                        duplicate_sessions=0,
                        completed_attention=True, attention_passed=True, quality_ok=True,
                        quality_reasons="", page_views=3, duration_s=100, event_count=1,
                        decision_count=1, session_length_steps=1, funnel_depth=1, conversion=0,
                        order_total=0, event_reward=value, action_cost=0, session_reward=value,
                        widget_clicks=0, widget_dismissals=0, fallback_decisions=0,
                        decision_errors=0, fallback_rate=0, attribution_valid=True,
                    )
                )
        self.assertEqual(analysis.adjusted_difference(records, "session_reward"), 1.0)

    def test_fidelity_requires_all_cells_and_tests_equivalence(self) -> None:
        records = []
        cells = {}
        for condition in analysis.CONDITIONS:
            for persona in analysis.PERSONAS:
                record = analysis.ParticipantRecord(
                    condition=condition, worker_id=f"{condition}-{persona}", persona=persona,
                    device_type="desktop", traffic_source="referral",
                    session_id="s", assigned=True, observed_session=True, recorded_landing=True,
                    duplicate_sessions=0, completed_attention=True, attention_passed=True,
                    quality_ok=True, quality_reasons="", page_views=3, duration_s=100,
                    event_count=1, decision_count=3, session_length_steps=3, funnel_depth=2,
                    conversion=0, order_total=0, event_reward=0, action_cost=0,
                    session_reward=0, widget_clicks=0, widget_dismissals=0,
                    decision_errors=0, fallback_decisions=0, fallback_rate=0,
                    attribution_valid=True,
                )
                records.append(record)
                cells[(condition, persona)] = {
                    "n_sessions": 2000,
                    "conversion_rate": 0.0,
                    "step_count_mean": 3.0,
                    "step_count_sd": 0.0,
                    "funnel_depth_mean": 2.0,
                    "funnel_depth_sd": 0.0,
                }
        result = analysis.fidelity_analysis(
            records, {"cells": cells}, bootstrap=50, seed=7
        )
        self.assertTrue(result["all_endpoints_equivalent"])
        self.assertEqual(result["endpoints"]["session_length_steps"]["estimate"], 0.0)
        self.assertTrue(
            analysis.math.isnan(
                result["endpoints"]["session_length_steps"]["descriptive_persona_spearman"]
            )
        )

        aggregate_cells = {
            key: {
                **cell,
                "conversion_rate": 0.5,
                "step_count_mean": 10.0,
                "funnel_depth_mean": 4.0,
            }
            for key, cell in cells.items()
        }
        context_cells = {
            (condition, persona, "desktop", "referral"): cell
            for (condition, persona), cell in cells.items()
        }
        standardized = analysis.fidelity_analysis(
            records,
            {"cells": aggregate_cells, "context_strata": context_cells},
            bootstrap=50,
            seed=8,
        )
        self.assertTrue(standardized["context_standardized"])
        self.assertEqual(
            standardized["endpoints"]["session_length_steps"]["estimate"],
            0.0,
        )

    @staticmethod
    def _ordering_record(condition: str, persona: str, index: int, depth: int):
        return analysis.ParticipantRecord(
            condition=condition, worker_id=f"{condition}-{persona}-{index}",
            persona=persona, device_type="desktop", traffic_source="referral",
            session_id=f"s-{condition}-{persona}-{index}", assigned=True,
            observed_session=True, recorded_landing=True, duplicate_sessions=0,
            completed_attention=True, attention_passed=True, quality_ok=True,
            quality_reasons="", page_views=3, duration_s=100, event_count=1,
            decision_count=1, session_length_steps=1, funnel_depth=depth,
            conversion=int(depth >= 5), order_total=0, event_reward=0,
            action_cost=0, session_reward=0, widget_clicks=0,
            widget_dismissals=0, decision_errors=0, fallback_decisions=0,
            fallback_rate=0, attribution_valid=True,
        )

    @classmethod
    def _ordering_records(cls, depth_counts: dict[str, dict[int, int]]):
        records = []
        for persona, counts in depth_counts.items():
            for condition in analysis.CONDITIONS:
                index = 0
                for depth, count in counts.items():
                    for _ in range(count):
                        records.append(
                            cls._ordering_record(condition, persona, index, depth)
                        )
                        index += 1
        return records

    @staticmethod
    def _ordering_baseline(histograms: dict[str, dict[int, int]]):
        cells = {}
        for persona, histogram in histograms.items():
            for condition in analysis.CONDITIONS:
                cells[(condition, persona)] = {
                    "n_sessions": sum(histogram.values()),
                    "funnel_depth_histogram": {
                        str(depth): histogram.get(depth, 0) for depth in range(6)
                    },
                }
        return {"cells": cells}

    def test_transition_ordering_confirms_predicted_direction(self) -> None:
        # Per persona and condition: engineered so the four confirmatory
        # contrasts hold decisively at n=40 per persona-condition (80 pooled).
        records = self._ordering_records({
            "fastbuyer": {1: 4, 2: 14, 3: 22},
            "detailedcomparator": {1: 10, 2: 24, 3: 6},
            "windowshopper": {1: 24, 2: 15, 3: 1},
        })
        baseline = self._ordering_baseline({
            "fastbuyer": {1: 200, 2: 800, 3: 1000},
            "detailedcomparator": {1: 500, 2: 1200, 3: 300},
            "windowshopper": {1: 1200, 2: 750, 3: 50},
        })
        result = analysis.transition_ordering_analysis(records, baseline)

        self.assertTrue(result["baseline_has_histograms"])
        self.assertTrue(result["analyzable"])
        self.assertTrue(result["all_confirmatory_hold"])
        self.assertEqual(
            result["personas_present"],
            ["fastbuyer", "detailedcomparator", "windowshopper"],
        )
        # Policies pool within a persona: 40 + 40 recorded landings.
        self.assertEqual(result["human"]["fastbuyer"]["n"], 80)
        confirmatory = [c for c in result["contrasts"] if c["confirmatory"]]
        self.assertEqual(
            len(confirmatory), len(analysis.CONFIRMATORY_ORDERING_CONTRASTS)
        )
        for contrast in confirmatory:
            self.assertTrue(contrast["estimable"])
            self.assertTrue(contrast["direction_matches"])
            self.assertLess(
                contrast["one_sided_p"], analysis.ORDERING_ALPHA_ONE_SIDED
            )
        # Orientation comes from the reference, not from insertion order.
        fb_ws = next(
            c for c in confirmatory
            if c["stage_pair"] == "landing_to_pdp"
            and {c["higher"], c["lower"]} == {"fastbuyer", "windowshopper"}
        )
        self.assertEqual(fb_ws["higher"], "fastbuyer")
        self.assertAlmostEqual(fb_ws["human_higher_proportion"], 72 / 80)
        # Descriptive pairs are present but never confirmatory.
        descriptive = [c for c in result["contrasts"] if not c["confirmatory"]]
        self.assertTrue(descriptive)

    def test_transition_ordering_skips_thin_cells_and_reversed_direction(self) -> None:
        # WindowShopper reaches the cart only twice per condition, so any
        # cart_to_checkout contrast against it must be shown untested; and
        # FastBuyer's human landing->pdp proportion is reversed below
        # DetailedComparator's, so that confirmatory contrast must not hold.
        records = self._ordering_records({
            "fastbuyer": {1: 20, 2: 6, 3: 4, 4: 10},
            "detailedcomparator": {1: 10, 2: 16, 3: 4, 4: 10},
            "windowshopper": {1: 24, 2: 14, 3: 1, 4: 1},
        })
        baseline = self._ordering_baseline({
            "fastbuyer": {1: 200, 2: 600, 3: 400, 4: 800},
            "detailedcomparator": {1: 500, 2: 900, 3: 400, 4: 200},
            "windowshopper": {1: 1200, 2: 700, 3: 60, 4: 40},
        })
        result = analysis.transition_ordering_analysis(records, baseline)

        self.assertTrue(result["analyzable"])
        self.assertFalse(result["all_confirmatory_hold"])
        fb_dc = next(
            c for c in result["contrasts"]
            if c["stage_pair"] == "landing_to_pdp"
            and {c["higher"], c["lower"]} == {"fastbuyer", "detailedcomparator"}
        )
        self.assertEqual(fb_dc["higher"], "fastbuyer")
        self.assertFalse(fb_dc["direction_matches"])
        self.assertFalse(fb_dc["holds"])
        thin = [
            c for c in result["contrasts"]
            if c["stage_pair"] == "cart_to_checkout"
            and "windowshopper" in (c["higher"], c["lower"])
        ]
        self.assertTrue(thin)
        for contrast in thin:
            self.assertFalse(contrast["estimable"])
            self.assertNotIn("holds", contrast)

    def test_transition_ordering_requires_histogram_bearing_baseline(self) -> None:
        records = self._ordering_records({
            "fastbuyer": {1: 4, 2: 14, 3: 22},
            "windowshopper": {1: 24, 2: 15, 3: 1},
        })
        legacy_cells = {
            (condition, persona): {"n_sessions": 2000, "funnel_depth_mean": 2.0}
            for condition in analysis.CONDITIONS
            for persona in ("fastbuyer", "windowshopper")
        }
        result = analysis.transition_ordering_analysis(
            records, {"cells": legacy_cells}
        )
        self.assertFalse(result["baseline_has_histograms"])
        self.assertFalse(result["analyzable"])
        self.assertFalse(result["all_confirmatory_hold"])
        self.assertEqual(result["contrasts"], [])

    def test_cutoff_date_keeps_the_named_day_whole(self) -> None:
        self.assertEqual(
            analysis.parse_cutoff("2026-08-24"), datetime(2026, 8, 25, 0, 0, 0)
        )
        self.assertEqual(
            analysis.parse_cutoff("2026-08-24T12:30:00"), datetime(2026, 8, 24, 12, 30, 0)
        )
        self.assertEqual(
            analysis.parse_cutoff("2026-08-24T14:30:00+02:00"),
            datetime(2026, 8, 24, 12, 30, 0),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            analysis.parse_cutoff("last Tuesday")

    def test_cutoff_drops_out_of_window_assignments_and_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = Path(tmp) / "dispatcher.db"
            in_window = datetime(2026, 8, 20, 9, 0, 0, tzinfo=timezone.utc)
            out_window = datetime(2026, 8, 26, 9, 0, 0, tzinfo=timezone.utc)
            _dispatcher_db(dispatcher, [
                ("w-in", "v2", "fastbuyer", in_window.timestamp()),
                ("w-out", "v2", "fastbuyer", out_window.timestamp()),
            ])
            sessions = [
                analysis.RawSession(
                    condition="v2", session_id="in", worker_id="w-in",
                    persona="fastbuyer",
                    timestamps=[in_window.replace(tzinfo=None)],
                ),
                analysis.RawSession(
                    condition="v2", session_id="out", worker_id="w-out",
                    persona="fastbuyer",
                    timestamps=[out_window.replace(tzinfo=None)],
                ),
            ]
            assignments = analysis.load_assignments(dispatcher)
            kept_sessions, kept_assignments, dropped_sessions, dropped_assignments = (
                analysis.apply_cutoff(sessions, assignments, analysis.parse_cutoff("2026-08-24"))
            )
            self.assertEqual((dropped_sessions, dropped_assignments), (1, 1))
            self.assertEqual([s.worker_id for s in kept_sessions], ["w-in"])
            self.assertEqual([a.worker_id for a in kept_assignments], ["w-in"])

            records, _ = analysis.build_participant_records(kept_sessions, kept_assignments)
            self.assertEqual([r.worker_id for r in records], ["w-in"])

    def test_cutoff_keeps_sessions_that_carry_no_timestamp(self) -> None:
        undated = analysis.RawSession(condition="v2", session_id="s", worker_id="w")
        kept, assignments, dropped_sessions, dropped_assignments = analysis.apply_cutoff(
            [undated], None, analysis.parse_cutoff("2026-08-24")
        )
        self.assertEqual(kept, [undated])
        self.assertIsNone(assignments)
        self.assertEqual((dropped_sessions, dropped_assignments), (0, 0))

    def test_wilson_interval_brackets_the_proportion(self) -> None:
        low, high = analysis._wilson_ci(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)
        self.assertGreater(low, 0.4)
        self.assertLess(high, 1.0)
        import math

        empty_low, empty_high = analysis._wilson_ci(0, 0)
        self.assertTrue(math.isnan(empty_low) and math.isnan(empty_high))

    def test_spearman_uses_average_tie_ranks(self) -> None:
        self.assertAlmostEqual(
            analysis._spearman([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0]),
            1.0,
        )

    def test_simulator_reference_uses_empirical_joint_context_weights(self) -> None:
        template = dict(
            condition="v2", worker_id="worker", persona="explorer",
            session_id="s", assigned=True, observed_session=True,
            recorded_landing=True, duplicate_sessions=0,
            completed_attention=True, attention_passed=True, quality_ok=True,
            quality_reasons="", page_views=1, duration_s=1, event_count=1,
            decision_count=1, session_length_steps=1, funnel_depth=1,
            conversion=0, order_total=0, event_reward=0, action_cost=0,
            session_reward=0, widget_clicks=0, widget_dismissals=0,
            decision_errors=0, fallback_decisions=0, fallback_rate=0,
            attribution_valid=True,
        )
        records = [
            analysis.ParticipantRecord(
                **template, device_type="mobile", traffic_source="referral"
            ),
            analysis.ParticipantRecord(
                **template, device_type="desktop", traffic_source="direct"
            ),
            analysis.ParticipantRecord(
                **template, device_type="desktop", traffic_source="direct"
            ),
        ]
        baseline = {
            "context_strata": {
                ("v2", "explorer", "mobile", "referral"): {
                    "n_sessions": 100,
                    "conversion_rate": 0.0,
                    "step_count_mean": 1.0,
                    "step_count_sd": 0.0,
                    "funnel_depth_mean": 1.0,
                    "funnel_depth_sd": 0.0,
                },
                ("v2", "explorer", "desktop", "direct"): {
                    "n_sessions": 100,
                    "conversion_rate": 0.0,
                    "step_count_mean": 4.0,
                    "step_count_sd": 0.0,
                    "funnel_depth_mean": 1.0,
                    "funnel_depth_sd": 0.0,
                },
            }
        }
        result = analysis._standardized_simulator_mean(
            baseline, "v2", "explorer", records, "session_length_steps"
        )
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(
            result["weights"],
            {"desktop|direct": 2 / 3, "mobile|referral": 1 / 3},
        )


if __name__ == "__main__":
    unittest.main()
