from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from OfflineTraining import build_bandit_policy


class CompleteBanditPolicyTests(unittest.TestCase):
    POINT_FEATURES = {
        "landing": ["device_type", "traffic_source"],
    }
    POINT_ACTIONS = {
        "landing": ["no-op", "help_popup"],
    }
    FEATURE_DOMAINS = {
        "device_type": ("unknown", "mobile", "tablet", "desktop"),
        "traffic_source": ("direct", "social"),
    }

    @classmethod
    def _small_policy(cls) -> build_bandit_policy.CompletePolicy:
        transitions = [
            {
                "decision_point": "landing",
                "state": {
                    "schema_version": 2,
                    "device_type": "mobile",
                    "traffic_source": "direct",
                },
                "action": "no-op",
                "reward": 2.0,
            },
            {
                "decision_point": "landing",
                "state": {
                    "schema_version": 2,
                    "device_type": "desktop",
                    "traffic_source": "direct",
                },
                "action": "no-op",
                "reward": 0.0,
            },
            {
                "decision_point": "landing",
                "state": {
                    "schema_version": 2,
                    "device_type": "tablet",
                    "traffic_source": "direct",
                },
                "action": "help_popup",
                "reward": 4.0,
            },
        ]
        return build_bandit_policy._build_complete_policy(
            transitions,
            point_features=cls.POINT_FEATURES,
            point_actions=cls.POINT_ACTIONS,
            expected_schema_version=2,
            feature_domains=cls.FEATURE_DOMAINS,
        )

    def test_pools_device_and_labels_structural_backoff(self) -> None:
        policy = self._small_policy()

        self.assertEqual(len(policy.required_context_keys), 8)
        self.assertEqual(len(policy.arms), 16)
        self.assertEqual(policy.raw_empirical_arm_rows, 3)
        self.assertEqual(policy.pooled_empirical_base_arm_rows, 2)

        unknown_direct = ("landing", "landing|unknown|direct", "no-op")
        mobile_direct = ("landing", "landing|mobile|direct", "no-op")
        unknown_social = ("landing", "landing|unknown|social", "no-op")

        # All devices receive the same estimate based on the two real
        # direct/no-op transitions.
        self.assertEqual(
            policy.arms[unknown_direct],
            build_bandit_policy.ArmStats(impressions=2, reward_sum=2.0),
        )
        self.assertEqual(policy.arms[mobile_direct], policy.arms[unknown_direct])
        self.assertEqual(
            policy.support[unknown_direct].support_kind,
            build_bandit_policy.DEVICE_POOLED_SUPPORT,
        )
        self.assertEqual(
            policy.support[unknown_direct].raw_empirical_impressions,
            0,
        )
        self.assertEqual(
            policy.support[mobile_direct].raw_empirical_impressions,
            1,
        )

        # The unseen social/no-op context gets one explicitly labeled
        # pseudocount at the landing/no-op empirical mean: (2 + 0) / 2.
        self.assertEqual(
            policy.arms[unknown_social],
            build_bandit_policy.ArmStats(impressions=1, reward_sum=1.0),
        )
        social_support = policy.support[unknown_social]
        self.assertEqual(
            social_support.support_kind,
            build_bandit_policy.BACKOFF_SUPPORT,
        )
        self.assertEqual(social_support.backoff_empirical_impressions, 2)
        self.assertEqual(social_support.pseudocount_impressions, 1)

        metadata = build_bandit_policy._arm_support_metadata(policy)
        self.assertTrue(metadata["coverage_validation_passed"])
        self.assertEqual(metadata["required_serving_contexts"], 8)
        self.assertEqual(metadata["required_serving_arm_rows"], 16)
        self.assertEqual(metadata["device_pooled_empirical_arm_rows"], 8)
        self.assertEqual(
            metadata["decision_point_action_pseudocount_arm_rows"],
            8,
        )
        self.assertEqual(metadata["raw_empirical_transition_impressions"], 3)

    def test_default_schema_materializes_848_contexts_and_all_actions(self) -> None:
        shared_schema_path = str(
            build_bandit_policy.REPO_ROOT / "SharedSchema"
        )
        if shared_schema_path not in sys.path:
            sys.path.insert(0, shared_schema_path)

        from shared_schema.constants import CONTEXT_SCHEMA_VERSION
        from shared_schema.constants import FEATURE_VALUE_DOMAINS
        from shared_schema.features import POINT_ACTIONS, POINT_FEATURES

        transitions = []
        for decision_point, actions in POINT_ACTIONS.items():
            state = {
                "schema_version": CONTEXT_SCHEMA_VERSION,
                **{
                    feature: FEATURE_VALUE_DOMAINS[feature][0]
                    for feature in POINT_FEATURES[decision_point]
                },
            }
            for action_index, action in enumerate(actions):
                transitions.append(
                    {
                        "decision_point": decision_point,
                        "state": dict(state),
                        "action": action,
                        "reward": float(action_index),
                    }
                )

        policy = build_bandit_policy._build_complete_policy(
            transitions,
            point_features=POINT_FEATURES,
            point_actions=POINT_ACTIONS,
            expected_schema_version=CONTEXT_SCHEMA_VERSION,
        )

        self.assertEqual(len(policy.required_context_keys), 848)
        self.assertEqual(len(policy.required_arm_keys), 5_088)
        self.assertEqual(set(policy.arms), policy.required_arm_keys)
        self.assertEqual(set(policy.support), policy.required_arm_keys)

    def test_fails_when_a_point_action_has_no_empirical_backoff(self) -> None:
        transition = {
            "decision_point": "landing",
            "state": {
                "schema_version": 2,
                "device_type": "mobile",
                "traffic_source": "direct",
            },
            "action": "no-op",
            "reward": 1.0,
        }
        with self.assertRaisesRegex(
            ValueError,
            "no empirical backoff source.*landing/help_popup",
        ):
            build_bandit_policy._build_complete_policy(
                [transition],
                point_features=self.POINT_FEATURES,
                point_actions=self.POINT_ACTIONS,
                expected_schema_version=2,
                feature_domains=self.FEATURE_DOMAINS,
            )

    def test_coverage_validator_rejects_a_missing_arm(self) -> None:
        policy = self._small_policy()
        policy.arms.pop(next(iter(policy.required_arm_keys)))

        with self.assertRaisesRegex(ValueError, "missing_arms=1"):
            build_bandit_policy._validate_complete_coverage(policy)

    def test_writes_per_arm_support_manifest(self) -> None:
        policy = self._small_policy()
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "policy.db"
            sqlite3.connect(database).close()
            build_bandit_policy._write_arm_support(database, policy.support)

            with closing(sqlite3.connect(database)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM policy_arm_support"
                ).fetchone()[0]
                row = connection.execute(
                    """
                    SELECT support_kind, raw_empirical_impressions,
                           pooled_context_empirical_impressions,
                           backoff_empirical_impressions,
                           pseudocount_impressions
                    FROM policy_arm_support
                    WHERE decision_point = 'landing'
                      AND context_key = 'landing|unknown|social'
                      AND action = 'no-op'
                    """
                ).fetchone()

            self.assertEqual(count, 16)
            self.assertEqual(
                row,
                (
                    build_bandit_policy.BACKOFF_SUPPORT,
                    0,
                    0,
                    2,
                    1,
                ),
            )

    def test_persisted_validator_reuses_complete_serving_contract(self) -> None:
        policy = self._small_policy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "policy.db"
            dataset = root / "transitions.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")

            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE bandit_arm_stats (
                        decision_point TEXT,
                        context_key TEXT,
                        action TEXT,
                        impressions INTEGER,
                        reward_sum REAL,
                        schema_version INTEGER
                    )
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO bandit_arm_stats (
                        decision_point, context_key, action, impressions,
                        reward_sum, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (*key, stats.impressions, stats.reward_sum, 2)
                        for key, stats in policy.arms.items()
                    ],
                )
                connection.commit()

            build_bandit_policy._write_arm_support(database, policy.support)
            build_bandit_policy._write_policy_metadata(
                database,
                dataset_path=dataset,
                n_rows=policy.n_transitions_used,
                n_skipped=policy.n_transitions_skipped,
                schema_version=2,
                seed_products=False,
                arm_support=build_bandit_policy._arm_support_metadata(policy),
                git_state=("a" * 40, False),
            )

            validation = build_bandit_policy.validate_persisted_policy(
                database,
                point_features=self.POINT_FEATURES,
                point_actions=self.POINT_ACTIONS,
                expected_schema_version=2,
                feature_domains=self.FEATURE_DOMAINS,
            )
            self.assertTrue(validation["coverage_validation_passed"])
            self.assertEqual(validation["serving_contexts"], 8)
            self.assertEqual(validation["serving_arm_rows"], 16)

            missing_key = next(iter(policy.required_arm_keys))
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    DELETE FROM bandit_arm_stats
                    WHERE decision_point = ? AND context_key = ? AND action = ?
                    """,
                    missing_key,
                )
                connection.commit()
            with self.assertRaisesRegex(ValueError, "missing_arms=1"):
                build_bandit_policy.validate_persisted_policy(
                    database,
                    point_features=self.POINT_FEATURES,
                    point_actions=self.POINT_ACTIONS,
                    expected_schema_version=2,
                    feature_domains=self.FEATURE_DOMAINS,
                )

    def test_atomic_publish_preserves_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "policy.db"
            staged = root / ".policy.db.building"
            output.write_bytes(b"old-policy")
            staged.write_bytes(b"new-policy")

            with self.assertRaises(FileExistsError):
                build_bandit_policy._publish_staged_database(
                    staged,
                    output,
                    force=False,
                )
            self.assertEqual(output.read_bytes(), b"old-policy")
            self.assertEqual(staged.read_bytes(), b"new-policy")

            build_bandit_policy._publish_staged_database(
                staged,
                output,
                force=True,
            )
            self.assertEqual(output.read_bytes(), b"new-policy")
            self.assertFalse(staged.exists())


if __name__ == "__main__":
    unittest.main()
