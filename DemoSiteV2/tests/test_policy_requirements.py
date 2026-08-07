from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SharedSchema"))

from DemoSiteV2.app import config, main
from DemoSiteV2.app.routers.decision import DecisionRequest
from pydantic import ValidationError
from shared_schema.constants import CONTEXT_SCHEMA_VERSION
from shared_schema.policy_contract import (
    required_bandit_arm_keys,
    required_bandit_context_keys,
)


class BanditPolicyRequirementTests(unittest.TestCase):
    def test_decision_request_rejects_unknown_serving_point(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(session_id="session", decision_point="unknown_point")

    def setUp(self):
        self.original_required = config.REQUIRE_BANDIT_POLICY
        self.original_frozen = config.FREEZE_POLICY
        self.original_path = config.DATABASE_PATH
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        config.REQUIRE_BANDIT_POLICY = self.original_required
        config.FREEZE_POLICY = self.original_frozen
        config.DATABASE_PATH = self.original_path
        self.temp_dir.cleanup()

    def _path(self) -> Path:
        return Path(self.temp_dir.name) / "bandit.db"

    def _write_complete_policy(self) -> None:
        arm_keys = sorted(required_bandit_arm_keys())
        connection = sqlite3.connect(self._path())
        try:
            connection.executescript(
                """
                CREATE TABLE bandit_arm_stats (
                    id INTEGER PRIMARY KEY,
                    decision_point TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    impressions INTEGER NOT NULL,
                    reward_sum REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE policy_arm_support (
                    decision_point TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    support_kind TEXT NOT NULL,
                    raw_empirical_impressions INTEGER NOT NULL,
                    pooled_context_empirical_impressions INTEGER NOT NULL,
                    backoff_empirical_impressions INTEGER NOT NULL,
                    pseudocount_impressions INTEGER NOT NULL
                );
                CREATE TABLE policy_build_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO bandit_arm_stats("
                "decision_point, context_key, action, impressions, reward_sum, schema_version"
                ") VALUES (?, ?, ?, 1, 0.0, ?)",
                [(*key, CONTEXT_SCHEMA_VERSION) for key in arm_keys],
            )
            connection.executemany(
                "INSERT INTO policy_arm_support VALUES "
                "(?, ?, ?, 'decision_point_action_pseudocount', 0, 0, 1, 1)",
                arm_keys,
            )
            support_manifest = {
                "coverage_validation_passed": True,
                "required_serving_contexts": len(required_bandit_context_keys()),
                "required_serving_arm_rows": len(arm_keys),
                "materialized_serving_arm_rows": len(arm_keys),
                "per_arm_support_table": "policy_arm_support",
            }
            connection.executemany(
                "INSERT INTO policy_build_metadata(key, value) VALUES (?, ?)",
                (
                    (
                        "context_schema_version",
                        json.dumps(CONTEXT_SCHEMA_VERSION),
                    ),
                    ("arm_support", json.dumps(support_manifest)),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def test_requirement_disabled_preserves_local_development(self):
        config.REQUIRE_BANDIT_POLICY = False
        config.DATABASE_PATH = str(self._path())
        main.validate_required_bandit_policy()

    def test_required_policy_needs_frozen_mode(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = False
        config.DATABASE_PATH = str(self._path())
        with self.assertRaisesRegex(RuntimeError, "requires FREEZE_POLICY"):
            main.validate_required_bandit_policy()

    def test_required_missing_or_empty_policy_fails_closed(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = True
        config.DATABASE_PATH = str(self._path())
        with self.assertRaisesRegex(RuntimeError, "missing"):
            main.validate_required_bandit_policy()

        sqlite3.connect(config.DATABASE_PATH).close()
        with self.assertRaisesRegex(RuntimeError, "serving-domain contract"):
            main.validate_required_bandit_policy()

    def test_valid_required_policy_is_accepted(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = True
        config.DATABASE_PATH = str(self._path())
        self._write_complete_policy()
        main.validate_required_bandit_policy()

    def test_partial_policy_is_rejected(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = True
        config.DATABASE_PATH = str(self._path())
        self._write_complete_policy()
        connection = sqlite3.connect(config.DATABASE_PATH)
        try:
            connection.execute("DELETE FROM bandit_arm_stats WHERE id = 1")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RuntimeError, "missing_arms=1"):
            main.validate_required_bandit_policy()

    def test_arm_statistics_must_match_support_accounting(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = True
        config.DATABASE_PATH = str(self._path())
        self._write_complete_policy()
        connection = sqlite3.connect(config.DATABASE_PATH)
        try:
            connection.execute(
                "UPDATE bandit_arm_stats SET impressions = 2 WHERE id = 1"
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RuntimeError, "invalid support accounting"):
            main.validate_required_bandit_policy()

    def test_startup_validates_before_database_creation(self):
        config.REQUIRE_BANDIT_POLICY = True
        config.FREEZE_POLICY = True
        config.DATABASE_PATH = str(self._path())
        manager = main.lifespan(main.app)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            asyncio.run(manager.__aenter__())
        self.assertFalse(self._path().exists())


if __name__ == "__main__":
    unittest.main()
