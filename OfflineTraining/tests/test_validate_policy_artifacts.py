from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from OfflineTraining import train_ppo_policy as ppo
from OfflineTraining import validate_policy_artifacts as validator
from shared_schema.constants import (
    ALL_ACTIONS,
    CONTEXT_SCHEMA_VERSION,
    HISTORY_PRIMED_CREDIT_DECAY,
    HISTORY_PRIMING_ACTIONS,
    HISTORY_PRIMING_AMOUNT,
)
from shared_schema.policy_contract import (
    PolicyContractError,
    required_bandit_arm_keys,
    required_bandit_context_keys,
    required_ppo_state_tokens,
)

DATASET_SHA256 = "a" * 64
GIT_COMMIT = "c" * 40


def _write_valid_ppo_pair(
    root: Path,
    *,
    dataset_sha256: str = DATASET_SHA256,
) -> tuple[Path, Path]:
    vocab = {
        token: index
        for index, token in enumerate(required_ppo_state_tokens())
    }
    action_vocab = {
        action: index
        for index, action in enumerate(ALL_ACTIONS)
    }
    torch.manual_seed(19)
    model = ppo.ActorCritic(len(vocab), len(action_vocab), (8,))
    policy, defaults = ppo.greedy_policy_from_model(
        model,
        [],
        vocab,
        action_vocab,
        sequential=True,
    )
    provenance = {
        "dataset_sha256": dataset_sha256,
        "training_script_sha256": "script-sha",
        "seed": 19,
        "mode": "online",
        "sequential": True,
        "timing_mode": "opportunity",
        "iterations": 1,
        "rollout_sessions": 1,
        "state_vocab_domain_version": 1,
        "unobserved_token_initialization": "zero_input_columns",
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "dynamics": {
            "delayed_reward_decay": HISTORY_PRIMED_CREDIT_DECAY,
            "priming_actions": list(HISTORY_PRIMING_ACTIONS),
            "priming_amount": HISTORY_PRIMING_AMOUNT,
        },
    }
    identity = ppo.build_export_identity(
        model,
        vocab,
        action_vocab,
        provenance,
    )
    provenance.update(identity)
    checkpoint_payload = {
        "algorithm": "ppo_clip_actor_critic",
        **identity,
        "mode": "online",
        "sequential": True,
        "timing_mode": "opportunity",
        "model_state_dict": model.state_dict(),
        "state_vocab": vocab,
        "action_vocab": action_vocab,
        "hidden_sizes": (8,),
        "provenance": provenance,
        "summary": {
            "mode": "online",
            "sequential": True,
            "iterations": 1,
            "rollout_sessions": 1,
            "n_rows": 1,
        },
        "vocabulary_coverage": ppo.vocabulary_coverage(
            vocab,
            action_vocab,
            observed_tokens=set(vocab),
        ),
    }
    policy_payload = {
        "algorithm": "ppo_clip_actor_critic",
        **identity,
        "provenance": provenance,
        "policy_export": {
            "source": "greedy_argmax_from_checkpoint_model",
            "source_model_state_sha256": identity["model_state_sha256"],
            "policy_sha256": ppo._canonical_json_sha256(policy),
            "defaults_sha256": ppo._canonical_json_sha256(defaults),
        },
        "policy": policy,
        "default_action_by_decision_point": defaults,
    }
    checkpoint = root / "ppo_policy.pt"
    json_policy = root / "trained_policy.json"
    ppo.write_policy_artifact_pair(
        checkpoint,
        json_policy,
        checkpoint_payload,
        policy_payload,
    )
    return checkpoint, json_policy


def _write_valid_bandit_database(
    path: Path,
    *,
    dataset_sha256: str = DATASET_SHA256,
) -> None:
    arm_keys = sorted(required_bandit_arm_keys())
    context_count = len(required_bandit_context_keys())
    with contextlib.closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE bandit_arm_stats (
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
            """
            INSERT INTO bandit_arm_stats (
                decision_point, context_key, action, impressions,
                reward_sum, schema_version
            ) VALUES (?, ?, ?, 1, 0.0, ?)
            """,
            (
                (*key, CONTEXT_SCHEMA_VERSION)
                for key in arm_keys
            ),
        )
        connection.executemany(
            """
            INSERT INTO policy_arm_support (
                decision_point, context_key, action, support_kind,
                raw_empirical_impressions,
                pooled_context_empirical_impressions,
                backoff_empirical_impressions,
                pseudocount_impressions
            ) VALUES (?, ?, ?, 'decision_point_action_pseudocount', 0, 0, 1, 1)
            """,
            arm_keys,
        )
        arm_support = {
            "coverage_validation_passed": True,
            "required_serving_contexts": context_count,
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
                ("arm_support", json.dumps(arm_support, sort_keys=True)),
                ("dataset_sha256", json.dumps(dataset_sha256)),
                ("build_git_commit", json.dumps(GIT_COMMIT)),
                ("build_git_dirty", json.dumps(False)),
            ),
        )
        connection.commit()


class PolicyArtifactValidatorTests(unittest.TestCase):
    def test_validates_real_ppo_pair_and_complete_bandit_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, json_policy = _write_valid_ppo_pair(root)
            bandit_db = root / "bandit.db"
            _write_valid_bandit_database(bandit_db)

            report = validator.validate_policy_artifacts(
                only="all",
                ppo_checkpoint=checkpoint,
                ppo_policy=json_policy,
                bandit_db=bandit_db,
            )

            self.assertTrue(report["valid"])
            self.assertEqual(
                report["checks"]["ppo"]["vocabulary"][
                    "required_state_token_count"
                ],
                46,
            )
            self.assertEqual(
                report["checks"]["bandit"]["contract"]["serving_arm_count"],
                5_088,
            )

    def test_rejects_stale_ppo_json_and_missing_bandit_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, json_policy = _write_valid_ppo_pair(root)
            payload = json.loads(json_policy.read_text(encoding="utf-8"))
            payload["export_id"] = "stale-export"
            json_policy.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Inconsistent PPO artifact pair"):
                validator.validate_ppo_artifacts(checkpoint, json_policy)

            bandit_db = root / "bandit.db"
            _write_valid_bandit_database(bandit_db)
            with contextlib.closing(sqlite3.connect(bandit_db)) as connection:
                connection.execute(
                    "DELETE FROM bandit_arm_stats WHERE rowid = "
                    "(SELECT MIN(rowid) FROM bandit_arm_stats)"
                )
                connection.commit()
            with self.assertRaisesRegex(PolicyContractError, "missing_arms=1"):
                validator.validate_bandit_artifact(bandit_db)

    def test_rejects_wrong_ppo_treatment_and_mixed_training_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, json_policy = _write_valid_ppo_pair(root)
            checkpoint_payload = ppo.load_checkpoint_for_validation(checkpoint)
            checkpoint_payload["mode"] = "offline"
            torch.save(checkpoint_payload, checkpoint)
            with (
                patch.object(
                    validator,
                    "validate_saved_artifact_pair",
                    return_value={"export_id": "paired"},
                ),
                self.assertRaisesRegex(
                    PolicyContractError,
                    "mode must be 'online'",
                ),
            ):
                validator.validate_ppo_artifacts(checkpoint, json_policy)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, json_policy = _write_valid_ppo_pair(root)
            checkpoint_payload = ppo.load_checkpoint_for_validation(checkpoint)
            checkpoint_payload["provenance"]["dynamics"][
                "delayed_reward_decay"
            ] = 0.5
            torch.save(checkpoint_payload, checkpoint)
            with (
                patch.object(
                    validator,
                    "validate_saved_artifact_pair",
                    return_value={"export_id": "paired"},
                ),
                self.assertRaisesRegex(
                    PolicyContractError,
                    "history recurrence delayed_reward_decay",
                ),
            ):
                validator.validate_ppo_artifacts(checkpoint, json_policy)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, json_policy = _write_valid_ppo_pair(root)
            bandit_db = root / "bandit.db"
            _write_valid_bandit_database(
                bandit_db,
                dataset_sha256="b" * 64,
            )
            with self.assertRaisesRegex(
                validator.ArtifactValidationError,
                "different datasets",
            ):
                validator.validate_policy_artifacts(
                    only="all",
                    ppo_checkpoint=checkpoint,
                    ppo_policy=json_policy,
                    bandit_db=bandit_db,
                )

    def test_preflight_rejects_malformed_architecture_and_tensor_shapes(
        self,
    ) -> None:
        cases = (
            (
                "hidden-sizes",
                lambda payload: payload.update(
                    {"hidden_sizes": ("8",)}
                ),
                "hidden_sizes must be",
            ),
            (
                "tensor-shape",
                lambda payload: payload.update(
                    {
                        "model_state_dict": {
                            **payload["model_state_dict"],
                            "backbone.0.weight": payload[
                                "model_state_dict"
                            ]["backbone.0.weight"][:, :-1],
                        }
                    }
                ),
                "architecture/state_dict cannot be loaded",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, json_policy = _write_valid_ppo_pair(root)
                checkpoint_payload = ppo.load_checkpoint_for_validation(
                    checkpoint
                )
                mutate(checkpoint_payload)
                torch.save(checkpoint_payload, checkpoint)

                # Isolate the model-load gate: a cryptographically relinked
                # malformed pair must still be rejected by preflight.
                with (
                    patch.object(
                        validator,
                        "validate_saved_artifact_pair",
                        return_value={"export_id": "paired"},
                    ),
                    self.assertRaisesRegex(
                        PolicyContractError,
                        expected_error,
                    ),
                ):
                    validator.validate_ppo_artifacts(
                        checkpoint,
                        json_policy,
                    )

    def test_shared_vocabulary_gate_is_independent_of_pair_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ppo_policy.pt"
            json_policy = root / "trained_policy.json"
            checkpoint.write_bytes(b"checkpoint")
            json_policy.write_text("{}", encoding="utf-8")
            incomplete_vocab = {
                token: index
                for index, token in enumerate(required_ppo_state_tokens())
                if token != "device_type=unknown"
            }
            action_vocab = {
                action: index
                for index, action in enumerate(ALL_ACTIONS)
            }

            with (
                patch.object(
                    validator,
                    "validate_saved_artifact_pair",
                    return_value={"export_id": "paired"},
                ),
                patch.object(
                    validator,
                    "load_checkpoint_for_validation",
                    return_value={
                        "state_vocab": incomplete_vocab,
                        "action_vocab": action_vocab,
                    },
                ),
                self.assertRaisesRegex(
                    PolicyContractError,
                    "device_type=unknown",
                ),
            ):
                validator.validate_ppo_artifacts(checkpoint, json_policy)

    def test_cli_returns_nonzero_for_a_missing_artifact(self) -> None:
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stderr(stderr),
        ):
            missing = Path(tmp) / "missing.db"
            result = validator.main(
                ["--only", "bandit", "--bandit-db", str(missing)]
            )

        self.assertEqual(result, 1)
        self.assertIn("validation failed", stderr.getvalue())
        self.assertIn("is missing", stderr.getvalue())


class DeploymentPreparationGateTests(unittest.TestCase):
    def test_script_validates_before_retaining_and_after_building(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "deploy"
            / "prepare-study-artifacts.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("OfflineTraining/validate_training_dataset.py", script)
        self.assertIn("--expected-sessions", script)
        self.assertIn("--require-sequential", script)
        self.assertLess(
            script.index("Validating the existing shared training dataset"),
            script.index("Keeping the validated existing shared training dataset"),
        )
        self.assertIn("OfflineTraining/validate_policy_artifacts.py", script)
        self.assertGreaterEqual(script.count("--only ppo"), 2)
        self.assertGreaterEqual(script.count("--only bandit"), 2)
        self.assertIn("--only all", script)
        self.assertIn("--training-dataset", script)
        self.assertIn("--require-clean-provenance", script)
        self.assertLess(
            script.index("Validating the existing V3 artifact pair"),
            script.index("Keeping the validated existing V3 artifacts"),
        )
        self.assertLess(
            script.index("Validating the existing V2 policy database"),
            script.index("Keeping the validated existing V2 database"),
        )


if __name__ == "__main__":
    unittest.main()
