from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from OfflineTraining import train_ppo_policy as ppo


def _sparse_rows() -> list[dict]:
    return [
        {
            "trajectory_id": "session-1",
            "t": 0,
            "decision_point": "landing",
            "state": {
                "decision_point": "landing",
                "schema_version": 2,
                "device_type": "mobile",
                "traffic_source": "direct",
                "page_depth_bucket": 1,
                "interventions_shown_bucket": 0,
                "steps_since_widget_bucket": 0,
                "session_step_bucket": 0,
                "primed_credit_bucket": 0,
            },
            "action": "no-op",
            "eligible_actions": [
                "no-op",
                "trending_carousel",
                "discount_banner",
                "frequently_bought_together",
                "trust_badge",
                "help_popup",
            ],
            "reward": 0.0,
            "done": True,
        }
    ]


class PpoProductionDomainTests(unittest.TestCase):
    def test_sparse_data_still_builds_complete_deterministic_domain(self) -> None:
        rows = _sparse_rows()
        vocab = ppo.build_vocab(rows)
        reversed_vocab = ppo.build_vocab(list(reversed(rows)))
        required = set(ppo.required_production_state_tokens())

        self.assertEqual(vocab, reversed_vocab)
        self.assertEqual(len(required), 46)
        self.assertTrue(required.issubset(vocab))
        self.assertIn("device_type=unknown", vocab)
        self.assertIn("item_count_bucket=3", vocab)
        self.assertIn("primed_credit_bucket=3", vocab)

        coverage = ppo.validate_vocabulary_coverage(
            vocab,
            ppo.build_action_vocab(rows),
            observed_tokens=ppo.observed_state_tokens(rows),
        )
        self.assertEqual(coverage["missing_state_tokens"], [])
        self.assertEqual(coverage["missing_actions"], [])
        self.assertIn(
            "device_type=unknown",
            coverage["unobserved_required_state_tokens"],
        )

        serving_state = {
            "decision_point": "checkout",
            "schema_version": 2,
            "device_type": "unknown",
            "traffic_source": "referral",
            "cart_total_bucket": 3,
            "item_count_bucket": 3,
            "interventions_shown_bucket": 3,
            "steps_since_widget_bucket": 3,
            "session_step_bucket": 3,
            "primed_credit_bucket": 3,
        }
        encoded = ppo.encode_state(serving_state, "checkout", vocab)
        for token in ppo.state_tokens(serving_state):
            self.assertEqual(float(encoded[vocab[token]].item()), 1.0, token)

    def test_train_initializes_unobserved_input_columns_neutrally(self) -> None:
        rows = _sparse_rows()
        model, vocab, _, summary = ppo.train_ppo(
            rows,
            mode="online",
            hidden_sizes=(8,),
            gamma=0.95,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            lr=3e-4,
            iterations=0,
            epochs=1,
            minibatch_size=8,
            max_grad_norm=0.5,
            rollout_sessions=1,
            t_max=1,
            dynamics={},
            sequential=True,
            archetypes={},
            mixture_prior={},
            seed=7,
        )

        first_linear = model.backbone[0]
        self.assertIsInstance(first_linear, torch.nn.Linear)
        unknown_column = first_linear.weight[:, vocab["device_type=unknown"]]
        observed_column = first_linear.weight[:, vocab["device_type=mobile"]]
        self.assertEqual(int(torch.count_nonzero(unknown_column).item()), 0)
        self.assertGreater(int(torch.count_nonzero(observed_column).item()), 0)
        self.assertEqual(
            summary["vocabulary_coverage"]["unobserved_token_initialization"],
            "zero_input_columns",
        )

    def test_validation_rejects_incomplete_or_non_dense_vocabulary(self) -> None:
        rows = _sparse_rows()
        vocab = ppo.build_vocab(rows)
        action_vocab = ppo.build_action_vocab(rows)

        del vocab["device_type=unknown"]
        with self.assertRaisesRegex(ValueError, "missing state tokens"):
            ppo.validate_vocabulary_coverage(vocab, action_vocab)


class PpoArtifactPairTests(unittest.TestCase):
    def test_pair_is_atomic_auditable_and_from_one_model(self) -> None:
        rows = _sparse_rows()
        vocab = ppo.build_vocab(rows)
        action_vocab = ppo.build_action_vocab(rows)
        torch.manual_seed(11)
        model = ppo.ActorCritic(len(vocab), len(action_vocab), (8,))
        ppo.initialize_unobserved_token_columns(
            model,
            vocab,
            ppo.observed_state_tokens(rows),
        )
        policy, defaults = ppo.greedy_policy_from_model(
            model,
            rows,
            vocab,
            action_vocab,
            sequential=True,
        )

        provenance = {
            "dataset_sha256": "dataset-sha",
            "training_script_sha256": "script-sha",
            "seed": 11,
            "mode": "online",
            "sequential": True,
            "timing_mode": "opportunity",
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
            "model_state_dict": model.state_dict(),
            "state_vocab": vocab,
            "action_vocab": action_vocab,
            "hidden_sizes": (8,),
            "provenance": provenance,
            "summary": {},
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

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint_path = output / "ppo_policy.pt"
            policy_path = output / "trained_policy.json"
            integrity = ppo.write_policy_artifact_pair(
                checkpoint_path,
                policy_path,
                checkpoint_payload,
                policy_payload,
            )

            checkpoint = ppo.load_checkpoint_for_validation(checkpoint_path)
            exported = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["export_id"], exported["export_id"])
            self.assertEqual(
                exported["checkpoint_sha256"],
                ppo._sha256(checkpoint_path),
            )
            self.assertEqual(integrity["export_id"], identity["export_id"])
            self.assertEqual(
                list(output.glob(".ppo_policy.pt.*.tmp"))
                + list(output.glob(".trained_policy.json.*.tmp")),
                [],
            )

            loaded_model = ppo.ActorCritic(len(vocab), len(action_vocab), (8,))
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            expected_policy, expected_defaults = ppo.greedy_policy_from_model(
                loaded_model,
                rows,
                vocab,
                action_vocab,
                sequential=True,
            )
            self.assertEqual(exported["policy"], expected_policy)
            self.assertEqual(
                exported["default_action_by_decision_point"],
                expected_defaults,
            )
            self.assertEqual(
                set(expected_defaults),
                set(ppo.PRODUCTION_DECISION_POINTS),
            )

            # A stale/mixed JSON is detected even though each file is valid on
            # its own.
            exported["export_id"] = "stale-export"
            policy_path.write_text(json.dumps(exported), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Inconsistent PPO artifact pair"):
                ppo.validate_saved_artifact_pair(checkpoint_path, policy_path)

    def test_main_exports_a_valid_pair_with_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "transitions.jsonl"
            dataset.write_text(
                json.dumps(_sparse_rows()[0]) + "\n",
                encoding="utf-8",
            )
            output = root / "outputs"
            archetypes = (
                Path(ppo.__file__).resolve().parent.parent
                / "CustomerSimulation"
                / "config"
                / "archetypes.yaml"
            )
            argv = [
                "train_ppo_policy.py",
                "--dataset",
                str(dataset),
                "--archetypes",
                str(archetypes),
                "--mode",
                "online",
                "--iterations",
                "0",
                "--hidden-sizes",
                "8",
                "--sequential",
                "--timing-mode",
                "opportunity",
                "--out-dir",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(ppo, "_git_state", return_value=("a" * 40, False)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                ppo.main()

            checkpoint_path = output / "ppo_policy.pt"
            policy_path = output / "trained_policy.json"
            report = ppo.validate_saved_artifact_pair(
                checkpoint_path,
                policy_path,
            )
            checkpoint = ppo.load_checkpoint_for_validation(checkpoint_path)
            exported = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertEqual(report["state_vocab_size"], 46)
            self.assertEqual(
                checkpoint["vocabulary_coverage"]["missing_state_tokens"],
                [],
            )
            self.assertEqual(
                exported["vocabulary_coverage"]["missing_state_tokens"],
                [],
            )
            self.assertEqual(
                checkpoint["provenance"],
                exported["provenance"],
            )


if __name__ == "__main__":
    unittest.main()
