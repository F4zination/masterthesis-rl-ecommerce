from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SharedSchema"))

from DemoSiteV3.app import main  # noqa: E402
from DemoSiteV3.app.routers.decision import DecisionRequest  # noqa: E402
from DemoSiteV3.app.services import decision  # noqa: E402
from DemoSiteV3.app.services import ppo  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from shared_schema.constants import ALL_ACTIONS  # noqa: E402
from shared_schema.policy_contract import required_ppo_state_tokens  # noqa: E402


class PolicyRequirementTests(unittest.TestCase):
    def test_decision_request_rejects_unknown_serving_point(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionRequest(session_id="session", decision_point="unknown_point")

    def test_requirement_disabled_preserves_development_fallback(self) -> None:
        with patch.object(main, "REQUIRE_PPO_CHECKPOINT", False):
            main._enforce_policy_requirements({"ppo_checkpoint_valid": False})

    def test_required_invalid_checkpoint_fails_closed(self) -> None:
        health = {"ppo_checkpoint_valid": False, "ppo_checkpoint_path": "missing.pt"}
        with (
            patch.object(main, "REQUIRE_PPO_CHECKPOINT", True),
            patch.object(main, "POLICY_MODE", "ppo_only"),
            self.assertRaisesRegex(RuntimeError, "missing or invalid"),
        ):
            main._enforce_policy_requirements(health)

    def test_required_checkpoint_needs_ppo_mode(self) -> None:
        with (
            patch.object(main, "REQUIRE_PPO_CHECKPOINT", True),
            patch.object(main, "POLICY_MODE", "offline_only"),
            self.assertRaisesRegex(RuntimeError, "requires POLICY_MODE"),
        ):
            main._enforce_policy_requirements({"ppo_checkpoint_valid": True})

    def test_valid_required_checkpoint_is_accepted(self) -> None:
        with (
            patch.object(main, "REQUIRE_PPO_CHECKPOINT", True),
            patch.object(main, "POLICY_MODE", "ppo_only"),
        ):
            main._enforce_policy_requirements(
                {
                    "ppo_checkpoint_valid": True,
                    "ppo_checkpoint_study_contract_valid": True,
                }
            )

    def test_required_checkpoint_rejects_wrong_study_treatment(self) -> None:
        with (
            patch.object(main, "REQUIRE_PPO_CHECKPOINT", True),
            patch.object(main, "POLICY_MODE", "ppo_only"),
            self.assertRaisesRegex(RuntimeError, "study contract"),
        ):
            main._enforce_policy_requirements(
                {
                    "ppo_checkpoint_valid": True,
                    "ppo_checkpoint_study_contract_valid": False,
                    "ppo_checkpoint_study_contract_error": "mode is offline",
                }
            )

    def test_ppo_only_request_fails_instead_of_serving_fallback(self) -> None:
        with (
            patch.object(decision, "POLICY_MODE", "ppo_only"),
            patch.object(decision, "run_ppo_policy", return_value=None),
            patch.object(decision, "_timing_gate", return_value=(True, 0, "eligible")),
            patch.object(decision, "_session_history", return_value={}),
            patch.object(decision, "_normalize_context", return_value=("context", {})),
            self.assertRaisesRegex(RuntimeError, "could not produce"),
        ):
            decision.get_decision(
                object(),
                session_id="session",
                decision_point="landing",
            )

    def test_startup_validates_checkpoint_before_database_writes(self) -> None:
        async def enter_lifespan() -> None:
            async with main.lifespan(object()):
                pass

        with (
            patch.object(main, "REQUIRE_PPO_CHECKPOINT", True),
            patch.object(main, "POLICY_MODE", "ppo_only"),
            patch.object(
                main,
                "get_policy_health_status",
                return_value={"ppo_checkpoint_valid": False, "ppo_checkpoint_path": "missing.pt"},
            ),
            patch.object(main, "init_db") as init_db,
            self.assertRaisesRegex(RuntimeError, "missing or invalid"),
        ):
            asyncio.run(enter_lifespan())
        init_db.assert_not_called()

    @unittest.skipIf(ppo.torch is None, "PyTorch is not installed")
    def test_checkpoint_requires_complete_serving_vocabulary(self) -> None:
        state_vocab = {
            token: index
            for index, token in enumerate(required_ppo_state_tokens())
        }
        action_vocab = {
            action: index for index, action in enumerate(ALL_ACTIONS)
        }
        model = ppo.ActorCritic(len(state_vocab), len(action_vocab), (8,))

        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.pt"
            invalid_path = Path(temp_dir) / "invalid.pt"
            malformed_index_path = Path(temp_dir) / "malformed-index.pt"
            payload = {
                "state_vocab": state_vocab,
                "action_vocab": action_vocab,
                "hidden_sizes": (8,),
                "model_state_dict": model.state_dict(),
            }
            ppo.torch.save(payload, valid_path)

            invalid_payload = dict(payload)
            incomplete_vocab = dict(state_vocab)
            incomplete_vocab.pop("device_type=unknown")
            invalid_payload["state_vocab"] = incomplete_vocab
            invalid_model = ppo.ActorCritic(
                len(incomplete_vocab), len(action_vocab), (8,)
            )
            invalid_payload["model_state_dict"] = invalid_model.state_dict()
            ppo.torch.save(invalid_payload, invalid_path)

            malformed_index_payload = dict(payload)
            malformed_vocab = dict(state_vocab)
            malformed_vocab["schema_version=2"] = "0"
            malformed_index_payload["state_vocab"] = malformed_vocab
            ppo.torch.save(malformed_index_payload, malformed_index_path)

            valid_payload, valid_model = ppo._load_checkpoint(valid_path)
            self.assertIsNotNone(valid_payload)
            self.assertIsNotNone(valid_model)
            invalid_loaded, invalid_model_loaded = ppo._load_checkpoint(invalid_path)
            self.assertIsNone(invalid_loaded)
            self.assertIsNone(invalid_model_loaded)
            malformed_loaded, malformed_model = ppo._load_checkpoint(
                malformed_index_path
            )
            self.assertIsNone(malformed_loaded)
            self.assertIsNone(malformed_model)

    @unittest.skipIf(ppo.torch is None, "PyTorch is not installed")
    def test_checkpoint_rejects_malformed_model_architecture(self) -> None:
        state_vocab = {
            token: index
            for index, token in enumerate(required_ppo_state_tokens())
        }
        action_vocab = {
            action: index for index, action in enumerate(ALL_ACTIONS)
        }
        model = ppo.ActorCritic(
            len(state_vocab),
            len(action_vocab),
            (8,),
        )
        payload = {
            "state_vocab": state_vocab,
            "action_vocab": action_vocab,
            "hidden_sizes": (8,),
            "model_state_dict": model.state_dict(),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            malformed_hidden_path = root / "malformed-hidden.pt"
            malformed_shape_path = root / "malformed-shape.pt"

            malformed_hidden = dict(payload)
            malformed_hidden["hidden_sizes"] = ("8",)
            ppo.torch.save(malformed_hidden, malformed_hidden_path)

            malformed_shape = dict(payload)
            malformed_shape["model_state_dict"] = {
                **model.state_dict(),
                "backbone.0.weight": model.state_dict()[
                    "backbone.0.weight"
                ][:, :-1],
            }
            ppo.torch.save(malformed_shape, malformed_shape_path)

            for path in (malformed_hidden_path, malformed_shape_path):
                with self.subTest(path=path.name):
                    loaded_payload, loaded_model = ppo._load_checkpoint(path)
                    self.assertIsNone(loaded_payload)
                    self.assertIsNone(loaded_model)


if __name__ == "__main__":
    unittest.main()
