from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_schema.constants import ALL_ACTIONS
from shared_schema.policy_contract import (
    PolicyContractError,
    required_bandit_arm_keys,
    required_bandit_context_keys,
    required_ppo_state_tokens,
    validate_ppo_vocabulary,
)


class PolicyContractTests(unittest.TestCase):
    def test_declared_serving_domains_have_expected_sizes(self) -> None:
        self.assertEqual(len(required_ppo_state_tokens()), 46)
        self.assertEqual(len(required_bandit_context_keys()), 848)
        self.assertEqual(len(required_bandit_arm_keys()), 5_088)

    def test_ppo_vocabulary_must_cover_every_required_token(self) -> None:
        tokens = required_ppo_state_tokens()
        state_vocab = {token: index for index, token in enumerate(tokens)}
        action_vocab = {
            action: index for index, action in enumerate(ALL_ACTIONS)
        }
        report = validate_ppo_vocabulary(state_vocab, action_vocab)
        self.assertEqual(report["required_state_token_count"], 46)

        state_vocab.pop("device_type=unknown")
        with self.assertRaisesRegex(PolicyContractError, "device_type=unknown"):
            validate_ppo_vocabulary(state_vocab, action_vocab)

    def test_ppo_vocabulary_rejects_coercible_non_integer_indices(self) -> None:
        state_vocab = {
            token: index
            for index, token in enumerate(required_ppo_state_tokens())
        }
        action_vocab = {
            action: index for index, action in enumerate(ALL_ACTIONS)
        }
        state_vocab["schema_version=2"] = "0"
        with self.assertRaisesRegex(PolicyContractError, "must be integers"):
            validate_ppo_vocabulary(state_vocab, action_vocab)


if __name__ == "__main__":
    unittest.main()
