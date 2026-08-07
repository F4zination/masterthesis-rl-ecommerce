"""Finite training/serving policy contracts and artifact validation helpers.

The production normalizers emit values from the domains declared in
``shared_schema.constants``.  These helpers turn that declaration into exact
PPO vocabulary and V2 contextual-bandit requirements so training, startup, and
release checks cannot silently disagree about what "complete" means.
"""

from __future__ import annotations

import json
import math
import sqlite3
from itertools import product
from typing import Any, Callable, Mapping

from .constants import (
    ALL_ACTIONS,
    CONTEXT_SCHEMA_VERSION,
    FEATURE_VALUE_DOMAINS,
    HISTORY_PRIMED_CREDIT_DECAY,
    HISTORY_PRIMING_ACTIONS,
    HISTORY_PRIMING_AMOUNT,
)
from .features import POINT_ACTIONS, POINT_FEATURES


class PolicyContractError(ValueError):
    """A policy artifact cannot represent the finite serving domain."""


def required_ppo_state_tokens(
    context_schema_version: int = CONTEXT_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Return every scalar token the production PPO encoder can emit."""
    tokens: list[str] = []
    for feature, values in FEATURE_VALUE_DOMAINS.items():
        domain = (
            (context_schema_version,)
            if feature == "schema_version"
            else values
        )
        tokens.extend(f"{feature}={value}" for value in domain)
    return tuple(tokens)


def validate_ppo_vocabulary(
    state_vocab: Mapping[str, Any],
    action_vocab: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate complete token/action coverage and dense unique indices."""
    required_tokens = set(required_ppo_state_tokens())
    required_actions = set(ALL_ACTIONS)
    state_tokens = {str(token) for token in state_vocab}
    action_tokens = {str(action) for action in action_vocab}

    if any(type(value) is not int for value in state_vocab.values()) or any(
        type(value) is not int for value in action_vocab.values()
    ):
        raise PolicyContractError(
            "PPO vocabulary indices must be integers"
        )
    state_indices = list(state_vocab.values())
    action_indices = list(action_vocab.values())

    missing_tokens = sorted(required_tokens - state_tokens)
    missing_actions = sorted(required_actions - action_tokens)
    state_indices_valid = (
        len(set(state_indices)) == len(state_indices)
        and set(state_indices) == set(range(len(state_vocab)))
    )
    action_indices_valid = (
        len(set(action_indices)) == len(action_indices)
        and set(action_indices) == set(range(len(action_vocab)))
    )

    errors: list[str] = []
    if missing_tokens:
        errors.append(f"missing state tokens: {missing_tokens}")
    if missing_actions:
        errors.append(f"missing actions: {missing_actions}")
    if not state_indices_valid:
        errors.append("state-vocabulary indices are not dense and unique")
    if not action_indices_valid:
        errors.append("action-vocabulary indices are not dense and unique")
    if errors:
        raise PolicyContractError("invalid PPO serving vocabulary: " + "; ".join(errors))

    return {
        "required_state_token_count": len(required_tokens),
        "required_action_count": len(required_actions),
        "state_vocab_size": len(state_vocab),
        "action_vocab_size": len(action_vocab),
        "missing_state_tokens": [],
        "missing_actions": [],
    }


def validate_and_load_ppo_checkpoint_model(
    payload: Mapping[str, Any],
    model_factory: Callable[[int, int, tuple[int, ...]], Any],
) -> tuple[Any, dict[str, Any]]:
    """Instantiate and strictly load the model described by a PPO checkpoint.

    ``model_factory`` keeps this shared contract independent of PyTorch: the
    offline validator and V3 runtime each pass their existing ``ActorCritic``
    implementation. Both callers consequently enforce the same vocabulary,
    architecture-metadata, and strict state-dict compatibility checks.
    """
    state_vocab = payload.get("state_vocab")
    action_vocab = payload.get("action_vocab")
    if not isinstance(state_vocab, Mapping) or not isinstance(
        action_vocab,
        Mapping,
    ):
        raise PolicyContractError(
            "PPO checkpoint does not contain mapping state/action vocabularies"
        )
    validate_ppo_vocabulary(state_vocab, action_vocab)

    raw_hidden_sizes = payload.get("hidden_sizes")
    if raw_hidden_sizes is None or (
        isinstance(raw_hidden_sizes, (list, tuple))
        and not raw_hidden_sizes
    ):
        # Preserve compatibility with the original V3 loader's default for
        # checkpoints that predate explicit architecture metadata.
        raw_hidden_sizes = (128, 64)
    if not isinstance(raw_hidden_sizes, (list, tuple)) or any(
        type(value) is not int or value <= 0
        for value in raw_hidden_sizes
    ):
        raise PolicyContractError(
            "PPO checkpoint hidden_sizes must be a list or tuple of "
            "positive integers"
        )
    hidden_sizes = tuple(raw_hidden_sizes)

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise PolicyContractError(
            "PPO checkpoint does not contain a mapping model_state_dict"
        )

    try:
        model = model_factory(
            len(state_vocab),
            len(action_vocab),
            hidden_sizes,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception as exc:
        raise PolicyContractError(
            "PPO checkpoint architecture/state_dict cannot be loaded: "
            f"{exc}"
        ) from exc

    return model, {
        "state_vocab_size": len(state_vocab),
        "action_vocab_size": len(action_vocab),
        "hidden_sizes": list(hidden_sizes),
        "state_dict_key_count": len(state_dict),
        "strict_state_dict_load": True,
    }


def validate_ppo_study_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require the trained checkpoint configuration used by the V3 treatment."""
    expected = {
        "algorithm": "ppo_clip_actor_critic",
        "mode": "online",
        "sequential": True,
        "timing_mode": "opportunity",
    }
    errors = [
        f"{key} must be {value!r}, found {payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]

    provenance = payload.get("provenance")
    summary = payload.get("summary")
    coverage = payload.get("vocabulary_coverage")
    if not isinstance(provenance, Mapping):
        errors.append("provenance metadata is missing")
        provenance = {}
    if not isinstance(summary, Mapping):
        errors.append("training summary is missing")
        summary = {}
    if not isinstance(coverage, Mapping):
        errors.append("vocabulary coverage metadata is missing")
        coverage = {}

    for key in ("mode", "sequential", "timing_mode"):
        if provenance.get(key) != expected[key]:
            errors.append(
                f"provenance {key} must be {expected[key]!r}, "
                f"found {provenance.get(key)!r}"
            )
    if summary.get("mode") != "online" or summary.get("sequential") is not True:
        errors.append("training summary does not describe online sequential PPO")

    for key in ("iterations", "rollout_sessions"):
        value = summary.get(key)
        provenance_value = provenance.get(key)
        if type(value) is not int or value <= 0:
            errors.append(f"summary {key} must be a positive integer")
        if provenance_value != value:
            errors.append(
                f"provenance {key} does not match the training summary"
            )
    n_rows = summary.get("n_rows")
    if type(n_rows) is not int or n_rows <= 0:
        errors.append("summary n_rows must be a positive integer")

    if provenance.get("state_vocab_domain_version") != 1:
        errors.append("unsupported or missing state_vocab_domain_version")
    if provenance.get("unobserved_token_initialization") != "zero_input_columns":
        errors.append("unobserved token initialization is not fail-safe")
    if coverage.get("domain_version") != 1:
        errors.append("vocabulary coverage domain_version is not 1")
    if coverage.get("unobserved_token_initialization") != "zero_input_columns":
        errors.append("vocabulary coverage does not record neutral initialization")
    if coverage.get("missing_state_tokens") not in ([], ()):
        errors.append("vocabulary coverage reports missing state tokens")
    if coverage.get("missing_actions") not in ([], ()):
        errors.append("vocabulary coverage reports missing actions")

    dynamics = provenance.get("dynamics")
    if not isinstance(dynamics, Mapping):
        errors.append("provenance dynamics metadata is missing")
        dynamics = {}
    expected_history = {
        "delayed_reward_decay": HISTORY_PRIMED_CREDIT_DECAY,
        "priming_actions": list(HISTORY_PRIMING_ACTIONS),
        "priming_amount": HISTORY_PRIMING_AMOUNT,
    }
    for key, expected_value in expected_history.items():
        if dynamics.get(key) != expected_value:
            errors.append(
                f"history recurrence {key} must be {expected_value!r}, "
                f"found {dynamics.get(key)!r}"
            )

    dataset_sha256 = provenance.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256.lower())
    ):
        errors.append("provenance dataset_sha256 is missing or invalid")
    git_commit = provenance.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit.lower())
    ):
        errors.append("provenance git_commit is missing or invalid")
    git_dirty = provenance.get("git_dirty")
    if type(git_dirty) is not bool:
        errors.append("provenance git_dirty must be a boolean")

    if errors:
        raise PolicyContractError(
            "invalid PPO study-treatment metadata: " + "; ".join(errors)
        )
    return {
        **expected,
        "iterations": summary["iterations"],
        "rollout_sessions": summary["rollout_sessions"],
        "n_rows": summary["n_rows"],
        "dataset_sha256": dataset_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "state_vocab_domain_version": 1,
        "unobserved_token_initialization": "zero_input_columns",
        "history_recurrence": expected_history,
    }


def required_bandit_context_keys() -> set[tuple[str, str]]:
    """Return every normalized V2 context key production can emit."""
    contexts: set[tuple[str, str]] = set()
    for decision_point, features in POINT_FEATURES.items():
        try:
            domains = [
                tuple(str(value) for value in FEATURE_VALUE_DOMAINS[feature])
                for feature in features
            ]
        except KeyError as exc:
            raise PolicyContractError(
                f"no finite domain for bandit feature {exc.args[0]!r}"
            ) from exc
        for values in product(*domains):
            contexts.add(
                (decision_point, "|".join((decision_point, *values)))
            )
    return contexts


def required_bandit_arm_keys() -> set[tuple[str, str, str]]:
    """Return every V2 ``(point, context_key, action)`` serving arm."""
    return {
        (decision_point, context_key, str(action))
        for decision_point, context_key in required_bandit_context_keys()
        for action in POINT_ACTIONS[decision_point]
    }


def _required_table(
    connection: sqlite3.Connection,
    table: str,
    columns: set[str],
) -> None:
    actual = {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    if not actual:
        raise PolicyContractError(f"required table {table!r} is missing")
    missing = sorted(columns - actual)
    if missing:
        raise PolicyContractError(
            f"required table {table!r} is missing columns {missing}"
        )


def _load_policy_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    _required_table(connection, "policy_build_metadata", {"key", "value"})
    metadata: dict[str, Any] = {}
    for key, value in connection.execute(
        "SELECT key, value FROM policy_build_metadata"
    ):
        try:
            metadata[str(key)] = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise PolicyContractError(
                f"policy metadata {key!r} is not valid JSON"
            ) from exc
    return metadata


def validate_bandit_policy_connection(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Validate an on-disk V2 database against the complete serving grid."""
    quick_check = connection.execute("PRAGMA quick_check").fetchone()
    if not quick_check or quick_check[0] != "ok":
        raise PolicyContractError(
            f"SQLite integrity check failed: {quick_check!r}"
        )

    _required_table(
        connection,
        "bandit_arm_stats",
        {
            "decision_point",
            "context_key",
            "action",
            "impressions",
            "reward_sum",
            "schema_version",
        },
    )
    _required_table(
        connection,
        "policy_arm_support",
        {
            "decision_point",
            "context_key",
            "action",
            "support_kind",
            "raw_empirical_impressions",
            "pooled_context_empirical_impressions",
            "backoff_empirical_impressions",
            "pseudocount_impressions",
        },
    )

    expected = required_bandit_arm_keys()
    arm_rows = connection.execute(
        "SELECT decision_point, context_key, action, impressions, reward_sum, "
        "schema_version FROM bandit_arm_stats"
    ).fetchall()
    arm_keys = {
        (str(point), str(context_key), str(action))
        for point, context_key, action, *_ in arm_rows
    }
    if len(arm_rows) != len(arm_keys):
        raise PolicyContractError("bandit_arm_stats contains duplicate serving arms")
    missing_arms = expected - arm_keys
    extra_arms = arm_keys - expected
    if missing_arms or extra_arms:
        raise PolicyContractError(
            "bandit serving-grid mismatch "
            f"(missing_arms={len(missing_arms)}, extra_arms={len(extra_arms)}, "
            f"expected_arms={len(expected)}, actual_arms={len(arm_keys)})"
        )
    arm_impressions: dict[tuple[str, str, str], int] = {}
    for point, context_key, action, impressions, reward_sum, schema_version in arm_rows:
        try:
            valid_stats = int(impressions) > 0 and math.isfinite(float(reward_sum))
            valid_schema = int(schema_version) == CONTEXT_SCHEMA_VERSION
        except (TypeError, ValueError):
            valid_stats = valid_schema = False
        if not valid_stats or not valid_schema:
            raise PolicyContractError(
                "invalid bandit arm statistics/schema for "
                f"{point!r}/{context_key!r}/{action!r}"
            )
        arm_impressions[(str(point), str(context_key), str(action))] = int(
            impressions
        )

    support_rows = connection.execute(
        "SELECT decision_point, context_key, action, support_kind, "
        "raw_empirical_impressions, pooled_context_empirical_impressions, "
        "backoff_empirical_impressions, pseudocount_impressions "
        "FROM policy_arm_support"
    ).fetchall()
    support_keys = {
        (str(point), str(context_key), str(action))
        for point, context_key, action, *_ in support_rows
    }
    if len(support_rows) != len(support_keys):
        raise PolicyContractError("policy_arm_support contains duplicate serving arms")
    missing_support = expected - support_keys
    extra_support = support_keys - expected
    if missing_support or extra_support:
        raise PolicyContractError(
            "bandit support-grid mismatch "
            f"(missing_support={len(missing_support)}, "
            f"extra_support={len(extra_support)})"
        )

    support_counts = {
        "device_pooled_empirical": 0,
        "decision_point_action_pseudocount": 0,
    }
    for row in support_rows:
        point, context_key, action, kind, raw, pooled, backoff, pseudo = row
        try:
            raw_i, pooled_i = int(raw), int(pooled)
            backoff_i, pseudo_i = int(backoff), int(pseudo)
        except (TypeError, ValueError) as exc:
            raise PolicyContractError(
                f"non-integer support accounting for {point!r}/{context_key!r}/{action!r}"
            ) from exc
        if kind == "device_pooled_empirical":
            valid = (
                raw_i >= 0
                and pooled_i > 0
                and raw_i <= pooled_i
                and backoff_i == 0
                and pseudo_i == 0
                and arm_impressions[(str(point), str(context_key), str(action))]
                == pooled_i
            )
        elif kind == "decision_point_action_pseudocount":
            valid = (
                raw_i == 0
                and pooled_i == 0
                and backoff_i > 0
                and pseudo_i > 0
                and arm_impressions[(str(point), str(context_key), str(action))]
                == pseudo_i
            )
        else:
            valid = False
        if not valid:
            raise PolicyContractError(
                f"invalid support accounting for {point!r}/{context_key!r}/{action!r}"
            )
        support_counts[str(kind)] += 1

    metadata = _load_policy_metadata(connection)
    arm_support = metadata.get("arm_support")
    if not isinstance(arm_support, dict):
        raise PolicyContractError("policy metadata has no arm_support manifest")
    required_metadata = {
        "coverage_validation_passed": True,
        "required_serving_arm_rows": len(expected),
        "materialized_serving_arm_rows": len(expected),
        "per_arm_support_table": "policy_arm_support",
    }
    mismatches = {
        key: {"expected": value, "actual": arm_support.get(key)}
        for key, value in required_metadata.items()
        if arm_support.get(key) != value
    }
    expected_contexts = len(required_bandit_context_keys())
    if arm_support.get("required_serving_contexts") != expected_contexts:
        mismatches["required_serving_contexts"] = {
            "expected": expected_contexts,
            "actual": arm_support.get("required_serving_contexts"),
        }
    if metadata.get("context_schema_version") != CONTEXT_SCHEMA_VERSION:
        mismatches["context_schema_version"] = {
            "expected": CONTEXT_SCHEMA_VERSION,
            "actual": metadata.get("context_schema_version"),
        }
    if mismatches:
        raise PolicyContractError(
            "bandit policy metadata does not attest complete coverage: "
            f"{mismatches}"
        )

    return {
        "quick_check": "ok",
        "serving_context_count": expected_contexts,
        "serving_arm_count": len(expected),
        "support_counts": support_counts,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
    }
