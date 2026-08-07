#!/usr/bin/env python3
"""Build a frozen V2 contextual-bandit policy from simulated transitions.

The frozen DemoSiteV2 policy is stored in ``bandit_arm_stats`` rows keyed by
``(decision_point, context_key, action)``.  Raw simulation transitions provide
empirical impressions and rewards, but a finite simulation sample does not
necessarily visit every context/action that the production normalizer can
emit.

This builder closes that training/serving gap explicitly:

* ``device_type`` is exogenous and behaviorally irrelevant in the simulator,
  so observations are pooled across device values and the pooled estimate is
  materialized for every valid serving value (including ``unknown``).
* Every remaining cell in the finite ``POINT_FEATURES`` grid receives a
  one-count empirical-Bayes backoff using the observed mean for the same
  decision point and action.
* ``policy_arm_support`` records the support source for every materialized arm,
  while ``policy_build_metadata`` records aggregate empirical, augmented, and
  pseudocount totals.  Pseudocounts are therefore never presented as raw
  simulation impressions.
* Coverage is validated before an output database is created.  A missing arm,
  invalid support record, unknown feature domain, or unavailable backoff source
  fails the build.

The output is a fully seeded DemoSiteV2 SQLite database ready to mount at
``/app/data/demosite.db``.

Usage:
    python build_bandit_policy.py \
        --dataset ../CustomerSimulation/output/offline_transitions.jsonl \
        --out ../DemoSiteV2/data/demosite.db --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_SCHEMA_PATH = str(REPO_ROOT / "SharedSchema")
if SHARED_SCHEMA_PATH not in sys.path:
    sys.path.insert(0, SHARED_SCHEMA_PATH)

from shared_schema.constants import DEVICE_TYPES, FEATURE_VALUE_DOMAINS  # noqa: E402

DEVICE_FEATURE = "device_type"

PSEUDOCOUNT_IMPRESSIONS = 1
DEVICE_POOLED_SUPPORT = "device_pooled_empirical"
BACKOFF_SUPPORT = "decision_point_action_pseudocount"

ArmKey = tuple[str, str, str]
ValueArmKey = tuple[str, tuple[str, ...], str]


@dataclass(frozen=True)
class ArmStats:
    """Effective statistics written to one ``bandit_arm_stats`` row."""

    impressions: int
    reward_sum: float


@dataclass(frozen=True)
class ArmSupport:
    """Auditable origin of the effective statistics for one materialized arm."""

    support_kind: str
    raw_empirical_impressions: int
    pooled_context_empirical_impressions: int
    backoff_empirical_impressions: int
    pseudocount_impressions: int


@dataclass
class CompletePolicy:
    """A fully materialized and coverage-validated finite bandit policy."""

    arms: dict[ArmKey, ArmStats]
    support: dict[ArmKey, ArmSupport]
    required_arm_keys: set[ArmKey]
    required_context_keys: set[tuple[str, str]]
    n_transitions_used: int
    n_transitions_skipped: int
    per_point_transitions: dict[str, int]
    raw_empirical_arm_rows: int
    pooled_empirical_base_arm_rows: int
    observed_device_values: tuple[str, ...]
    feature_domains: dict[str, tuple[str, ...]]


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to the current working directory."""
    path = Path(path_str)
    return path if path.is_absolute() else (Path.cwd() / path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    safe_repo = str(REPO_ROOT.resolve()).replace("\\", "/")
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={safe_repo}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", f"safe.directory={safe_repo}", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_policy_metadata(
    db_path: Path,
    *,
    dataset_path: Path,
    n_rows: int,
    n_skipped: int,
    schema_version: int,
    seed_products: bool,
    arm_support: Mapping[str, Any] | None = None,
    git_state: tuple[str | None, bool | None] | None = None,
) -> None:
    """Embed reproducible build provenance inside the generated policy DB."""
    git_commit, git_dirty = git_state if git_state is not None else _git_state()
    metadata: dict[str, Any] = {
        "provenance_schema_version": 2,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "build_git_commit": git_commit,
        "build_git_dirty": git_dirty,
        "build_script_path": str(Path(__file__).resolve()),
        "build_script_sha256": _sha256(Path(__file__).resolve()),
        "n_transitions_used": n_rows,
        "n_transitions_skipped": n_skipped,
        "context_schema_version": schema_version,
        "product_catalog_seeded": seed_products,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if arm_support is not None:
        metadata["arm_support"] = dict(arm_support)

    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS policy_build_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT OR REPLACE INTO policy_build_metadata(key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in metadata.items()
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _write_arm_support(
    db_path: Path,
    support: Mapping[ArmKey, ArmSupport],
) -> None:
    """Write per-arm empirical/augmented support accounting.

    This table is intentionally separate from ``bandit_arm_stats``: serving
    continues to consume the effective statistics without a schema change,
    while audits can distinguish raw observations, device pooling, and
    empirical-Bayes pseudocounts for every individual arm.
    """
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_arm_support (
                decision_point TEXT NOT NULL,
                context_key TEXT NOT NULL,
                action TEXT NOT NULL,
                support_kind TEXT NOT NULL,
                raw_empirical_impressions INTEGER NOT NULL,
                pooled_context_empirical_impressions INTEGER NOT NULL,
                backoff_empirical_impressions INTEGER NOT NULL,
                pseudocount_impressions INTEGER NOT NULL,
                PRIMARY KEY (decision_point, context_key, action)
            )
            """
        )
        connection.execute("DELETE FROM policy_arm_support")
        connection.executemany(
            """
            INSERT INTO policy_arm_support (
                decision_point,
                context_key,
                action,
                support_kind,
                raw_empirical_impressions,
                pooled_context_empirical_impressions,
                backoff_empirical_impressions,
                pseudocount_impressions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    decision_point,
                    context_key,
                    action,
                    row.support_kind,
                    row.raw_empirical_impressions,
                    row.pooled_context_empirical_impressions,
                    row.backoff_empirical_impressions,
                    row.pseudocount_impressions,
                )
                for (decision_point, context_key, action), row in sorted(
                    support.items()
                )
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects with line-numbered errors for malformed input."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: transition must be an object")
            yield record


def _accumulate(
    rows: dict[ValueArmKey | tuple[str, str], list[float]],
    key: ValueArmKey | tuple[str, str],
    reward: float,
) -> None:
    row = rows[key]
    row[0] += 1
    row[1] += reward


def _validate_schema_domains(
    point_features: Mapping[str, Sequence[str]],
    point_actions: Mapping[str, Sequence[str]],
    feature_domains: Mapping[str, Sequence[str]],
) -> None:
    if not point_features:
        raise ValueError("POINT_FEATURES is empty")

    for decision_point, features in point_features.items():
        if DEVICE_FEATURE not in features:
            raise ValueError(
                f"{decision_point!r} does not include {DEVICE_FEATURE!r}; "
                "device-domain augmentation cannot be applied safely"
            )
        if not point_actions.get(decision_point):
            raise ValueError(f"{decision_point!r} has no eligible action domain")
        for feature in features:
            domain = tuple(str(value) for value in feature_domains.get(feature, ()))
            if not domain:
                raise ValueError(
                    f"no finite serving domain declared for feature {feature!r}"
                )
            if len(domain) != len(set(domain)):
                raise ValueError(f"serving domain for {feature!r} contains duplicates")


def _required_grid_keys(
    point_features: Mapping[str, Sequence[str]],
    point_actions: Mapping[str, Sequence[str]],
    normalized_domains: Mapping[str, Sequence[str]],
) -> tuple[set[tuple[str, str]], set[ArmKey]]:
    """Enumerate the exact finite serving contexts and context/action cells."""
    required_context_keys: set[tuple[str, str]] = set()
    required_arm_keys: set[ArmKey] = set()
    for decision_point, features in point_features.items():
        domains = [normalized_domains[feature] for feature in features]
        actions = tuple(str(value) for value in point_actions[decision_point])
        for values in product(*domains):
            context_key = "|".join((decision_point, *values))
            required_context_keys.add((decision_point, context_key))
            required_arm_keys.update(
                (decision_point, context_key, action) for action in actions
            )
    return required_context_keys, required_arm_keys


def _build_complete_policy(
    transitions: Iterable[Mapping[str, Any]],
    *,
    point_features: Mapping[str, Sequence[str]],
    point_actions: Mapping[str, Sequence[str]],
    expected_schema_version: int,
    feature_domains: Mapping[str, Sequence[str | int]] = FEATURE_VALUE_DOMAINS,
) -> CompletePolicy:
    """Aggregate transitions and materialize the complete finite serving grid."""
    _validate_schema_domains(point_features, point_actions, feature_domains)
    normalized_domains = {
        feature: tuple(str(value) for value in domain)
        for feature, domain in feature_domains.items()
    }
    if normalized_domains.get(DEVICE_FEATURE) != tuple(
        str(value) for value in DEVICE_TYPES
    ):
        raise ValueError(
            "device_type serving domain does not match shared DEVICE_TYPES"
        )

    raw_rows: dict[ValueArmKey, list[float]] = defaultdict(lambda: [0, 0.0])
    pooled_rows: dict[ValueArmKey, list[float]] = defaultdict(lambda: [0, 0.0])
    point_action_rows: dict[tuple[str, str], list[float]] = defaultdict(
        lambda: [0, 0.0]
    )
    per_point: dict[str, int] = defaultdict(int)
    observed_devices: set[str] = set()
    n_rows = 0
    n_skipped = 0

    for transition_index, transition in enumerate(transitions, start=1):
        decision_point = str(transition.get("decision_point") or "")
        features = point_features.get(decision_point)
        state = transition.get("state")
        if features is None or not isinstance(state, Mapping):
            n_skipped += 1
            continue
        missing = [feature for feature in features if feature not in state]
        if missing:
            n_skipped += 1
            continue

        try:
            schema_version = int(
                state.get("schema_version", expected_schema_version)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"transition {transition_index}: invalid schema_version"
            ) from exc
        if schema_version != expected_schema_version:
            raise ValueError(
                f"transition {transition_index}: schema_version {schema_version} "
                f"does not match serving version {expected_schema_version}"
            )

        values = tuple(str(state[feature]) for feature in features)
        for feature, value in zip(features, values):
            if value not in normalized_domains[feature]:
                raise ValueError(
                    f"transition {transition_index}: value {value!r} is outside "
                    f"the serving domain for {feature!r}"
                )

        action = str(transition.get("action") or "")
        eligible_actions = tuple(str(value) for value in point_actions[decision_point])
        if action not in eligible_actions:
            raise ValueError(
                f"transition {transition_index}: action {action!r} is not eligible "
                f"at {decision_point!r}"
            )
        try:
            reward = float(transition["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"transition {transition_index}: missing or invalid reward"
            ) from exc
        if not math.isfinite(reward):
            raise ValueError(
                f"transition {transition_index}: reward must be finite"
            )

        device_index = list(features).index(DEVICE_FEATURE)
        device_value = values[device_index]
        base_values = tuple(
            value
            for feature, value in zip(features, values)
            if feature != DEVICE_FEATURE
        )

        _accumulate(raw_rows, (decision_point, values, action), reward)
        _accumulate(pooled_rows, (decision_point, base_values, action), reward)
        _accumulate(point_action_rows, (decision_point, action), reward)
        observed_devices.add(device_value)
        per_point[decision_point] += 1
        n_rows += 1

    if n_rows == 0:
        raise ValueError("dataset contains no usable serving-schema transitions")

    # A point/action mean is the explicit backoff source for every structurally
    # unseen context.  Refuse to invent a value when no empirical source exists.
    missing_backoffs = [
        (decision_point, action)
        for decision_point, actions in point_actions.items()
        for action in actions
        if point_action_rows[(decision_point, str(action))][0] <= 0
    ]
    if missing_backoffs:
        preview = ", ".join(f"{dp}/{action}" for dp, action in missing_backoffs[:8])
        raise ValueError(
            "cannot materialize complete coverage: no empirical backoff source "
            f"for {preview}"
        )

    raw_empirical_arm_rows = len(raw_rows)
    pooled_empirical_base_arm_rows = len(pooled_rows)
    arms: dict[ArmKey, ArmStats] = {}
    support: dict[ArmKey, ArmSupport] = {}
    required_context_keys, required_arm_keys = _required_grid_keys(
        point_features,
        point_actions,
        normalized_domains,
    )

    for decision_point, features in point_features.items():
        feature_domains_for_point = [
            normalized_domains[feature] for feature in features
        ]
        device_index = list(features).index(DEVICE_FEATURE)
        actions = tuple(str(value) for value in point_actions[decision_point])

        for values in product(*feature_domains_for_point):
            context_key = "|".join((decision_point, *values))
            base_values = tuple(
                value
                for feature, value in zip(features, values)
                if feature != DEVICE_FEATURE
            )
            current_device = values[device_index]

            for action in actions:
                arm_key = (decision_point, context_key, action)
                raw_count = int(
                    raw_rows.get(
                        (decision_point, tuple(values), action),
                        (0, 0.0),
                    )[0]
                )
                pooled_count, pooled_reward = pooled_rows.get(
                    (decision_point, base_values, action),
                    (0, 0.0),
                )

                if pooled_count > 0:
                    stats = ArmStats(
                        impressions=int(pooled_count),
                        reward_sum=float(pooled_reward),
                    )
                    support_row = ArmSupport(
                        support_kind=DEVICE_POOLED_SUPPORT,
                        raw_empirical_impressions=raw_count,
                        pooled_context_empirical_impressions=int(pooled_count),
                        backoff_empirical_impressions=0,
                        pseudocount_impressions=0,
                    )
                else:
                    backoff_count, backoff_reward = point_action_rows[
                        (decision_point, action)
                    ]
                    backoff_mean = float(backoff_reward) / int(backoff_count)
                    stats = ArmStats(
                        impressions=PSEUDOCOUNT_IMPRESSIONS,
                        reward_sum=backoff_mean * PSEUDOCOUNT_IMPRESSIONS,
                    )
                    support_row = ArmSupport(
                        support_kind=BACKOFF_SUPPORT,
                        raw_empirical_impressions=0,
                        pooled_context_empirical_impressions=0,
                        backoff_empirical_impressions=int(backoff_count),
                        pseudocount_impressions=PSEUDOCOUNT_IMPRESSIONS,
                    )

                # ``current_device`` is deliberately read above to make the
                # per-device materialization explicit even though the pooled
                # estimate itself is device invariant.
                if current_device not in normalized_domains[DEVICE_FEATURE]:
                    raise AssertionError("materialized an invalid device value")
                arms[arm_key] = stats
                support[arm_key] = support_row

    policy = CompletePolicy(
        arms=arms,
        support=support,
        required_arm_keys=required_arm_keys,
        required_context_keys=required_context_keys,
        n_transitions_used=n_rows,
        n_transitions_skipped=n_skipped,
        per_point_transitions=dict(sorted(per_point.items())),
        raw_empirical_arm_rows=raw_empirical_arm_rows,
        pooled_empirical_base_arm_rows=pooled_empirical_base_arm_rows,
        observed_device_values=tuple(
            value
            for value in normalized_domains[DEVICE_FEATURE]
            if value in observed_devices
        ),
        feature_domains={
            feature: normalized_domains[feature]
            for features in point_features.values()
            for feature in features
        },
    )
    _validate_complete_coverage(policy)
    return policy


def _validate_complete_coverage(policy: CompletePolicy) -> None:
    """Fail if any required serving arm lacks valid, auditable support."""
    arm_keys = set(policy.arms)
    support_keys = set(policy.support)
    missing_arms = policy.required_arm_keys - arm_keys
    extra_arms = arm_keys - policy.required_arm_keys
    missing_support = policy.required_arm_keys - support_keys
    extra_support = support_keys - policy.required_arm_keys
    if missing_arms or extra_arms or missing_support or extra_support:
        raise ValueError(
            "finite-grid coverage mismatch: "
            f"missing_arms={len(missing_arms)}, extra_arms={len(extra_arms)}, "
            f"missing_support={len(missing_support)}, "
            f"extra_support={len(extra_support)}"
        )

    for arm_key in sorted(policy.required_arm_keys):
        stats = policy.arms[arm_key]
        origin = policy.support[arm_key]
        if stats.impressions <= 0 or not math.isfinite(stats.reward_sum):
            raise ValueError(f"arm {arm_key!r} has invalid effective statistics")

        if origin.support_kind == DEVICE_POOLED_SUPPORT:
            if (
                origin.pooled_context_empirical_impressions <= 0
                or origin.pseudocount_impressions != 0
                or stats.impressions
                != origin.pooled_context_empirical_impressions
                or origin.raw_empirical_impressions
                > origin.pooled_context_empirical_impressions
            ):
                raise ValueError(f"arm {arm_key!r} has invalid device-pooled support")
        elif origin.support_kind == BACKOFF_SUPPORT:
            if (
                origin.raw_empirical_impressions != 0
                or origin.pooled_context_empirical_impressions != 0
                or origin.backoff_empirical_impressions <= 0
                or origin.pseudocount_impressions <= 0
                or stats.impressions != origin.pseudocount_impressions
            ):
                raise ValueError(f"arm {arm_key!r} has invalid backoff support")
        else:
            raise ValueError(
                f"arm {arm_key!r} has unknown support kind {origin.support_kind!r}"
            )


def _arm_support_metadata(policy: CompletePolicy) -> dict[str, Any]:
    """Summarize raw, pooled, and pseudocount support without conflation."""
    pooled_rows = [
        row
        for row in policy.support.values()
        if row.support_kind == DEVICE_POOLED_SUPPORT
    ]
    backoff_rows = [
        row
        for row in policy.support.values()
        if row.support_kind == BACKOFF_SUPPORT
    ]
    materialized_empirical_impressions = sum(
        row.pooled_context_empirical_impressions for row in pooled_rows
    )
    materialized_pseudocount_impressions = sum(
        row.pseudocount_impressions for row in backoff_rows
    )
    device_pool_added_impressions = sum(
        row.pooled_context_empirical_impressions - row.raw_empirical_impressions
        for row in pooled_rows
    )

    return {
        "support_accounting_schema_version": 1,
        "coverage_validation_passed": True,
        "coverage_scope": "complete_finite_POINT_FEATURES_serving_grid",
        "device_feature": DEVICE_FEATURE,
        "device_domain": list(policy.feature_domains[DEVICE_FEATURE]),
        "observed_device_values": list(policy.observed_device_values),
        "device_pooling_assumption": (
            "device_type is exogenous and behaviorally irrelevant in the simulator"
        ),
        "backoff_method": "decision_point_action_empirical_mean",
        "pseudocount_impressions_per_backoff_arm": PSEUDOCOUNT_IMPRESSIONS,
        "feature_domains": {
            feature: list(domain)
            for feature, domain in sorted(policy.feature_domains.items())
        },
        "raw_empirical_transition_impressions": policy.n_transitions_used,
        "raw_empirical_arm_rows": policy.raw_empirical_arm_rows,
        "pooled_empirical_base_arm_rows": policy.pooled_empirical_base_arm_rows,
        "required_serving_contexts": len(policy.required_context_keys),
        "required_serving_arm_rows": len(policy.required_arm_keys),
        "materialized_serving_arm_rows": len(policy.arms),
        "device_pooled_empirical_arm_rows": len(pooled_rows),
        "device_augmented_without_exact_empirical_arm_rows": sum(
            row.raw_empirical_impressions == 0 for row in pooled_rows
        ),
        "decision_point_action_pseudocount_arm_rows": len(backoff_rows),
        "materialized_empirical_effective_impressions": (
            materialized_empirical_impressions
        ),
        "device_pool_added_effective_impressions": device_pool_added_impressions,
        "materialized_pseudocount_effective_impressions": (
            materialized_pseudocount_impressions
        ),
        "materialized_total_effective_impressions": (
            materialized_empirical_impressions
            + materialized_pseudocount_impressions
        ),
        "per_arm_support_table": "policy_arm_support",
    }


def validate_persisted_policy(
    db_path: Path,
    *,
    point_features: Mapping[str, Sequence[str]] | None = None,
    point_actions: Mapping[str, Sequence[str]] | None = None,
    expected_schema_version: int | None = None,
    feature_domains: Mapping[
        str, Sequence[str | int]
    ] = FEATURE_VALUE_DOMAINS,
) -> dict[str, int | bool]:
    """Validate a built database against the complete shared serving contract.

    The database is opened read-only so a missing or malformed file cannot be
    silently created or repaired by the audit.  This helper is deliberately
    public: release/runtime tooling can reuse the exact fail-closed validation
    the builder runs immediately before publication.
    """
    if point_features is None or point_actions is None:
        from shared_schema.features import POINT_ACTIONS, POINT_FEATURES

        point_features = POINT_FEATURES if point_features is None else point_features
        point_actions = POINT_ACTIONS if point_actions is None else point_actions
    if expected_schema_version is None:
        from shared_schema.constants import CONTEXT_SCHEMA_VERSION

        expected_schema_version = CONTEXT_SCHEMA_VERSION

    _validate_schema_domains(point_features, point_actions, feature_domains)
    normalized_domains = {
        feature: tuple(str(value) for value in domain)
        for feature, domain in feature_domains.items()
    }
    if normalized_domains.get(DEVICE_FEATURE) != tuple(
        str(value) for value in DEVICE_TYPES
    ):
        raise ValueError(
            "device_type serving domain does not match shared DEVICE_TYPES"
        )
    required_context_keys, required_arm_keys = _required_grid_keys(
        point_features,
        point_actions,
        normalized_domains,
    )

    path = Path(db_path).resolve()
    if not path.is_file():
        raise ValueError(f"policy database is missing: {path}")

    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {
            "bandit_arm_stats",
            "policy_arm_support",
            "policy_build_metadata",
        }
        missing_tables = required_tables - tables
        if missing_tables:
            raise ValueError(
                "persisted policy is missing required tables: "
                + ", ".join(sorted(missing_tables))
            )

        persisted_arm_rows = connection.execute(
            """
            SELECT decision_point, context_key, action, impressions,
                   reward_sum, schema_version
            FROM bandit_arm_stats
            """
        ).fetchall()
        arms: dict[ArmKey, ArmStats] = {}
        schema_versions: set[int] = set()
        for decision_point, context_key, action, impressions, reward_sum, version in (
            persisted_arm_rows
        ):
            key = (str(decision_point), str(context_key), str(action))
            arms[key] = ArmStats(int(impressions), float(reward_sum))
            schema_versions.add(int(version))
        if len(arms) != len(persisted_arm_rows):
            raise ValueError("persisted bandit contains duplicate arm keys")
        if schema_versions != {expected_schema_version}:
            raise ValueError(
                "persisted arm schema versions do not match serving version "
                f"{expected_schema_version}: {sorted(schema_versions)}"
            )

        persisted_support_rows = connection.execute(
            """
            SELECT decision_point, context_key, action, support_kind,
                   raw_empirical_impressions,
                   pooled_context_empirical_impressions,
                   backoff_empirical_impressions,
                   pseudocount_impressions
            FROM policy_arm_support
            """
        ).fetchall()
        support: dict[ArmKey, ArmSupport] = {}
        for (
            decision_point,
            context_key,
            action,
            support_kind,
            raw_impressions,
            pooled_impressions,
            backoff_impressions,
            pseudocount_impressions,
        ) in persisted_support_rows:
            key = (str(decision_point), str(context_key), str(action))
            support[key] = ArmSupport(
                support_kind=str(support_kind),
                raw_empirical_impressions=int(raw_impressions),
                pooled_context_empirical_impressions=int(pooled_impressions),
                backoff_empirical_impressions=int(backoff_impressions),
                pseudocount_impressions=int(pseudocount_impressions),
            )
        if len(support) != len(persisted_support_rows):
            raise ValueError("persisted support table contains duplicate arm keys")

        raw_metadata = connection.execute(
            "SELECT key, value FROM policy_build_metadata"
        ).fetchall()
        try:
            metadata = {
                str(key): json.loads(str(value)) for key, value in raw_metadata
            }
        except json.JSONDecodeError as exc:
            raise ValueError("persisted policy metadata contains invalid JSON") from exc
    except (TypeError, ValueError, sqlite3.Error) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"could not validate persisted policy: {exc}") from exc
    finally:
        connection.close()

    persisted_policy = CompletePolicy(
        arms=arms,
        support=support,
        required_arm_keys=required_arm_keys,
        required_context_keys=required_context_keys,
        n_transitions_used=int(metadata.get("n_transitions_used") or 0),
        n_transitions_skipped=int(metadata.get("n_transitions_skipped") or 0),
        per_point_transitions={},
        raw_empirical_arm_rows=0,
        pooled_empirical_base_arm_rows=0,
        observed_device_values=(),
        feature_domains={
            feature: normalized_domains[feature]
            for features in point_features.values()
            for feature in features
        },
    )
    _validate_complete_coverage(persisted_policy)

    if int(metadata.get("provenance_schema_version") or 0) < 2:
        raise ValueError("persisted policy provenance schema is too old")
    if int(metadata.get("context_schema_version") or -1) != expected_schema_version:
        raise ValueError("persisted policy metadata has the wrong context schema")
    support_metadata = metadata.get("arm_support")
    if not isinstance(support_metadata, Mapping):
        raise ValueError("persisted policy has no aggregate arm-support metadata")

    expected_metadata = {
        "required_serving_contexts": len(required_context_keys),
        "required_serving_arm_rows": len(required_arm_keys),
        "materialized_serving_arm_rows": len(arms),
        "device_pooled_empirical_arm_rows": sum(
            row.support_kind == DEVICE_POOLED_SUPPORT for row in support.values()
        ),
        "decision_point_action_pseudocount_arm_rows": sum(
            row.support_kind == BACKOFF_SUPPORT for row in support.values()
        ),
        "materialized_pseudocount_effective_impressions": sum(
            row.pseudocount_impressions for row in support.values()
        ),
    }
    if support_metadata.get("coverage_validation_passed") is not True:
        raise ValueError("persisted policy metadata does not declare valid coverage")
    for key, expected_value in expected_metadata.items():
        persisted_value = support_metadata.get(key)
        try:
            persisted_count = int(persisted_value)
        except (TypeError, ValueError):
            persisted_count = -1
        if persisted_count != expected_value:
            raise ValueError(
                f"persisted support metadata mismatch for {key}: "
                f"expected {expected_value}, found {persisted_value!r}"
            )

    return {
        "coverage_validation_passed": True,
        "serving_contexts": len(required_context_keys),
        "serving_arm_rows": len(required_arm_keys),
        "device_pooled_empirical_arm_rows": expected_metadata[
            "device_pooled_empirical_arm_rows"
        ],
        "decision_point_action_pseudocount_arm_rows": expected_metadata[
            "decision_point_action_pseudocount_arm_rows"
        ],
    }


def _publish_staged_database(
    staged_path: Path,
    out_path: Path,
    *,
    force: bool,
) -> None:
    """Atomically publish a validated sibling database."""
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Re-run with --force to overwrite."
        )
    os.replace(staged_path, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(
            REPO_ROOT / "CustomerSimulation" / "output" / "offline_transitions.jsonl"
        ),
        metavar="PATH",
        help=(
            "Simulation transitions JSONL "
            "(default: CustomerSimulation/output/offline_transitions.jsonl)."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "DemoSiteV2" / "data" / "demosite.db"),
        metavar="PATH",
        help="Output SQLite DB path (default: DemoSiteV2/data/demosite.db).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output DB if it already exists.",
    )
    parser.add_argument(
        "--no-seed-products",
        action="store_true",
        help="Skip catalog seeding (only populate bandit_arm_stats).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve(args.dataset)
    out_path = _resolve(args.out)

    if not dataset_path.exists():
        sys.exit(f"ERROR: dataset not found: {dataset_path}")
    if out_path.exists() and not args.force:
        sys.exit(f"ERROR: {out_path} already exists. Re-run with --force to overwrite.")

    # Shared schema imports are needed to validate and materialize the complete
    # policy before creating (or replacing) an output database.
    from shared_schema.constants import CONTEXT_SCHEMA_VERSION  # noqa: E402
    from shared_schema.features import POINT_ACTIONS, POINT_FEATURES  # noqa: E402

    try:
        policy = _build_complete_policy(
            _iter_jsonl(dataset_path),
            point_features=POINT_FEATURES,
            point_actions=POINT_ACTIONS,
            expected_schema_version=CONTEXT_SCHEMA_VERSION,
        )
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: bandit policy coverage build failed: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture source provenance before creating the sibling staging file so an
    # unignored temporary DB cannot mark its own build as dirty.
    build_git_state = _git_state()
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".building",
        dir=out_path.parent,
    )
    os.close(descriptor)
    staged_path = Path(staged_name)
    engine = None
    try:
        # DemoSiteV2's config reads DATABASE_PATH at import time, so point it at
        # the staging DB before importing the app.
        os.environ["DATABASE_PATH"] = str(staged_path)
        demosite_path = str(REPO_ROOT / "DemoSiteV2")
        if demosite_path not in sys.path:
            sys.path.insert(0, demosite_path)

        from app.database import SessionLocal, engine, init_db  # noqa: E402
        from app.models import BanditArmStat  # noqa: E402
        from app.seed import seed_database  # noqa: E402

        init_db()
        if not args.no_seed_products:
            seed_database(force=True)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        db = SessionLocal()
        try:
            db.bulk_save_objects(
                [
                    BanditArmStat(
                        decision_point=decision_point,
                        context_key=context_key,
                        action=action,
                        impressions=stats.impressions,
                        reward_sum=stats.reward_sum,
                        schema_version=CONTEXT_SCHEMA_VERSION,
                        updated_at=now,
                    )
                    for (decision_point, context_key, action), stats in sorted(
                        policy.arms.items()
                    )
                ]
            )
            db.commit()
        finally:
            db.close()

        _write_arm_support(staged_path, policy.support)
        support_metadata = _arm_support_metadata(policy)
        _write_policy_metadata(
            staged_path,
            dataset_path=dataset_path,
            n_rows=policy.n_transitions_used,
            n_skipped=policy.n_transitions_skipped,
            schema_version=CONTEXT_SCHEMA_VERSION,
            seed_products=not args.no_seed_products,
            arm_support=support_metadata,
            git_state=build_git_state,
        )

        # SQLAlchemy's pool may retain a Windows file handle after sessions are
        # closed.  Dispose it before the read-only audit and atomic replace.
        engine.dispose()
        persisted_validation = validate_persisted_policy(
            staged_path,
            point_features=POINT_FEATURES,
            point_actions=POINT_ACTIONS,
            expected_schema_version=CONTEXT_SCHEMA_VERSION,
        )
        _publish_staged_database(staged_path, out_path, force=args.force)
    except Exception as exc:
        sys.exit(f"ERROR: staged bandit database build failed: {exc}")
    finally:
        if engine is not None:
            engine.dispose()
        if staged_path.exists():
            staged_path.unlink()

    print(
        f"\nBandit policy built from {policy.n_transitions_used} transitions "
        f"({policy.n_transitions_skipped} skipped)"
    )
    print(f"  serving contexts:  {len(policy.required_context_keys)}")
    print(f"  arms written:      {len(policy.arms)}")
    print(
        "  empirical-backed: ",
        persisted_validation["device_pooled_empirical_arm_rows"],
    )
    print(
        "  pseudocount-backed:",
        persisted_validation["decision_point_action_pseudocount_arm_rows"],
    )
    print("  per decision_point:", policy.per_point_transitions)
    print(f"  output DB:         {out_path}")
    print(
        "\nMount it with `make docker-run-v2` "
        "(FREEZE_POLICY=true serves this frozen policy)."
    )


if __name__ == "__main__":
    main()
