#!/usr/bin/env python3
"""Fail-closed validation for a shared policy-training JSONL dataset.

The study artifact workflow calls this before reusing or training from a
dataset.  Validation streams the complete file, checks trajectory integrity
and the serving-state domain, and verifies the sibling summary's exact count
and SHA-256 identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_SCHEMA_ROOT = REPO_ROOT / "SharedSchema"
if str(SHARED_SCHEMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_SCHEMA_ROOT))

from shared_schema.constants import (  # noqa: E402
    ALL_ACTIONS,
    CONTEXT_SCHEMA_VERSION,
    DECISION_POINTS,
    DEVICE_TYPES,
    FEATURE_VALUE_DOMAINS,
    TRAFFIC_SOURCES,
)
from shared_schema.features import POINT_FEATURES  # noqa: E402


DATASET_CONTRACT_VERSION = 1
REQUIRED_ROW_FIELDS = {
    "trajectory_id",
    "t",
    "decision_id",
    "timestamp",
    "session_id",
    "decision_point",
    "state",
    "action",
    "propensity",
    "eligible_actions",
    "reward",
    "reward_without_cost",
    "action_cost",
    "next_state",
    "done",
}
SEQUENTIAL_STATE_FIELDS = {
    "interventions_shown_bucket",
    "steps_since_widget_bucket",
    "session_step_bucket",
    "primed_credit_bucket",
}


class DatasetValidationError(ValueError):
    """The selected training dataset is incomplete or violates its contract."""


def _require_nonempty_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        is_file = resolved.is_file()
        size = resolved.stat().st_size if is_file else 0
    except OSError as exc:
        raise DatasetValidationError(
            f"cannot inspect {label} at {resolved}: {exc}"
        ) from exc
    if not is_file:
        raise DatasetValidationError(f"{label} is missing: {resolved}")
    if size <= 0:
        raise DatasetValidationError(f"{label} is empty: {resolved}")
    return resolved


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DatasetValidationError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _validate_state(
    state: Any,
    *,
    label: str,
    require_sequential: bool,
) -> None:
    if not isinstance(state, dict):
        raise DatasetValidationError(f"{label} must be an object")

    decision_point = state.get("decision_point")
    if decision_point not in DECISION_POINTS:
        raise DatasetValidationError(
            f"{label}.decision_point is unsupported: {decision_point!r}"
        )
    if state.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"{label}.schema_version must be {CONTEXT_SCHEMA_VERSION}"
        )
    if state.get("device_type") not in DEVICE_TYPES:
        raise DatasetValidationError(
            f"{label}.device_type is outside the serving domain: "
            f"{state.get('device_type')!r}"
        )
    if (
        "traffic_source" in POINT_FEATURES[str(decision_point)]
        and state.get("traffic_source") not in TRAFFIC_SOURCES
    ):
        raise DatasetValidationError(
            f"{label}.traffic_source is outside the serving domain: "
            f"{state.get('traffic_source')!r}"
        )

    missing_point_features = set(POINT_FEATURES[str(decision_point)]) - state.keys()
    if missing_point_features:
        raise DatasetValidationError(
            f"{label} is missing point features: "
            f"{sorted(missing_point_features)}"
        )
    if require_sequential:
        missing_history = SEQUENTIAL_STATE_FIELDS - state.keys()
        if missing_history:
            raise DatasetValidationError(
                f"{label} is missing sequential features: "
                f"{sorted(missing_history)}"
            )

    for feature, value in state.items():
        domain = FEATURE_VALUE_DOMAINS.get(feature)
        if domain is None:
            continue
        if isinstance(value, bool) or value not in domain:
            raise DatasetValidationError(
                f"{label}.{feature} is outside the serving domain: {value!r}"
            )


def _validate_row(
    row: Any,
    *,
    line_number: int,
    require_sequential: bool,
) -> tuple[str, int, bool]:
    label = f"dataset line {line_number}"
    if not isinstance(row, dict):
        raise DatasetValidationError(f"{label} must contain a JSON object")
    missing = REQUIRED_ROW_FIELDS - row.keys()
    if missing:
        raise DatasetValidationError(
            f"{label} is missing required fields: {sorted(missing)}"
        )

    session_id = row["session_id"]
    trajectory_id = row["trajectory_id"]
    if not isinstance(session_id, str) or not session_id:
        raise DatasetValidationError(f"{label}.session_id must be nonempty text")
    if trajectory_id != session_id:
        raise DatasetValidationError(
            f"{label}.trajectory_id must equal session_id"
        )
    step = _require_int(row["t"], f"{label}.t")
    if _require_int(row["decision_id"], f"{label}.decision_id") != step:
        raise DatasetValidationError(f"{label}.decision_id must equal t")

    decision_point = row["decision_point"]
    if decision_point not in DECISION_POINTS:
        raise DatasetValidationError(
            f"{label}.decision_point is unsupported: {decision_point!r}"
        )
    _validate_state(
        row["state"],
        label=f"{label}.state",
        require_sequential=require_sequential,
    )
    _validate_state(
        row["next_state"],
        label=f"{label}.next_state",
        require_sequential=require_sequential,
    )
    if row["state"].get("decision_point") != decision_point:
        raise DatasetValidationError(
            f"{label}.state.decision_point does not match the row"
        )

    action = row["action"]
    eligible_actions = row["eligible_actions"]
    if action not in ALL_ACTIONS:
        raise DatasetValidationError(f"{label}.action is unsupported: {action!r}")
    if (
        not isinstance(eligible_actions, list)
        or not eligible_actions
        or any(action_name not in ALL_ACTIONS for action_name in eligible_actions)
    ):
        raise DatasetValidationError(
            f"{label}.eligible_actions must be a nonempty list of serving actions"
        )
    if action not in eligible_actions:
        raise DatasetValidationError(
            f"{label}.action is absent from eligible_actions"
        )

    propensity = row["propensity"]
    if (
        isinstance(propensity, bool)
        or not isinstance(propensity, (int, float))
        or not math.isfinite(float(propensity))
        or not 0.0 < float(propensity) <= 1.0
    ):
        raise DatasetValidationError(
            f"{label}.propensity must be finite and in (0, 1]"
        )
    for field in ("reward", "reward_without_cost", "action_cost"):
        value = row[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise DatasetValidationError(f"{label}.{field} must be finite")
    if type(row["done"]) is not bool:
        raise DatasetValidationError(f"{label}.done must be a boolean")

    return session_id, step, row["done"]


def validate_training_dataset(
    dataset_path: Path,
    *,
    summary_path: Path | None = None,
    expected_sessions: int | None = None,
    require_sequential: bool = False,
) -> dict[str, Any]:
    """Validate all rows and the exact summary/hash pair."""
    dataset = _require_nonempty_file(dataset_path, "training dataset")
    summary = _require_nonempty_file(
        summary_path or dataset.parent / "dataset_summary.json",
        "training dataset summary",
    )
    if expected_sessions is not None:
        _require_int(expected_sessions, "expected_sessions", minimum=1)

    digest = hashlib.sha256()
    transition_count = 0
    last_step_by_session: dict[str, int] = {}
    completed_sessions: set[str] = set()
    try:
        with dataset.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise DatasetValidationError(
                        f"dataset line {line_number} is blank"
                    )
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DatasetValidationError(
                        f"dataset line {line_number} is invalid JSON: {exc}"
                    ) from exc

                session_id, step, done = _validate_row(
                    row,
                    line_number=line_number,
                    require_sequential=require_sequential,
                )
                if session_id in completed_sessions:
                    raise DatasetValidationError(
                        f"dataset line {line_number} follows a terminal row for "
                        f"session {session_id!r}"
                    )
                expected_step = last_step_by_session.get(session_id, -1) + 1
                if step != expected_step:
                    raise DatasetValidationError(
                        f"dataset line {line_number} has t={step} for session "
                        f"{session_id!r}; expected {expected_step}"
                    )
                last_step_by_session[session_id] = step
                if done:
                    completed_sessions.add(session_id)
                transition_count += 1
    except OSError as exc:
        raise DatasetValidationError(f"cannot read training dataset: {exc}") from exc

    if transition_count == 0:
        raise DatasetValidationError("training dataset contains no transitions")
    session_count = len(last_step_by_session)
    unfinished = set(last_step_by_session) - completed_sessions
    if unfinished:
        examples = sorted(unfinished)[:3]
        raise DatasetValidationError(
            f"training dataset has {len(unfinished)} unfinished sessions; "
            f"examples={examples}"
        )
    if expected_sessions is not None and session_count != expected_sessions:
        raise DatasetValidationError(
            f"training dataset contains {session_count} sessions; "
            f"expected {expected_sessions}"
        )

    try:
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"training dataset summary is invalid JSON: {exc}"
        ) from exc
    if not isinstance(summary_payload, dict):
        raise DatasetValidationError("training dataset summary must be an object")
    if summary_payload.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
        raise DatasetValidationError(
            "training dataset summary has no supported dataset_contract_version"
        )
    summary_transitions = _require_int(
        summary_payload.get("n_transitions"),
        "summary n_transitions",
        minimum=1,
    )
    summary_sessions = _require_int(
        summary_payload.get("n_sessions"),
        "summary n_sessions",
        minimum=1,
    )
    if summary_transitions != transition_count or summary_sessions != session_count:
        raise DatasetValidationError(
            "training dataset summary counts do not match JSONL: "
            f"summary=({summary_sessions} sessions, {summary_transitions} "
            f"transitions), jsonl=({session_count} sessions, "
            f"{transition_count} transitions)"
        )
    dataset_sha256 = digest.hexdigest()
    if summary_payload.get("jsonl_sha256") != dataset_sha256:
        raise DatasetValidationError(
            "training dataset summary jsonl_sha256 does not match JSONL bytes"
        )

    archetype_distribution = summary_payload.get("archetype_distribution")
    if not isinstance(archetype_distribution, dict) or any(
        type(count) is not int or count < 0
        for count in archetype_distribution.values()
    ):
        raise DatasetValidationError(
            "summary archetype_distribution must map names to nonnegative counts"
        )
    if sum(archetype_distribution.values()) != session_count:
        raise DatasetValidationError(
            "summary archetype_distribution does not sum to n_sessions"
        )

    return {
        "valid": True,
        "dataset": str(dataset),
        "summary": str(summary),
        "sha256": dataset_sha256,
        "n_sessions": session_count,
        "n_transitions": transition_count,
        "sequential_required": require_sequential,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, metavar="PATH")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Summary JSON (default: dataset_summary.json beside the dataset).",
    )
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=None,
        metavar="N",
    )
    parser.add_argument(
        "--require-sequential",
        action="store_true",
        help="Require all four trajectory-history state fields.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_training_dataset(
            args.dataset,
            summary_path=args.summary,
            expected_sessions=args.expected_sessions,
            require_sequential=args.require_sequential,
        )
    except Exception as exc:
        print(f"ERROR: training dataset validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
