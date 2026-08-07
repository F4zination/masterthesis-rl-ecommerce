#!/usr/bin/env python3
"""Fail-closed validation for the frozen V2 and V3 policy artifacts.

The deployment helper uses this command before retaining an existing artifact
and after creating a new one.  A non-zero exit means the files must not be
served:

* the PPO checkpoint and JSON export must be one cryptographically linked
  export and cover the complete finite serving vocabulary;
* the contextual-bandit database must contain the complete serving arm grid,
  its per-arm support accounting, and the matching coverage manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
for import_root in (REPO_ROOT, REPO_ROOT / "SharedSchema"):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from OfflineTraining.train_ppo_policy import (  # noqa: E402
    ActorCritic,
    load_checkpoint_for_validation,
    validate_saved_artifact_pair,
)
from shared_schema.policy_contract import (  # noqa: E402
    validate_and_load_ppo_checkpoint_model,
    validate_bandit_policy_connection,
    validate_ppo_study_metadata,
    validate_ppo_vocabulary,
)


DEFAULT_PPO_CHECKPOINT = REPO_ROOT / "OfflineTraining" / "outputs" / "ppo_policy.pt"
DEFAULT_PPO_POLICY = REPO_ROOT / "OfflineTraining" / "outputs" / "trained_policy.json"
DEFAULT_BANDIT_DB = REPO_ROOT / "DemoSiteV2" / "data" / "demosite.db"


class ArtifactValidationError(ValueError):
    """A required policy artifact is absent or structurally unusable."""


def _require_nonempty_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        is_file = resolved.is_file()
        size = resolved.stat().st_size if is_file else 0
    except OSError as exc:
        raise ArtifactValidationError(
            f"cannot inspect {label} at {resolved}: {exc}"
        ) from exc
    if not is_file:
        raise ArtifactValidationError(f"{label} is missing: {resolved}")
    if size <= 0:
        raise ArtifactValidationError(f"{label} is empty: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ArtifactValidationError(f"{label} is missing or invalid")
    return value.lower()


def _require_git_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ArtifactValidationError(f"{label} is missing or invalid")
    return value.lower()


def validate_ppo_artifacts(
    checkpoint_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    """Validate the linked PPO export and its complete serving vocabulary."""
    checkpoint = _require_nonempty_file(checkpoint_path, "PPO checkpoint")
    policy = _require_nonempty_file(policy_path, "PPO JSON policy")

    pair_report = validate_saved_artifact_pair(checkpoint, policy)
    checkpoint_payload = load_checkpoint_for_validation(checkpoint)
    state_vocab = checkpoint_payload.get("state_vocab")
    action_vocab = checkpoint_payload.get("action_vocab")
    if not isinstance(state_vocab, dict) or not isinstance(action_vocab, dict):
        raise ArtifactValidationError(
            "PPO checkpoint does not contain mapping state/action vocabularies"
        )
    vocabulary_report = validate_ppo_vocabulary(state_vocab, action_vocab)
    study_report = validate_ppo_study_metadata(checkpoint_payload)
    _, model_report = validate_and_load_ppo_checkpoint_model(
        checkpoint_payload,
        ActorCritic,
    )

    return {
        "checkpoint": str(checkpoint),
        "policy": str(policy),
        "pair": pair_report,
        "vocabulary": vocabulary_report,
        "model": model_report,
        "study_treatment": study_report,
    }


def validate_bandit_artifact(database_path: Path) -> dict[str, Any]:
    """Validate the contextual-bandit database without permitting writes."""
    database = _require_nonempty_file(database_path, "Bandit policy database")
    uri = f"{database.as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        contract_report = validate_bandit_policy_connection(connection)
        metadata = {
            str(key): json.loads(str(value))
            for key, value in connection.execute(
                "SELECT key, value FROM policy_build_metadata"
            )
        }
    dataset_sha256 = _require_sha256(
        metadata.get("dataset_sha256"),
        "Bandit provenance dataset_sha256",
    )
    git_commit = _require_git_commit(
        metadata.get("build_git_commit"),
        "Bandit provenance build_git_commit",
    )
    git_dirty = metadata.get("build_git_dirty")
    if type(git_dirty) is not bool:
        raise ArtifactValidationError(
            "Bandit provenance build_git_dirty must be a boolean"
        )
    return {
        "database": str(database),
        "contract": contract_report,
        "provenance": {
            "dataset_sha256": dataset_sha256,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    }


def validate_policy_artifacts(
    *,
    only: str,
    ppo_checkpoint: Path,
    ppo_policy: Path,
    bandit_db: Path,
    training_dataset: Path | None = None,
    require_clean_provenance: bool = False,
) -> dict[str, Any]:
    """Run the selected validation group and return a machine-readable report."""
    report: dict[str, Any] = {"valid": True, "checks": {}}
    if only in {"all", "ppo"}:
        report["checks"]["ppo"] = validate_ppo_artifacts(
            ppo_checkpoint,
            ppo_policy,
        )
    if only in {"all", "bandit"}:
        report["checks"]["bandit"] = validate_bandit_artifact(bandit_db)

    dataset_hashes: dict[str, str] = {}
    if "ppo" in report["checks"]:
        dataset_hashes["ppo"] = report["checks"]["ppo"]["study_treatment"][
            "dataset_sha256"
        ]
    if "bandit" in report["checks"]:
        dataset_hashes["bandit"] = report["checks"]["bandit"]["provenance"][
            "dataset_sha256"
        ]
    if len(set(dataset_hashes.values())) > 1:
        raise ArtifactValidationError(
            "PPO and Bandit were trained from different datasets: "
            f"{dataset_hashes}"
        )
    provenance_by_treatment: dict[str, dict[str, Any]] = {}
    if "ppo" in report["checks"]:
        provenance_by_treatment["ppo"] = report["checks"]["ppo"][
            "study_treatment"
        ]
    if "bandit" in report["checks"]:
        provenance_by_treatment["bandit"] = report["checks"]["bandit"][
            "provenance"
        ]
    commits = {
        treatment: provenance["git_commit"]
        for treatment, provenance in provenance_by_treatment.items()
    }
    dirty_flags = {
        treatment: provenance["git_dirty"]
        for treatment, provenance in provenance_by_treatment.items()
    }
    if len(set(commits.values())) > 1 or len(set(dirty_flags.values())) > 1:
        raise ArtifactValidationError(
            "PPO and Bandit provenance comes from different repository states: "
            f"commits={commits}, dirty={dirty_flags}"
        )
    if require_clean_provenance and any(dirty_flags.values()):
        raise ArtifactValidationError(
            "study deployment requires artifacts built from a clean worktree: "
            f"{dirty_flags}"
        )
    if training_dataset is not None:
        dataset_path = _require_nonempty_file(
            training_dataset,
            "selected shared training dataset",
        )
        selected_hash = _sha256(dataset_path)
        mismatches = {
            treatment: embedded_hash
            for treatment, embedded_hash in dataset_hashes.items()
            if embedded_hash != selected_hash
        }
        if mismatches:
            raise ArtifactValidationError(
                "artifact dataset provenance does not match the selected "
                f"training dataset {selected_hash}: {mismatches}"
            )
        report["training_dataset"] = {
            "path": str(dataset_path),
            "sha256": selected_hash,
        }
    report["policy_set"] = {
        "shared_dataset_sha256": next(iter(dataset_hashes.values()), None),
        "git_commit": next(iter(commits.values()), None),
        "git_dirty": next(iter(dirty_flags.values()), None),
        "treatments": sorted(dataset_hashes),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "ppo", "bandit"),
        default="all",
        help="Validation group to run (default: all).",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=DEFAULT_PPO_CHECKPOINT,
        metavar="PATH",
    )
    parser.add_argument(
        "--ppo-policy",
        type=Path,
        default=DEFAULT_PPO_POLICY,
        metavar="PATH",
    )
    parser.add_argument(
        "--bandit-db",
        type=Path,
        default=DEFAULT_BANDIT_DB,
        metavar="PATH",
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optionally require both artifacts to match this exact dataset.",
    )
    parser.add_argument(
        "--require-clean-provenance",
        action="store_true",
        help="Reject artifacts whose embedded Git worktree state was dirty.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_policy_artifacts(
            only=args.only,
            ppo_checkpoint=args.ppo_checkpoint,
            ppo_policy=args.ppo_policy,
            bandit_db=args.bandit_db,
            training_dataset=args.training_dataset,
            require_clean_provenance=args.require_clean_provenance,
        )
    except Exception as exc:
        print(
            f"ERROR: policy artifact validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
