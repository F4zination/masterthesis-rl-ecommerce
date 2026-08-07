#!/usr/bin/env python3
"""Build an auditable, fail-closed Clickworker study-release manifest.

The manifest identifies the exact frozen policies, policy-matched simulator
reference, training inputs, catalogue, code/configuration, dependency files,
container images, and deployment settings intended for one study release.  It
does not make an incomplete release look deployable: unresolved preregistration
fields, a dirty worktree, missing inputs, mutable/missing image identifiers, or
legacy policy artifacts without embedded build provenance all keep
``ready_for_recruitment`` false.

The destination must be new or empty.  A release is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import runpy
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SHARED_SCHEMA_PATH = str(REPO_ROOT / "SharedSchema")
if SHARED_SCHEMA_PATH not in sys.path:
    sys.path.insert(0, SHARED_SCHEMA_PATH)

from shared_schema.constants import DEVICE_TYPES, TRAFFIC_SOURCES  # noqa: E402
from shared_schema.policy_contract import (  # noqa: E402
    PolicyContractError,
    required_ppo_state_tokens,
    validate_bandit_policy_connection,
)

DEFAULT_BASELINE_DIR = (
    REPO_ROOT
    / "Experiments"
    / "study_policy_baselines"
    / "domain_aligned_20260728_v2"
)
REQUIRED_IMAGE_LABELS = ("v2", "v3", "dispatcher", "nginx", "certbot")
IMMUTABLE_IMAGE_RE = re.compile(r"(?:^sha256:|@sha256:)[0-9a-fA-F]{64}$")
BASELINE_POLICIES = ("v2_bandit", "v3_ppo")
BASELINE_ARCHETYPES = (
    "Explorer",
    "FastBuyer",
    "DetailedComparator",
    "DiscountHunter",
    "WindowShopper",
)
BASELINE_DEVICES = DEVICE_TYPES
BASELINE_TRAFFIC = TRAFFIC_SOURCES


class ReleaseManifestError(RuntimeError):
    """A safe manifest build cannot be completed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "path": display_path(path, repo_root),
        "exists": path.is_file(),
    }
    if path.is_file():
        record.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
    return record


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path.strip())


def parse_image(value: str) -> tuple[str, str]:
    label, separator, identifier = value.partition("=")
    if not separator or not label.strip() or not identifier.strip():
        raise argparse.ArgumentTypeError("expected LABEL=IMMUTABLE_IMAGE_ID")
    return label.strip(), identifier.strip()


def immutable_image_identifier(identifier: str) -> bool:
    return bool(IMMUTABLE_IMAGE_RE.search(identifier))


def run_command(command: list[str], cwd: Path, timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if completed.returncode:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    return {"available": True, "value": completed.stdout.strip()}


def inspect_git(repo_root: Path) -> dict[str, Any]:
    prefix = ["git", "-c", f"safe.directory={repo_root.resolve().as_posix()}"]
    head = run_command(prefix + ["rev-parse", "HEAD"], repo_root)
    status = run_command(
        prefix + ["status", "--porcelain=v1", "--untracked-files=all"], repo_root
    )
    entries = status.get("value", "").splitlines() if status.get("available") else []
    return {
        "commit": head.get("value") if head.get("available") else None,
        "dirty": bool(entries) if status.get("available") else None,
        "dirty_entries": entries,
        "head_error": head.get("error"),
        "status_error": status.get("error"),
    }


def inspect_runtime(repo_root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("fastapi", "jinja2", "PyYAML", "SQLAlchemy", "torch", "uvicorn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_version = run_command(["git", "--version"], repo_root)
    docker_version = run_command(
        ["docker", "version", "--format", "{{.Client.Version}}"], repo_root, timeout=5
    )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_client": git_version.get("value"),
        "docker_client": docker_version.get("value"),
        "docker_client_error": docker_version.get("error"),
    }


def _jsonable_sql_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    return value


def inspect_catalog(db_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"database": file_record(db_path)}
    if not db_path.is_file():
        result.update(valid=False, error="database file is missing")
        return result
    try:
        with closing(
            sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        ) as conn:
            quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
            tables: dict[str, Any] = {}
            canonical: dict[str, Any] = {}
            for table in ("categories", "products"):
                columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                if not columns:
                    raise ValueError(f"required table {table!r} is missing")
                quoted_columns = ", ".join(f'"{column}"' for column in columns)
                rows = conn.execute(
                    f'SELECT {quoted_columns} FROM "{table}" ORDER BY "id"'
                ).fetchall()
                canonical[table] = [
                    [_jsonable_sql_value(value) for value in row] for row in rows
                ]
                tables[table] = {"columns": columns, "row_count": len(rows)}
            canonical_bytes = json.dumps(
                canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
            result.update(
                valid=quick_check == "ok",
                sqlite_quick_check=quick_check,
                tables=tables,
                canonical_catalog_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
            )
    except (sqlite3.Error, ValueError) as exc:
        result.update(valid=False, error=str(exc))
    return result


def inspect_pre_recruitment_data(
    v2_path: Path,
    v3_path: Path,
    dispatcher_path: Path,
) -> dict[str, Any]:
    """Count rows that would leak pilot/test activity into confirmatory data."""
    shop_tables = ("events", "orders", "decision_logs", "cart_items", "session_discounts")

    def counts(path: Path, tables: tuple[str, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "readable": False,
            "counts": {},
            "total_rows": None,
        }
        if not path.is_file():
            return result
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                existing = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                table_counts = {
                    table: (
                        int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                        if table in existing
                        else 0
                    )
                    for table in tables
                }
            finally:
                connection.close()
            result.update(
                readable=True,
                counts=table_counts,
                total_rows=sum(table_counts.values()),
            )
        except sqlite3.Error as exc:
            result["error"] = str(exc)
        return result

    v2 = counts(v2_path, shop_tables)
    v3 = counts(v3_path, shop_tables)
    dispatcher = counts(dispatcher_path, ("assignments",))
    clean = bool(
        v2.get("readable")
        and v3.get("readable")
        and dispatcher.get("readable")
        and v2.get("total_rows") == 0
        and v3.get("total_rows") == 0
        and dispatcher.get("total_rows") == 0
    )
    return {
        "v2": v2,
        "v3": v3,
        "dispatcher": dispatcher,
        "clean_for_confirmatory_recruitment": clean,
        "rule": (
            "Archive pilot/test volumes and release empty event, order, decision, cart, "
            "discount, and assignment tables before confirmatory recruitment."
        ),
    }


def inspect_v2_provenance(db_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"embedded": False, "metadata": {}}
    if not db_path.is_file():
        result["error"] = "database file is missing"
        return result
    try:
        with closing(
            sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        ) as conn:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            table = next(
                (
                    name
                    for name in ("policy_build_metadata", "artifact_metadata")
                    if name in tables
                ),
                None,
            )
            if table is None:
                result["error"] = "no embedded policy-build metadata table"
                return result
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
            metadata: dict[str, Any]
            if {"key", "value"}.issubset(columns):
                key_i, value_i = columns.index("key"), columns.index("value")
                metadata = {str(row[key_i]): row[value_i] for row in rows}
            elif len(rows) == 1:
                metadata = dict(zip(columns, rows[0]))
            else:
                metadata = {"rows": [dict(zip(columns, row)) for row in rows]}
            result.update(embedded=True, table=table, metadata=metadata)
    except sqlite3.Error as exc:
        result["error"] = str(exc)
    return result


def inspect_v2_serving_contract(db_path: Path) -> dict[str, Any]:
    """Validate that the frozen Bandit covers every normalized serving arm."""
    result: dict[str, Any] = {"valid": False}
    if not db_path.is_file():
        result["error"] = "database file is missing"
        return result
    try:
        with closing(
            sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        ) as conn:
            result.update(validate_bandit_policy_connection(conn))
        result["valid"] = True
    except (sqlite3.Error, PolicyContractError) as exc:
        result["error"] = str(exc)
    return result


def inspect_v3_artifact_pair(
    checkpoint_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    """Validate that PPO checkpoint and JSON export are one complete pair."""
    result: dict[str, Any] = {"valid": False}
    if not checkpoint_path.is_file() or not policy_path.is_file():
        result["error"] = "checkpoint or JSON policy is missing"
        return result
    try:
        from OfflineTraining.train_ppo_policy import validate_saved_artifact_pair

        result.update(validate_saved_artifact_pair(checkpoint_path, policy_path))
        result["valid"] = True
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    return result


def _metadata_has(metadata: dict[str, Any], *keys: str) -> bool:
    return any(key in metadata and metadata[key] not in (None, "", {}) for key in keys)


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] not in (None, "", {}):
            value = metadata[key]
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
    return None


def required_v3_serving_tokens(context_schema_version: int = 2) -> set[str]:
    """Finite token domains emitted by the production context normalizer."""
    return set(required_ppo_state_tokens(context_schema_version))


def inspect_v3_provenance(checkpoint_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"inspectable": False, "metadata_keys": []}
    if not checkpoint_path.is_file():
        result["error"] = "checkpoint file is missing"
        return result
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root is not a mapping")
        provenance = checkpoint.get("provenance")
        state_vocab = checkpoint.get("state_vocab") or {}
        action_vocab = checkpoint.get("action_vocab") or {}
        metadata = dict(provenance) if isinstance(provenance, dict) else {}
        for key, value in checkpoint.items():
            if key not in {"model_state", "model_state_dict", "state_vocab", "action_vocab"}:
                metadata.setdefault(key, value)
        result.update(
            inspectable=True,
            metadata_keys=sorted(str(key) for key in metadata),
            provenance_block_present=isinstance(provenance, dict),
            serving_vocabulary={
                "required_state_token_count": len(required_v3_serving_tokens()),
                "missing_state_tokens": sorted(
                    required_v3_serving_tokens() - set(state_vocab)
                ) if isinstance(state_vocab, dict) else sorted(required_v3_serving_tokens()),
                "missing_actions": sorted(
                    {
                        "no-op",
                        "frequently_bought_together",
                        "discount_banner",
                        "trust_badge",
                        "help_popup",
                        "trending_carousel",
                    } - set(action_vocab)
                ) if isinstance(action_vocab, dict) else [
                    "discount_banner",
                    "frequently_bought_together",
                    "help_popup",
                    "no-op",
                    "trending_carousel",
                    "trust_badge",
                ],
            },
        )
        required_groups = {
            "training_dataset_sha256": ("training_dataset_sha256", "dataset_sha256"),
            "archetype_config_sha256": ("archetype_config_sha256", "archetypes_sha256"),
            "training_script_sha256": ("training_script_sha256",),
            "training_seed": ("training_seed", "seed"),
            "training_git_commit": ("training_git_commit", "git_commit"),
            "training_git_dirty": ("training_git_dirty", "git_dirty"),
            "dynamics": ("dynamics",),
            "t_max": ("t_max",),
            "iterations": ("iterations",),
            "rollout_sessions": ("rollout_sessions",),
            "mode": ("mode",),
            "sequential": ("sequential",),
        }
        result["missing_required_fields"] = [
            label
            for label, aliases in required_groups.items()
            if not _metadata_has(metadata, *aliases)
        ]
        result["declared"] = {
            label: _metadata_value(metadata, *aliases)
            for label, aliases in required_groups.items()
            if _metadata_has(metadata, *aliases)
        }
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def parse_compose_environment(compose_path: Path) -> dict[str, dict[str, str]]:
    if not compose_path.is_file():
        return {}
    services: dict[str, dict[str, str]] = {}
    current_service: str | None = None
    in_environment = False
    for line in compose_path.read_text(encoding="utf-8").splitlines():
        service_match = re.match(r"^  ([a-zA-Z0-9_-]+):\s*$", line)
        if service_match:
            current_service = service_match.group(1)
            services.setdefault(current_service, {})
            in_environment = False
            continue
        if current_service and re.match(r"^    environment:\s*$", line):
            in_environment = True
            continue
        if in_environment:
            env_match = re.match(r"^      ([A-Z][A-Z0-9_]*):\s*(.*?)\s*$", line)
            if env_match:
                value = env_match.group(2)
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                services[current_service][env_match.group(1)] = value
            elif line.strip() and not line.lstrip().startswith("#") and len(line) - len(line.lstrip()) <= 4:
                in_environment = False
    return services


def inspect_preregistration(path: Path, git_commit: str | None) -> dict[str, Any]:
    record = file_record(path)
    if not path.is_file():
        return {**record, "unfilled_fields": [], "status": None}
    lines = path.read_text(encoding="utf-8").splitlines()
    unfilled = []
    for line_number, line in enumerate(lines, 1):
        if "TO_BE_FILLED" not in line:
            continue
        if "Fields marked" in line or line.strip().startswith("- [ ] Fill all"):
            continue
        unfilled.append({"line": line_number, "text": line.strip()})
    status_match = next((re.search(r"\*\*Status\*\*:\s*(.+)", line) for line in lines if "**Status**" in line), None)
    release_match = next(
        (
            re.search(r"\*\*Study release commit\*\*:\s*`([^`]+)`", line)
            for line in lines
            if "**Study release commit**" in line
        ),
        None,
    )
    declared_commit = release_match.group(1) if release_match else None
    return {
        **record,
        "status": status_match.group(1).strip() if status_match else None,
        "unfilled_fields": unfilled,
        "declared_release_commit": declared_commit,
        "declared_commit_matches_head": bool(
            git_commit
            and declared_commit
            and declared_commit != "TO_BE_FILLED"
            and git_commit.startswith(declared_commit)
        ),
    }


def inspect_baseline(path: Path, artifacts: dict[str, dict[str, Any]], git_commit: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"file": file_record(path), "valid_json": False}
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result.update(
        valid_json=True,
        study_id=payload.get("study_id"),
        git_head=payload.get("git_head"),
        n_cells=(payload.get("design") or {}).get("n_cells"),
        sessions_per_cell=(payload.get("design") or {}).get("sessions_per_cell"),
    )
    context_design = (payload.get("design") or {}).get("context_standardization") or {}
    context_cells = [
        cell for cell in (payload.get("context_strata") or [])
        if isinstance(cell, dict)
    ]
    expected_context_keys = {
        (policy, archetype, device, traffic)
        for policy in BASELINE_POLICIES
        for archetype in BASELINE_ARCHETYPES
        for device in BASELINE_DEVICES
        for traffic in BASELINE_TRAFFIC
    }
    observed_context_keys = [
        (
            str(cell.get("policy") or ""),
            str(cell.get("archetype") or ""),
            str(cell.get("device_type") or ""),
            str(cell.get("traffic_source") or ""),
        )
        for cell in context_cells
    ]
    observed_context_key_set = set(observed_context_keys)
    duplicate_context_cells = len(observed_context_keys) - len(observed_context_key_set)
    insufficient_context_cells = sum(
        int(cell.get("n_sessions") or 0) < 2 for cell in context_cells
    )
    expected_context_cells = len(expected_context_keys)
    context_complete = (
        bool(context_design.get("enabled"))
        and observed_context_key_set == expected_context_keys
        and duplicate_context_cells == 0
        and insufficient_context_cells == 0
        and int(context_design.get("sessions_per_stratum") or 0) >= 2
        and int(context_design.get("n_strata_cells") or 0) == expected_context_cells
        and bool(context_design.get("independent_seeds_across_strata_cells"))
    )
    result["context_standardization"] = {
        "enabled": bool(context_design.get("enabled")),
        "sessions_per_stratum": context_design.get("sessions_per_stratum"),
        "n_strata_cells": len(context_cells),
        "expected_strata_cells": expected_context_cells,
        "duplicate_strata_cells": duplicate_context_cells,
        "insufficient_session_strata_cells": insufficient_context_cells,
        "missing_strata": ["|".join(key) for key in sorted(expected_context_keys - observed_context_key_set)],
        "unexpected_strata": ["|".join(key) for key in sorted(observed_context_key_set - expected_context_keys)],
        "independent_seeds_across_strata_cells": bool(
            context_design.get("independent_seeds_across_strata_cells")
        ),
        "complete": context_complete,
        "rule": context_design.get("rule"),
    }
    inputs = payload.get("inputs") or {}
    comparisons = {
        "v2_db": ((inputs.get("v2_bandit_db") or {}).get("sha256"), artifacts["v2_db"].get("sha256")),
        "v3_checkpoint": ((inputs.get("v3_ppo_checkpoint") or {}).get("sha256"), artifacts["v3_checkpoint"].get("sha256")),
        "archetype_config": ((inputs.get("archetype_config") or {}).get("sha256"), artifacts["archetype_config"].get("sha256")),
    }
    result["input_hash_matches"] = {
        label: bool(expected and actual and expected == actual)
        for label, (expected, actual) in comparisons.items()
    }
    code_checks = []
    for expected in inputs.get("code") or []:
        candidate = REPO_ROOT / str(expected.get("path", ""))
        actual = file_record(candidate)
        code_checks.append(
            {
                "path": expected.get("path"),
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual.get("sha256"),
                "matches": bool(expected.get("sha256") and expected.get("sha256") == actual.get("sha256")),
            }
        )
    result["code_hash_checks"] = code_checks
    all_baseline_cells = list(payload.get("cells") or []) + context_cells
    v2_cells = [
        cell
        for cell in all_baseline_cells
        if isinstance(cell, dict) and cell.get("policy") == "v2_bandit"
    ]
    v2_decisions = sum(
        int((cell.get("diagnostics") or {}).get("decisions") or 0)
        for cell in v2_cells
    )
    result["v2_serving_diagnostics"] = {
        "decisions": v2_decisions,
        "unseen_context_decisions": sum(
            int((cell.get("diagnostics") or {}).get("unseen_context_decisions") or 0)
            for cell in v2_cells
        ),
        "partially_unseen_context_decisions": sum(
            int(
                (cell.get("diagnostics") or {}).get(
                    "partially_unseen_context_decisions"
                )
                or 0
            )
            for cell in v2_cells
        ),
        "selected_unseen_arm_decisions": sum(
            int(
                (cell.get("diagnostics") or {}).get(
                    "selected_unseen_arm_decisions"
                )
                or 0
            )
            for cell in v2_cells
        ),
        "missing_feature_decisions": sum(
            int((cell.get("diagnostics") or {}).get("missing_feature_decisions") or 0)
            for cell in v2_cells
        ),
    }
    result["v2_serving_diagnostics"]["unseen_context_rate"] = (
        result["v2_serving_diagnostics"]["unseen_context_decisions"] / v2_decisions
        if v2_decisions
        else None
    )
    v3_cells = [
        cell
        for cell in all_baseline_cells
        if isinstance(cell, dict) and cell.get("policy") == "v3_ppo"
    ]
    v3_decisions = sum(int((cell.get("diagnostics") or {}).get("decisions") or 0) for cell in v3_cells)
    v3_oov_decisions = sum(
        int((cell.get("diagnostics") or {}).get("decisions_with_oov_tokens") or 0)
        for cell in v3_cells
    )
    v3_fallback_decisions = sum(
        int((cell.get("diagnostics") or {}).get("fallback_decisions") or 0)
        for cell in v3_cells
    )
    result["v3_serving_diagnostics"] = {
        "decisions": v3_decisions,
        "decisions_with_oov_tokens": v3_oov_decisions,
        "oov_decision_rate": v3_oov_decisions / v3_decisions if v3_decisions else None,
        "fallback_decisions": v3_fallback_decisions,
        "all_cells_serving_equivalent": bool(v3_cells)
        and all(bool((cell.get("diagnostics") or {}).get("serving_equivalent")) for cell in v3_cells),
    }
    result["git_head_matches_release"] = bool(
        git_commit and payload.get("git_head") == git_commit
    )
    return result


def dependency_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    manifests = [
        repo_root / "CustomerSimulation" / "requirements.txt",
        repo_root / "DemoSiteV2" / "requirements.txt",
        repo_root / "DemoSiteV3" / "requirements.txt",
        repo_root / "StudyDispatcher" / "requirements.txt",
        repo_root / "SharedSchema" / "pyproject.toml",
    ]
    locks = []
    for pattern in ("poetry.lock", "Pipfile.lock", "uv.lock", "pdm.lock", "requirements.lock", "requirements*.lock"):
        locks.extend(repo_root.glob(pattern))
        locks.extend(repo_root.glob(f"*/{pattern}"))
    return manifests, sorted(set(locks))


def inspect_reward_schema(constants_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    record = file_record(constants_path, repo_root)
    if not constants_path.is_file():
        return {"source": record, "valid": False, "error": "constants file is missing"}
    try:
        namespace = runpy.run_path(str(constants_path))
        values = {
            "context_schema_version": namespace["CONTEXT_SCHEMA_VERSION"],
            "action_cost": namespace["ACTION_COST"],
            "event_reward": namespace["EVENT_REWARD"],
            "history_priming_actions": namespace["HISTORY_PRIMING_ACTIONS"],
            "history_priming_amount": namespace["HISTORY_PRIMING_AMOUNT"],
            "history_primed_credit_decay": namespace["HISTORY_PRIMED_CREDIT_DECAY"],
        }
        canonical = json.dumps(
            values, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "source": record,
            "valid": True,
            "values": values,
            "canonical_reward_schema_sha256": hashlib.sha256(canonical).hexdigest(),
            "note": "Order-total purchase bonus logic is additionally pinned by the features.py source hash.",
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {"source": record, "valid": False, "error": str(exc)}


def _add_blocker(blockers: list[dict[str, str]], code: str, detail: str) -> None:
    if not any(item["code"] == code and item["detail"] == detail for item in blockers):
        blockers.append({"code": code, "detail": detail})


def build_manifest(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    git = inspect_git(repo_root)
    if not git.get("commit") or git.get("dirty") is None:
        _add_blocker(blockers, "git_state_unavailable", "Git commit or worktree status could not be read.")
    elif git["dirty"]:
        _add_blocker(blockers, "dirty_worktree", "Commit or stash every intended source change before recruitment.")

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    labelled_datasets = dict(args.training_dataset)
    if not labelled_datasets:
        shared_dataset = (
            repo_root
            / "OfflineTraining"
            / "data"
            / "deployment-sequential"
            / "offline_transitions.jsonl"
        )
        labelled_datasets = {
            "v2_sequential_transitions": shared_dataset,
            "v3_sequential_transitions": shared_dataset,
        }
    else:
        labelled_datasets = {label: resolve(path) for label, path in labelled_datasets.items()}

    artifact_paths = {
        "v2_db": resolve(args.v2_db),
        "v3_checkpoint": resolve(args.v3_checkpoint),
        "v3_policy": resolve(args.v3_policy),
        "v3_product_db": resolve(args.v3_product_db),
        "dispatcher_db": resolve(args.dispatcher_db),
        "baseline_json": resolve(args.baseline_json),
        "baseline_csv": resolve(args.baseline_csv),
        "power_analysis": resolve(args.power_analysis),
        "archetype_config": resolve(args.archetypes),
    }
    artifacts = {label: file_record(path, repo_root) for label, path in artifact_paths.items()}
    for label, record in artifacts.items():
        if not record["exists"]:
            _add_blocker(blockers, "required_input_missing", f"{label}: {record['path']}")

    training_inputs = {
        label: file_record(path, repo_root) for label, path in labelled_datasets.items()
    }
    for label, record in training_inputs.items():
        if not record["exists"]:
            _add_blocker(blockers, "required_input_missing", f"training dataset {label}: {record['path']}")

    relevant_paths = [
        repo_root / "docker-compose.yml",
        repo_root / "example.env",
        repo_root / "CustomerSimulation" / "simulation" / "simulator.py",
        repo_root / "CustomerSimulation" / "simulation" / "state.py",
        repo_root / "DemoSiteV2" / "Dockerfile",
        repo_root / "DemoSiteV2" / "app" / "database.py",
        repo_root / "DemoSiteV2" / "app" / "main.py",
        repo_root / "DemoSiteV2" / "app" / "models.py",
        repo_root / "DemoSiteV2" / "app" / "routers" / "api.py",
        repo_root / "DemoSiteV2" / "app" / "routers" / "decision.py",
        repo_root / "DemoSiteV2" / "app" / "routers" / "events.py",
        repo_root / "DemoSiteV2" / "app" / "routers" / "shop.py",
        repo_root / "DemoSiteV2" / "app" / "services" / "cart.py",
        repo_root / "DemoSiteV2" / "app" / "services" / "decision.py",
        repo_root / "DemoSiteV2" / "app" / "services" / "tracking.py",
        repo_root / "DemoSiteV2" / "app" / "seed.py",
        repo_root / "DemoSiteV2" / "app" / "static" / "js" / "cart.js",
        repo_root / "DemoSiteV2" / "app" / "static" / "js" / "decision.js",
        repo_root / "DemoSiteV2" / "app" / "static" / "js" / "tracking.js",
        repo_root / "DemoSiteV2" / "app" / "templates" / "attention.html",
        repo_root / "DemoSiteV2" / "app" / "templates" / "base.html",
        repo_root / "DemoSiteV2" / "app" / "templates" / "checkout.html",
        repo_root / "DemoSiteV2" / "app" / "templates" / "confirmation.html",
        repo_root / "DemoSiteV2" / "app" / "templates" / "end_session.html",
        repo_root / "DemoSiteV2" / "app" / "templates" / "home.html",
        repo_root / "DemoSiteV3" / "Dockerfile",
        repo_root / "DemoSiteV3" / "app" / "config.py",
        repo_root / "DemoSiteV3" / "app" / "database.py",
        repo_root / "DemoSiteV3" / "app" / "main.py",
        repo_root / "DemoSiteV3" / "app" / "models.py",
        repo_root / "DemoSiteV3" / "app" / "routers" / "api.py",
        repo_root / "DemoSiteV3" / "app" / "routers" / "decision.py",
        repo_root / "DemoSiteV3" / "app" / "routers" / "events.py",
        repo_root / "DemoSiteV3" / "app" / "routers" / "shop.py",
        repo_root / "DemoSiteV3" / "app" / "seed.py",
        repo_root / "DemoSiteV3" / "app" / "services" / "cart.py",
        repo_root / "DemoSiteV3" / "app" / "services" / "decision.py",
        repo_root / "DemoSiteV3" / "app" / "services" / "ppo.py",
        repo_root / "DemoSiteV3" / "app" / "services" / "tracking.py",
        repo_root / "DemoSiteV3" / "app" / "static" / "js" / "cart.js",
        repo_root / "DemoSiteV3" / "app" / "static" / "js" / "decision.js",
        repo_root / "DemoSiteV3" / "app" / "static" / "js" / "tracking.js",
        repo_root / "DemoSiteV3" / "app" / "templates" / "attention.html",
        repo_root / "DemoSiteV3" / "app" / "templates" / "base.html",
        repo_root / "DemoSiteV3" / "app" / "templates" / "checkout.html",
        repo_root / "DemoSiteV3" / "app" / "templates" / "confirmation.html",
        repo_root / "DemoSiteV3" / "app" / "templates" / "end_session.html",
        repo_root / "DemoSiteV3" / "app" / "templates" / "home.html",
        repo_root / "StudyDispatcher" / "Dockerfile",
        repo_root / "StudyDispatcher" / "app" / "admin.py",
        repo_root / "StudyDispatcher" / "app" / "config.py",
        repo_root / "StudyDispatcher" / "app" / "legal.py",
        repo_root / "StudyDispatcher" / "app" / "main.py",
        repo_root / "StudyDispatcher" / "app" / "store.py",
        repo_root / "deploy" / "nginx" / "templates" / "default.conf.template",
        repo_root / "deploy" / "init-letsencrypt.sh",
        repo_root / "SharedSchema" / "shared_schema" / "constants.py",
        repo_root / "SharedSchema" / "shared_schema" / "features.py",
        repo_root / "SharedSchema" / "shared_schema" / "policy_contract.py",
        repo_root / "OfflineTraining" / "build_bandit_policy.py",
        repo_root / "OfflineTraining" / "train_ppo_policy.py",
        repo_root / "OfflineTraining" / "validate_policy_artifacts.py",
        repo_root / "deploy" / "prepare-study-artifacts.sh",
        repo_root / "Experiments" / "evaluate_study_policy_baseline.py",
        repo_root / "Experiments" / "calculate_clickworker_power.py",
        repo_root / "Experiments" / "analyze_clickworker_study.py",
        repo_root / "Experiments" / "verify_completion_codes.py",
        Path(__file__),
    ]
    relevant_files = [file_record(path, repo_root) for path in relevant_paths]
    for record in relevant_files:
        if not record["exists"]:
            _add_blocker(blockers, "required_input_missing", f"code/config: {record['path']}")

    prereg_path = resolve(args.preregistration)
    prereg = inspect_preregistration(prereg_path, git.get("commit"))
    if not prereg.get("exists"):
        _add_blocker(blockers, "required_input_missing", f"preregistration: {prereg['path']}")
    if prereg.get("unfilled_fields"):
        _add_blocker(blockers, "preregistration_unfilled", "Resolve every substantive TO_BE_FILLED field before recruitment.")
    if "draft" in str(prereg.get("status", "")).lower():
        _add_blocker(blockers, "preregistration_not_locked", "Preregistration status is still draft.")
    if prereg.get("declared_release_commit") not in (None, "TO_BE_FILLED") and not prereg.get("declared_commit_matches_head"):
        _add_blocker(blockers, "preregistration_commit_mismatch", "Declared study release commit does not match Git HEAD.")

    manifests, locks = dependency_files(repo_root)
    dependency_manifests = [file_record(path, repo_root) for path in manifests]
    dependency_locks = [file_record(path, repo_root) for path in locks]
    if any(not record["exists"] for record in dependency_manifests):
        _add_blocker(blockers, "required_input_missing", "One or more dependency manifests are missing.")
    if not dependency_locks:
        _add_blocker(blockers, "dependency_lock_missing", "No reproducible dependency lock file was found.")

    v2_catalog = inspect_catalog(artifact_paths["v2_db"])
    v3_catalog = inspect_catalog(artifact_paths["v3_product_db"])
    catalog_match = bool(
        v2_catalog.get("canonical_catalog_sha256")
        and v2_catalog.get("canonical_catalog_sha256") == v3_catalog.get("canonical_catalog_sha256")
    )
    if not v2_catalog.get("valid") or not v3_catalog.get("valid"):
        _add_blocker(blockers, "catalog_unreadable", "Both product databases must pass SQLite validation and contain categories/products.")
    elif not catalog_match:
        _add_blocker(blockers, "catalog_mismatch", "V2 and V3 canonical product catalogues differ.")

    pre_recruitment_data = inspect_pre_recruitment_data(
        artifact_paths["v2_db"],
        artifact_paths["v3_product_db"],
        artifact_paths["dispatcher_db"],
    )
    if not pre_recruitment_data["clean_for_confirmatory_recruitment"]:
        _add_blocker(
            blockers,
            "pre_recruitment_data_not_empty",
            "Archive pilot/test data and deploy clean shop and dispatcher ledgers.",
        )

    v2_provenance = inspect_v2_provenance(artifact_paths["v2_db"])
    v2_serving_contract = inspect_v2_serving_contract(artifact_paths["v2_db"])
    if not v2_serving_contract.get("valid"):
        _add_blocker(
            blockers,
            "v2_serving_domain_incomplete",
            "Rebuild V2 with all finite serving contexts/actions and auditable support.",
        )
    v2_metadata = v2_provenance.get("metadata") or {}
    v2_required = {
        "training_dataset_sha256": ("training_dataset_sha256", "dataset_sha256"),
        "build_git_commit": ("build_git_commit", "git_commit"),
        "build_git_dirty": ("build_git_dirty", "git_dirty"),
        "build_script_sha256": ("build_script_sha256",),
    }
    v2_missing = [label for label, aliases in v2_required.items() if not _metadata_has(v2_metadata, *aliases)]
    v2_provenance["missing_required_fields"] = v2_missing
    if not v2_provenance.get("embedded") or v2_missing:
        _add_blocker(blockers, "v2_policy_provenance_incomplete", "Rebuild V2 with embedded dataset, code, and Git provenance.")
    else:
        dataset_hashes = {
            record.get("sha256") for record in training_inputs.values() if record.get("sha256")
        }
        v2_declared = {
            label: _metadata_value(v2_metadata, *aliases)
            for label, aliases in v2_required.items()
        }
        v2_build_script = file_record(repo_root / "OfflineTraining" / "build_bandit_policy.py", repo_root)
        v2_verification = {
            "training_dataset_hash_matches_selected_input": v2_declared["training_dataset_sha256"] in dataset_hashes,
            "build_git_commit_matches_release": v2_declared["build_git_commit"] == git.get("commit"),
            "policy_was_built_from_clean_worktree": v2_declared["build_git_dirty"] is False,
            "build_script_hash_matches_release": v2_declared["build_script_sha256"] == v2_build_script.get("sha256"),
        }
        v2_provenance["declared"] = v2_declared
        v2_provenance["verification"] = v2_verification
        if not all(v2_verification.values()):
            _add_blocker(blockers, "v2_policy_provenance_mismatch", "Embedded V2 provenance does not match selected release inputs.")

    v3_provenance = inspect_v3_provenance(artifact_paths["v3_checkpoint"])
    v3_artifact_pair = inspect_v3_artifact_pair(
        artifact_paths["v3_checkpoint"],
        artifact_paths["v3_policy"],
    )
    if not v3_artifact_pair.get("valid"):
        _add_blocker(
            blockers,
            "v3_artifact_pair_inconsistent",
            "Re-export the PPO checkpoint and JSON policy as one validated pair.",
        )
    serving_vocabulary = v3_provenance.get("serving_vocabulary") or {}
    if serving_vocabulary.get("missing_state_tokens") or serving_vocabulary.get("missing_actions"):
        _add_blocker(
            blockers,
            "v3_serving_vocabulary_incomplete",
            "The checkpoint must encode the full finite production state-token and action domains.",
        )
    if not v3_provenance.get("inspectable") or v3_provenance.get("missing_required_fields"):
        _add_blocker(blockers, "v3_policy_provenance_incomplete", "Retrain/export V3 with complete embedded training provenance.")
    else:
        declared = v3_provenance.get("declared") or {}
        dataset_hashes = {
            record.get("sha256") for record in training_inputs.values() if record.get("sha256")
        }
        v3_verification = {
            "training_dataset_hash_matches_selected_input": declared.get("training_dataset_sha256") in dataset_hashes,
            "archetype_config_hash_matches_release": declared.get("archetype_config_sha256") == artifacts["archetype_config"].get("sha256"),
            "training_script_hash_matches_release": declared.get("training_script_sha256") == file_record(repo_root / "OfflineTraining" / "train_ppo_policy.py", repo_root).get("sha256"),
            "training_git_commit_matches_release": declared.get("training_git_commit") == git.get("commit"),
            "policy_was_trained_from_clean_worktree": declared.get("training_git_dirty") is False,
        }
        v3_provenance["verification"] = v3_verification
        if not all(v3_verification.values()):
            _add_blocker(blockers, "v3_policy_provenance_mismatch", "Embedded V3 provenance does not match selected release inputs.")

    baseline = inspect_baseline(
        artifact_paths["baseline_json"], artifacts, git.get("commit")
    )
    if not baseline.get("valid_json"):
        _add_blocker(blockers, "baseline_invalid", "Policy-matched baseline JSON is missing or invalid.")
    else:
        if not all((baseline.get("input_hash_matches") or {}).values()):
            _add_blocker(blockers, "baseline_artifact_mismatch", "Baseline was not generated from the selected policy/config artifacts.")
        if any(not item.get("matches") for item in baseline.get("code_hash_checks") or []):
            _add_blocker(blockers, "baseline_code_mismatch", "Code has changed since the baseline was generated.")
        if not baseline.get("git_head_matches_release"):
            _add_blocker(blockers, "baseline_commit_mismatch", "Baseline Git revision differs from the release revision.")
        if not (baseline.get("context_standardization") or {}).get("complete"):
            _add_blocker(
                blockers,
                "baseline_context_strata_incomplete",
                "Generate all policy x persona x device x traffic simulator strata for fidelity standardization.",
            )
        v2_diagnostics = baseline.get("v2_serving_diagnostics") or {}
        if int(v2_diagnostics.get("missing_feature_decisions") or 0) > 0:
            _add_blocker(
                blockers,
                "v2_baseline_not_serving_equivalent",
                "The policy-matched reference encountered a V2 missing-feature fallback.",
            )
        if (
            int(v2_diagnostics.get("unseen_context_decisions") or 0) > 0
            or int(v2_diagnostics.get("selected_unseen_arm_decisions") or 0) > 0
        ):
            _add_blocker(
                blockers,
                "v2_state_coverage_incomplete",
                "The frozen bandit has no trained evidence for one or more visited production contexts/actions.",
            )
        v3_diagnostics = baseline.get("v3_serving_diagnostics") or {}
        if int(v3_diagnostics.get("fallback_decisions") or 0) > 0 or not v3_diagnostics.get("all_cells_serving_equivalent"):
            _add_blocker(
                blockers,
                "v3_baseline_not_serving_equivalent",
                "The policy-matched reference encountered a PPO action failure; retrain or repair before recruitment.",
            )
        if int(v3_diagnostics.get("decisions_with_oov_tokens") or 0) > 0:
            _add_blocker(
                blockers,
                "v3_state_coverage_incomplete",
                "The PPO checkpoint vocabulary does not cover every serving-time state token.",
            )

    power_reference: dict[str, Any] = {
        "file": artifacts["power_analysis"],
        "valid_json": False,
        "baseline_hash_matches": False,
    }
    try:
        power_payload = json.loads(
            artifact_paths["power_analysis"].read_text(encoding="utf-8")
        )
        declared_baseline_hash = str(
            (power_payload.get("baseline") or {}).get("sha256") or ""
        )
        power_reference.update(
            valid_json=True,
            declared_baseline_sha256=declared_baseline_hash,
            baseline_hash_matches=(
                declared_baseline_hash == artifacts["baseline_json"].get("sha256")
            ),
            method=power_payload.get("method"),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    if not power_reference["valid_json"]:
        _add_blocker(
            blockers,
            "power_analysis_invalid",
            "The pre-recruitment power analysis is missing or invalid.",
        )
    elif not power_reference["baseline_hash_matches"]:
        _add_blocker(
            blockers,
            "power_analysis_stale",
            "The power analysis was not generated from the selected simulator baseline.",
        )

    compose_path = repo_root / "docker-compose.yml"
    compose_environment = parse_compose_environment(compose_path)
    expected_environment = {
        "v2": {
            "FREEZE_POLICY": "true",
            "REQUIRE_BANDIT_POLICY": "true",
        },
        "v3": {
            "FREEZE_POLICY": "true",
            "LEARNER_ENABLED": "false",
            "POLICY_MODE": "ppo_only",
            "REQUIRE_PPO_CHECKPOINT": "true",
            "TIMING_ENABLED": "false",
        },
    }
    deployment_checks: dict[str, bool] = {}
    for service, expected in expected_environment.items():
        for key, value in expected.items():
            label = f"{service}.{key}"
            deployment_checks[label] = compose_environment.get(service, {}).get(key) == value
    if not all(deployment_checks.values()):
        _add_blocker(blockers, "deployment_setting_mismatch", "Compose does not enforce all frozen PPO-only/no-learner/no-timing settings.")

    image_pairs = list(args.image)
    images: dict[str, dict[str, Any]] = {}
    for label, identifier in image_pairs:
        if label in images:
            raise ReleaseManifestError(f"duplicate --image label: {label}")
        images[label] = {
            "identifier": identifier,
            "immutable": immutable_image_identifier(identifier),
        }
    missing_images = [label for label in REQUIRED_IMAGE_LABELS if label not in images]
    if missing_images:
        _add_blocker(blockers, "image_identifiers_missing", ", ".join(missing_images))
    mutable_images = [label for label, item in images.items() if not item["immutable"]]
    if mutable_images:
        _add_blocker(blockers, "image_identifiers_mutable", ", ".join(mutable_images))

    status = "READY_FOR_RECRUITMENT" if not blockers else "DRAFT_NOT_READY"
    constants_path = repo_root / "SharedSchema" / "shared_schema" / "constants.py"
    reward_schema = inspect_reward_schema(constants_path, repo_root)
    if not reward_schema.get("valid"):
        _add_blocker(blockers, "reward_schema_unavailable", "Shared reward/schema constants could not be archived.")
        status = "DRAFT_NOT_READY"
    return {
        "schema_version": 1,
        "release_id": args.release_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "readiness": {
            "ready_for_recruitment": not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "rule": "Ready only when every fail-closed check passes; draft manifests remain useful audit artifacts.",
        },
        "version_control": git,
        "artifacts": artifacts,
        "training_inputs": training_inputs,
        "relevant_code_and_configuration": relevant_files,
        "dependencies": {
            "manifests": dependency_manifests,
            "lock_files": dependency_locks,
        },
        "product_catalogue": {
            "v2": v2_catalog,
            "v3": v3_catalog,
            "canonical_content_identical": catalog_match,
            "seed_strategy": "deterministic static catalogues; no random catalogue seed",
            "v2_seed_source": file_record(repo_root / "DemoSiteV2" / "app" / "seed.py", repo_root),
            "v3_seed_source": file_record(repo_root / "DemoSiteV3" / "app" / "seed.py", repo_root),
        },
        "pre_recruitment_data": pre_recruitment_data,
        "reward_and_feature_schema": reward_schema,
        "policy_provenance": {"v2_bandit": v2_provenance, "v3_ppo": v3_provenance},
        "policy_serving_contracts": {
            "v2_bandit": v2_serving_contract,
            "v3_ppo_artifact_pair": v3_artifact_pair,
        },
        "policy_matched_simulator_reference": baseline,
        "power_analysis_reference": power_reference,
        "preregistration": prereg,
        "deployment": {
            "compose": file_record(compose_path, repo_root),
            "expected_environment": expected_environment,
            "observed_environment": {service: compose_environment.get(service, {}) for service in expected_environment},
            "checks": deployment_checks,
            "policy_updates_during_study": False,
            "v3_runtime_fallback_allowed": False,
        },
        "container_images": {
            "required_labels": list(REQUIRED_IMAGE_LABELS),
            "images": images,
        },
        "runtime": inspect_runtime(repo_root),
    }


def write_manifest_safely(manifest: dict[str, Any], output_root: Path, release_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id):
        raise ReleaseManifestError("release ID may contain only letters, digits, dot, underscore, and hyphen")
    release_dir = output_root.resolve() / release_id
    if release_dir.exists() and (not release_dir.is_dir() or any(release_dir.iterdir())):
        raise ReleaseManifestError(f"refusing to overwrite non-empty release path: {release_dir}")
    release_dir.mkdir(parents=True, exist_ok=True)
    output = release_dir / "release_manifest.json"
    # Exclusive creation guards against a race between the safety check and write.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True, help="Unique directory-safe study release identifier")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "Experiments" / "study_releases"))
    parser.add_argument("--v2-db", default=str(REPO_ROOT / "DemoSiteV2" / "data" / "demosite.db"))
    parser.add_argument("--v3-checkpoint", default=str(REPO_ROOT / "OfflineTraining" / "outputs" / "ppo_policy.pt"))
    parser.add_argument("--v3-policy", default=str(REPO_ROOT / "OfflineTraining" / "outputs" / "trained_policy.json"))
    parser.add_argument("--v3-product-db", default=str(REPO_ROOT / "DemoSiteV3" / "data" / "demosite.db"))
    parser.add_argument("--dispatcher-db", default=str(REPO_ROOT / "StudyDispatcher" / "data" / "dispatcher.db"))
    parser.add_argument("--baseline-json", default=str(DEFAULT_BASELINE_DIR / "policy_archetype_baseline.json"))
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE_DIR / "policy_archetype_baseline.csv"))
    parser.add_argument("--power-analysis", default=str(DEFAULT_BASELINE_DIR / "power_analysis.json"))
    parser.add_argument("--archetypes", default=str(REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml"))
    parser.add_argument("--preregistration", default=str(REPO_ROOT / "Experiments" / "ClickworkerPreregistration.md"))
    parser.add_argument(
        "--training-dataset",
        action="append",
        default=[],
        type=parse_labelled_path,
        metavar="LABEL=PATH",
        help="Repeat to replace the default V2 and V3 training-dataset records",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=parse_image,
        metavar="LABEL=IMMUTABLE_ID",
        help="Repeat for v2, v3, dispatcher, nginx, and certbot image IDs/digests",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_manifest(args)
        output = write_manifest_safely(manifest, Path(args.output_root), args.release_id)
    except ReleaseManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "manifest": display_path(output),
        "status": manifest["status"],
        "ready_for_recruitment": manifest["readiness"]["ready_for_recruitment"],
        "blocker_count": manifest["readiness"]["blocker_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
