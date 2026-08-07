import csv
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


DATASET_CONTRACT_VERSION = 1


@contextmanager
def _atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Write a sibling temporary file and publish it with one atomic replace."""
    temporary_path: Path | None = None
    handle: TextIO | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(handle.name)
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetWriter:
    """Writes simulated transitions to JSONL, flat CSV and a summary JSON.

    Output format is drop-in compatible with OfflineTraining/extract_offline_dataset.py
    so train_offline_policy.py can consume simulated data without modification.
    """

    def write(
        self,
        all_sessions: list[list[dict]],
        output_dir: str,
        archetype_counts: dict[str, int],
    ) -> tuple[Path, Path, Path]:
        os.makedirs(output_dir, exist_ok=True)
        out = Path(output_dir)

        flat = [t for session in all_sessions for t in session]

        jsonl_path = self._write_jsonl(flat, out)
        csv_path = self._write_csv(flat, out)
        summary_path = self._write_summary(
            flat,
            all_sessions,
            archetype_counts,
            out,
            jsonl_sha256=_sha256(jsonl_path),
        )

        return jsonl_path, csv_path, summary_path

    # ------------------------------------------------------------------

    @staticmethod
    def _user_type(transition: dict) -> str | None:
        """Return the additive archetype label used by persisted datasets.

        Historical in-memory transitions kept the label under
        ``metadata.user_type`` while :meth:`_write_jsonl` deliberately removed
        the complete metadata object.  Accept top-level aliases as well so the
        writer remains compatible with both old and future transition
        producers.
        """
        value = transition.get("user_type") or transition.get("archetype")
        metadata = transition.get("metadata")
        if value is None and isinstance(metadata, dict):
            value = metadata.get("user_type") or metadata.get("archetype")
        return str(value) if value is not None else None

    def _write_jsonl(self, flat: list[dict], out: Path) -> Path:
        path = out / "offline_transitions.jsonl"
        with _atomic_text_writer(path) as f:
            for t in flat:
                row = {k: v for k, v in t.items() if k != "metadata"}
                user_type = self._user_type(t)
                if user_type is not None:
                    # Additive field: existing offline-training consumers
                    # ignore unknown keys, while fidelity analyses can retain
                    # the session's generating archetype.
                    row["user_type"] = user_type
                f.write(json.dumps(row) + "\n")
        return path

    def _write_csv(self, flat: list[dict], out: Path) -> Path:
        path = out / "offline_transitions_flat.csv"
        fieldnames = [
            "trajectory_id", "t", "decision_id", "timestamp", "session_id",
            "user_type", "decision_point", "state_json", "action", "propensity",
            "eligible_actions_json", "reward", "reward_without_cost",
            "action_cost", "next_state_json", "done",
        ]
        with _atomic_text_writer(path, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in flat:
                writer.writerow({
                    "trajectory_id": t["trajectory_id"],
                    "t": t["t"],
                    "decision_id": t["decision_id"],
                    "timestamp": t["timestamp"],
                    "session_id": t["session_id"],
                    "user_type": self._user_type(t) or "",
                    "decision_point": t["decision_point"],
                    "state_json": json.dumps(t["state"], sort_keys=True),
                    "action": t["action"],
                    "propensity": t["propensity"],
                    "eligible_actions_json": json.dumps(t["eligible_actions"]),
                    "reward": t["reward"],
                    "reward_without_cost": t["reward_without_cost"],
                    "action_cost": t["action_cost"],
                    "next_state_json": json.dumps(t["next_state"], sort_keys=True),
                    "done": t["done"],
                })
        return path

    def _write_summary(
        self,
        flat: list[dict],
        all_sessions: list[list[dict]],
        archetype_counts: dict[str, int],
        out: Path,
        *,
        jsonl_sha256: str,
    ) -> Path:
        rewards = [t["reward"] for t in flat]
        purchases = sum(
            1 for t in flat
            if isinstance(t.get("metadata"), dict)
            and "purchase" in t["metadata"].get("generated_events", [])
        )
        n_sessions = len(all_sessions)
        n_transitions = len(flat)

        # Keep the existing aggregate fields intact and add session-level
        # metrics for each generating archetype.  A purchase is counted once
        # per session even if a custom transition producer emits it more than
        # once.
        sessions_by_archetype: dict[str, list[list[dict]]] = {
            str(name): [] for name in archetype_counts
        }
        for session in all_sessions:
            user_type = next(
                (self._user_type(t) for t in session if self._user_type(t) is not None),
                "unknown",
            )
            sessions_by_archetype.setdefault(str(user_type), []).append(session)

        per_archetype: dict[str, dict] = {}
        for user_type, sessions in sessions_by_archetype.items():
            session_rewards = [sum(float(t["reward"]) for t in session) for session in sessions]
            transition_rewards = [float(t["reward"]) for session in sessions for t in session]
            converted = sum(
                1
                for session in sessions
                if any(
                    isinstance(t.get("metadata"), dict)
                    and "purchase" in (t["metadata"].get("generated_events") or [])
                    for t in session
                )
            )
            n_type_sessions = len(sessions)
            n_type_transitions = sum(len(session) for session in sessions)
            mean_session_reward = (
                sum(session_rewards) / n_type_sessions if n_type_sessions else 0.0
            )
            if n_type_sessions > 1:
                reward_variance = sum(
                    (value - mean_session_reward) ** 2 for value in session_rewards
                ) / (n_type_sessions - 1)
                reward_sd = math.sqrt(reward_variance)
            else:
                reward_sd = 0.0

            per_archetype[user_type] = {
                "n_sessions": n_type_sessions,
                "n_transitions": n_type_transitions,
                "conversion_rate": converted / n_type_sessions if n_type_sessions else 0.0,
                "mean_reward_per_session": mean_session_reward,
                "std_reward_per_session": reward_sd,
                "reward_mean_per_transition": (
                    sum(transition_rewards) / len(transition_rewards)
                    if transition_rewards else 0.0
                ),
                "avg_session_length": (
                    n_type_transitions / n_type_sessions if n_type_sessions else 0.0
                ),
            }

        summary = {
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "jsonl_sha256": jsonl_sha256,
            "n_transitions": n_transitions,
            "n_sessions": n_sessions,
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "reward_min": min(rewards) if rewards else 0.0,
            "reward_max": max(rewards) if rewards else 0.0,
            "conversion_rate": purchases / n_sessions if n_sessions else 0.0,
            "avg_session_length": n_transitions / n_sessions if n_sessions else 0.0,
            "archetype_distribution": archetype_counts,
            "per_archetype": per_archetype,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        path = out / "dataset_summary.json"
        with _atomic_text_writer(path) as f:
            json.dump(summary, f, indent=2)
        return path
