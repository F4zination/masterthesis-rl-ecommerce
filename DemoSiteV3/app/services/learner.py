from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..config import (
    DATABASE_URL,
    LEARNER_ALGO,
    LEARNER_ENABLED,
    LEARNER_INTERVAL_SECONDS,
    LEARNER_LOOKBACK_HOURS,
    LEARNER_MAX_TRAIN_MINUTES,
    LEARNER_MIN_DECISIONS,
    LEARNER_OUTPUT_DIR,
    LEARNER_TIMING_MODE,
    OFFLINE_TRAINING_DIR,
    OFFLINE_POLICY_PATH,
    ACTIVE_POLICY_POINTER_PATH,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CHECKPOINT_PATH,
    PPO_HIDDEN_SIZES,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_PRETRAIN_DATASET_PATH,
    PPO_VALUE_COEF,
)
from ..database import SessionLocal
from ..models import DecisionLog

logger = logging.getLogger("demosite.learner")

_STATUS_LOCK = threading.Lock()
_STATUS = {
    "enabled": LEARNER_ENABLED,
    "running": False,
    "last_status": "idle",
    "last_error": "",
    "last_started_at": None,
    "last_finished_at": None,
    "last_duration_seconds": None,
    "last_trained_rows": 0,
    "active_version": None,
    "candidate_version": None,
}


@dataclass
class BackgroundLearner:
    interval_seconds: int
    _thread: threading.Thread | None = None
    _stop_event: threading.Event | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="policy-learner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run_loop(self) -> None:
        # Run once at startup, then periodically.
        _run_cycle_safe()
        while self._stop_event and not self._stop_event.wait(self.interval_seconds):
            _run_cycle_safe()


def get_learner_status() -> dict:
    with _STATUS_LOCK:
        status = dict(_STATUS)

    if not status.get("active_version"):
        pointer_version = _read_active_version(Path(ACTIVE_POLICY_POINTER_PATH))
        if pointer_version:
            status["active_version"] = pointer_version
        else:
            policy_path = Path(OFFLINE_POLICY_PATH)
            if policy_path.exists() and policy_path.parent.name.startswith("v"):
                status["active_version"] = policy_path.parent.name

    return status


def _set_status(**kwargs) -> None:
    with _STATUS_LOCK:
        _STATUS.update(kwargs)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _db_path_from_url() -> Path:
    prefix = "sqlite:///"
    if not DATABASE_URL.startswith(prefix):
        raise ValueError("Only sqlite DATABASE_URL is supported by background learner")
    return Path(DATABASE_URL[len(prefix):])


def _read_active_version(pointer_path: Path) -> str | None:
    if not pointer_path.exists():
        return None
    try:
        value = pointer_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _next_version(pointer_path: Path) -> str:
    current = _read_active_version(pointer_path)
    if not current:
        return "v000001"
    if current.startswith("v") and current[1:].isdigit():
        return f"v{int(current[1:]) + 1:06d}"
    return f"v{int(time.time()):d}"


def _write_active_pointer(pointer_path: Path, version: str) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    tmp.write_text(version, encoding="utf-8")
    tmp.replace(pointer_path)


def _run_command(args: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _validate_artifact(policy_path: Path) -> tuple[bool, str, int]:
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid policy json: {exc}", 0

    policy = payload.get("policy")
    defaults = payload.get("default_action_by_decision_point")
    if not isinstance(policy, dict) or not isinstance(defaults, dict):
        return False, "missing required policy/default keys", 0

    allowed_actions = {
        "no-op",
        "trending_carousel",
        "discount_banner",
        "frequently_bought_together",
        "trust_badge",
        "help_popup",
    }
    for action in defaults.values():
        if action not in allowed_actions:
            return False, f"unknown default action {action}", 0

    return True, "ok", int(payload.get("n_rows") or 0)


def _run_cycle_safe() -> None:
    started = time.monotonic()
    _set_status(running=True, last_status="running", last_error="", last_started_at=_now_iso())
    try:
        run_background_learning_cycle()
    except Exception as exc:  # noqa: BLE001 - final safety boundary for scheduler thread
        logger.exception("Learner cycle failed")
        _set_status(last_status="failed", last_error=str(exc))
    finally:
        duration = round(time.monotonic() - started, 3)
        _set_status(running=False, last_finished_at=_now_iso(), last_duration_seconds=duration)


def _merge_jsonl_files(output_path: Path, sources: list[Path]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for source in sources:
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    out_f.write(line)
                    out_f.write("\n")
                    row_count += 1
    return row_count


def run_background_learning_cycle() -> None:
    out_root = Path(LEARNER_OUTPUT_DIR)
    pointer_path = Path(ACTIVE_POLICY_POINTER_PATH)
    db_path = _db_path_from_url()

    if not db_path.exists():
        _set_status(last_status="skipped", last_error=f"db file not found: {db_path}")
        logger.warning("Skipping learner cycle because DB file is missing: %s", db_path)
        return

    window_end = datetime.now()
    window_start = window_end - timedelta(hours=LEARNER_LOOKBACK_HOURS)

    with SessionLocal() as db:
        decisions_in_window = (
            db.query(DecisionLog)
            .filter(DecisionLog.timestamp >= window_start, DecisionLog.timestamp < window_end)
            .count()
        )

    if decisions_in_window < LEARNER_MIN_DECISIONS:
        msg = f"insufficient decisions in window: {decisions_in_window} < {LEARNER_MIN_DECISIONS}"
        logger.info("Skipping learner cycle: %s", msg)
        _set_status(last_status="skipped", last_error=msg, last_trained_rows=0)
        return

    version = _next_version(pointer_path)
    pending_dir = out_root / f"{version}_pending"
    final_dir = out_root / version
    pending_dir.mkdir(parents=True, exist_ok=True)

    extract_script = Path(OFFLINE_TRAINING_DIR) / "extract_offline_dataset.py"
    train_script_name = "train_ppo_policy.py" if LEARNER_ALGO == "ppo" else "train_offline_policy.py"
    train_script = Path(OFFLINE_TRAINING_DIR) / train_script_name

    timeout_seconds = max(60, LEARNER_MAX_TRAIN_MINUTES * 60)

    extract_cmd = [
        sys.executable,
        str(extract_script),
        "--db-path",
        str(db_path),
        "--out-dir",
        str(pending_dir),
        "--timing-mode",
        LEARNER_TIMING_MODE,
        "--min-timestamp",
        window_start.replace(microsecond=0).isoformat(),
        "--max-timestamp",
        window_end.replace(microsecond=0).isoformat(),
    ]
    extract_res = _run_command(extract_cmd, timeout_seconds)
    if extract_res.returncode != 0:
        raise RuntimeError(f"extract failed: {extract_res.stderr.strip() or extract_res.stdout.strip()}")

    dataset_path = pending_dir / "offline_transitions.jsonl"
    training_dataset_path = dataset_path
    training_sources = [dataset_path]

    if LEARNER_ALGO == "ppo":
        bootstrap_path = Path(PPO_PRETRAIN_DATASET_PATH)
        checkpoint_path = Path(PPO_CHECKPOINT_PATH)
        if bootstrap_path.exists() and not checkpoint_path.exists():
            training_dataset_path = pending_dir / "ppo_training_transitions.jsonl"
            training_sources = [bootstrap_path, dataset_path]
            merged_rows = _merge_jsonl_files(training_dataset_path, training_sources)
            logger.info(
                "PPO training dataset prepared: sources=%s rows=%s",
                ",".join(str(source) for source in training_sources),
                merged_rows,
            )
        else:
            if bootstrap_path.exists():
                logger.info(
                    "PPO checkpoint already exists, fine-tuning on live extracted transitions only: %s",
                    checkpoint_path,
                )
            else:
                logger.info(
                    "PPO bootstrap dataset not found, training on live extracted transitions only: %s",
                    bootstrap_path,
                )

    train_cmd = [
        sys.executable,
        str(train_script),
        "--dataset",
        str(training_dataset_path),
        "--gamma",
        str(PPO_GAMMA),
        "--clip-eps",
        str(PPO_CLIP_EPS),
        "--entropy-coef",
        str(PPO_ENTROPY_COEF),
        "--value-coef",
        str(PPO_VALUE_COEF),
        "--lr",
        str(PPO_LR),
        "--epochs",
        str(PPO_EPOCHS),
        "--minibatch-size",
        str(PPO_MINIBATCH_SIZE),
        "--max-grad-norm",
        str(PPO_MAX_GRAD_NORM),
        "--hidden-sizes",
        PPO_HIDDEN_SIZES,
        "--gae-lambda",
        str(PPO_GAE_LAMBDA),
        "--out-dir",
        str(pending_dir),
        "--timing-mode",
        LEARNER_TIMING_MODE,
    ]
    train_res = _run_command(train_cmd, timeout_seconds)
    if train_res.returncode != 0:
        raise RuntimeError(f"train failed: {train_res.stderr.strip() or train_res.stdout.strip()}")

    policy_file = pending_dir / "trained_policy.json"
    valid, reason, trained_rows = _validate_artifact(policy_file)
    if not valid:
        raise RuntimeError(f"validation failed: {reason}")

    metadata = {
        "version": version,
        "window_start": window_start.replace(microsecond=0).isoformat(),
        "window_end": window_end.replace(microsecond=0).isoformat(),
        "decisions_in_window": decisions_in_window,
        "trained_rows": trained_rows,
        "timing_mode": LEARNER_TIMING_MODE,
        "training_algorithm": LEARNER_ALGO,
        "training_sources": [str(source) for source in training_sources],
        "created_at": _now_iso(),
        "extract_stdout": extract_res.stdout,
        "train_stdout": train_res.stdout,
    }
    (pending_dir / "learner_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if final_dir.exists():
        shutil.rmtree(final_dir)
    pending_dir.replace(final_dir)
    _write_active_pointer(pointer_path, version)

    # Keep flat-file compatibility for existing loaders.
    top_policy = Path(OFFLINE_POLICY_PATH)
    top_policy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_dir / "trained_policy.json", top_policy)

    checkpoint_file = final_dir / "ppo_policy.pt"
    if checkpoint_file.exists():
        checkpoint_target = Path(PPO_CHECKPOINT_PATH)
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_file, checkpoint_target)
    else:
        logger.warning("PPO checkpoint missing from published artifact: %s", checkpoint_file)

    _set_status(
        last_status="succeeded",
        last_error="",
        active_version=version,
        candidate_version=version,
        last_trained_rows=trained_rows,
    )
    logger.info(
        "Learner cycle succeeded: version=%s rows=%s decisions_window=%s",
        version,
        trained_rows,
        decisions_in_window,
    )
