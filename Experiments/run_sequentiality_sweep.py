#!/usr/bin/env python3
"""Sequentiality sweep — the V3-vs-V2 experiment driver.

For each grid point (one dynamics knob set to a value, the others off) and each
seed, this script runs the full pipeline *in-process* and reports how the
V3-minus-V2 gap grows with the amount of sequential structure in the shared
environment:

    1. generate a dataset by rolling the uniform behaviour policy through the
       shared ``SessionSimulator`` under the grid point's ``dynamics`` (with
       ``sequential=True`` so trajectory-history features are emitted);
    2. build the V2 contextual bandit by aggregating arm stats (same context_key
       as DemoSiteV2 — POINT_FEATURES only, so it ignores history);
    3. train the tabular-sequential FQI baseline and the corrected on-policy PPO
       on the same dataset / same environment;
    4. evaluate every policy through the *same* simulator and *same* paired seed.

It reuses the exact functions the standalone scripts use (``train_ppo.train_ppo``,
``train_offline_policy.fitted_q_iteration``, ``eval_policy_sim.run_policy``,
``eval_all_policies_sim.PPOPolicy``/``BanditPolicy``), so a sweep cell is the
real pipeline — just orchestrated in one process for speed and Windows
robustness.

Outputs (under ``--out-dir``, default ``Experiments/sweep_results``):
    run_manifest.json    immutable input/code fingerprint for safe resume
    run_status.json      authoritative progress/completeness indicator
    cells/*.json         one durable result per completed grid cell
    *.partial.csv        inspectable progress tables (not final evidence)
    long_results.csv   one row per (axis, value, seed, policy)
    gaps.csv           one row per (axis, value, seed): V3/V2 means + gaps
    summary.json       aggregated mean ± 95% CI per (axis, value)
    gap_vs_<axis>.svg  the deliverable curve (dependency-free SVG)

Primary metric: V3 (PPO) minus V2 (bandit) mean reward per session; the
tabular FQI minus V2 curve is overlaid (the three-way read).  Hypotheses
H0–H2 are stated in ``Experiments/preregistration.md``.

Examples::

    python run_sequentiality_sweep.py --smoke        # ~1 min, validates wiring
    python run_sequentiality_sweep.py --axes transition_coupling_strength
    python run_sequentiality_sweep.py --axes fatigue_rate \
        --expose-history-to-bandit                    # history ablation
    python run_sequentiality_sweep.py --ppo-mode offline \
        --axes fatigue_rate delayed_reward_strength \
               transition_coupling_strength          # offline-PPO ablation
    python run_sequentiality_sweep.py                # full default grid

Re-running an interrupted command with the same output directory resumes from
the durable cell cache. A changed script, archetype config, grid, or training
setting is rejected to prevent accidental mixing of experimental conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "OfflineTraining"))
sys.path.insert(0, str(_repo_root / "CustomerSimulation"))
sys.path.insert(0, str(_repo_root / "SharedSchema"))

from simulation.archetype import load_archetypes  # noqa: E402
from simulation.behavior_policy import BehaviorPolicy  # noqa: E402
from simulation.simulator import SessionSimulator  # noqa: E402
from shared_schema.features import POINT_FEATURES  # noqa: E402

from eval_policy_sim import run_policy, NoOpPolicy  # noqa: E402
from eval_all_policies_sim import PPOPolicy, BanditPolicy  # noqa: E402
import train_ppo_policy as ppo  # noqa: E402
import train_offline_policy as fqi  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# Sweep grid (mirrors the pre-registration table; CLI can override)
# ---------------------------------------------------------------------------
DEFAULT_AXES: dict[str, list[float]] = {
    "fatigue_rate": [0.0, 0.1, 0.2, 0.4, 0.6],
    "delayed_reward_strength": [0.0, 0.25, 0.5, 1.0, 2.0],
    "transition_coupling_strength": [0.0, 0.5, 1.0, 2.0, 3.0],
    "t_max": [20, 30, 40, 60],
}
# t_max is swept as a horizon axis rather than a dynamics knob.
HORIZON_AXIS = "t_max"
SWEEP_CACHE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Durable, configuration-safe cell cache
# ---------------------------------------------------------------------------

def _json_bytes(value: Any) -> bytes:
    """Return a canonical JSON representation for hashing and persistence."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact, so interruption cannot truncate it."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a CSV artifact. Empty tables are intentionally absent."""
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _cell_identity(axis: str, value: float, seed: int) -> dict[str, Any]:
    return {"axis": axis, "value": value, "seed": seed}


def _cell_cache_path(cache_dir: Path, axis: str, value: float, seed: int) -> Path:
    identity = _cell_identity(axis, value, seed)
    suffix = hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]
    return cache_dir / f"cell_{suffix}.json"


def _resume_spec(
    cfg: argparse.Namespace,
    axes_grid: dict[str, list[float]],
    archetypes_path: Path,
) -> dict[str, Any]:
    """Describe every input that can change a cell result.

    The output directory and smoke convenience flag are deliberately excluded;
    the expanded grid and effective training settings below are what matter.
    """
    excluded = {
        "out_dir", "smoke", "axis_values", "expose_history_to_bandit",
    }
    effective_cfg = {
        key: value for key, value in vars(cfg).items() if key not in excluded
    }
    implementation_paths = [
        Path(__file__).resolve(),
        _repo_root / "CustomerSimulation" / "simulation" / "archetype.py",
        _repo_root / "CustomerSimulation" / "simulation" / "behavior_policy.py",
        _repo_root / "CustomerSimulation" / "simulation" / "simulator.py",
        _repo_root / "CustomerSimulation" / "simulation" / "state.py",
        _repo_root / "SharedSchema" / "shared_schema" / "constants.py",
        _repo_root / "SharedSchema" / "shared_schema" / "features.py",
        _repo_root / "OfflineTraining" / "train_ppo_policy.py",
        _repo_root / "OfflineTraining" / "train_offline_policy.py",
        _repo_root / "OfflineTraining" / "eval_policy_sim.py",
        _repo_root / "OfflineTraining" / "eval_all_policies_sim.py",
    ]
    return {
        "cache_schema_version": SWEEP_CACHE_SCHEMA_VERSION,
        "implementation_sha256": {
            str(path.relative_to(_repo_root)).replace("\\", "/"): _sha256_file(path)
            for path in implementation_paths
        },
        "runtime": {
            "python": sys.version,
            "torch": str(getattr(torch, "__version__", "unavailable")),
        },
        "archetypes_path": str(archetypes_path.resolve()),
        "archetypes_sha256": _sha256_file(archetypes_path),
        "axes_grid": {axis: axes_grid[axis] for axis in cfg.axes},
        "effective_config": effective_cfg,
        "bandit_history_exposed": all(
            "interventions_shown_bucket" in features
            for features in POINT_FEATURES.values()
        ),
    }


def _initialise_resume_manifest(
    out_dir: Path,
    spec: dict[str, Any],
) -> tuple[str, Path]:
    """Create or validate the run manifest and return its stable fingerprint."""
    manifest_path = out_dir / "run_manifest.json"
    fingerprint = hashlib.sha256(_json_bytes(spec)).hexdigest()
    expected = {
        "cache_schema_version": SWEEP_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "spec": spec,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot safely resume: invalid {manifest_path}: {exc}")
        if existing != expected:
            raise SystemExit(
                "Cannot safely resume into an output directory created with a "
                "different script, archetype config, grid, or hyperparameters. "
                f"Choose a new --out-dir. Existing manifest: {manifest_path}"
            )
    else:
        _atomic_write_json(manifest_path, expected)
    return fingerprint, manifest_path


def _load_cached_cell(
    path: Path,
    identity: dict[str, Any],
    fingerprint: str,
) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot safely resume: invalid cell cache {path}: {exc}")
    if cached.get("identity") != identity or cached.get("run_fingerprint") != fingerprint:
        raise SystemExit(f"Cannot safely resume: cell cache does not match run: {path}")
    stats = cached.get("stats")
    if not isinstance(stats, dict):
        raise SystemExit(f"Cannot safely resume: cell cache has no stats object: {path}")
    return stats


def _save_cached_cell(
    path: Path,
    identity: dict[str, Any],
    fingerprint: str,
    stats: dict[str, dict[str, float]],
) -> None:
    _atomic_write_json(path, {
        "cache_schema_version": SWEEP_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "run_fingerprint": fingerprint,
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# Pipeline stages (in-process; same functions as the standalone scripts)
# ---------------------------------------------------------------------------

def generate_dataset(
    archetypes: dict[str, Any],
    mixture_prior: dict[str, float],
    dynamics: dict[str, Any],
    sequential: bool,
    t_max: int,
    n_sessions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Roll the uniform behaviour policy → flat list of transition rows."""
    rng = random.Random(seed)
    sim = SessionSimulator(dynamics=dynamics, sequential=sequential)
    policy = BehaviorPolicy()
    arch_names = list(mixture_prior.keys())
    arch_weights = [mixture_prior[a] for a in arch_names]
    rows: list[dict[str, Any]] = []
    for _ in range(n_sessions):
        archetype = archetypes[rng.choices(arch_names, weights=arch_weights, k=1)[0]]
        rows.extend(sim.simulate_session(archetype, policy, t_max=t_max, rng=rng))
    return rows


def build_bandit_db(rows: list[dict[str, Any]], db_path: Path) -> None:
    """Aggregate V2 arm stats and write a minimal bandit_arm_stats sqlite.

    Mirrors ``build_bandit_policy.py`` arm aggregation exactly (context_key over
    POINT_FEATURES, reward_sum over the cost-adjusted reward), but writes only
    the table ``BanditPolicy`` reads — no DemoSiteV2 app dependency.
    """
    arms: dict[tuple[str, str, str], list] = defaultdict(lambda: [0, 0.0])
    for r in rows:
        dp = r["decision_point"]
        feats = POINT_FEATURES.get(dp)
        state = r.get("state") or {}
        if feats is None or any(f not in state for f in feats):
            continue
        ck = "|".join([dp] + [str(state[f]) for f in feats])
        arm = arms[(dp, ck, r["action"])]
        arm[0] += 1
        arm[1] += float(r["reward"])

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE bandit_arm_stats ("
        "decision_point TEXT, context_key TEXT, action TEXT, "
        "impressions INTEGER, reward_sum REAL)"
    )
    conn.executemany(
        "INSERT INTO bandit_arm_stats VALUES (?,?,?,?,?)",
        [(dp, ck, a, imp, rs) for (dp, ck, a), (imp, rs) in arms.items()],
    )
    conn.commit()
    conn.close()


def train_ppo_policy_obj(
    rows: list[dict[str, Any]],
    archetypes: dict[str, Any],
    mixture_prior: dict[str, float],
    dynamics: dict[str, Any],
    sequential: bool,
    t_max: int,
    seed: int,
    cfg: argparse.Namespace,
    tmp_pt: Path,
) -> PPOPolicy:
    """Train corrected PPO (online GAE) and wrap it in the served PPOPolicy."""
    model, vocab, action_vocab, _ = ppo.train_ppo(
        rows=rows,
        mode=cfg.ppo_mode,
        hidden_sizes=ppo.parse_hidden_sizes(cfg.hidden_sizes),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        lr=cfg.lr,
        iterations=cfg.ppo_iterations,
        epochs=cfg.ppo_epochs,
        minibatch_size=cfg.ppo_minibatch,
        max_grad_norm=0.5,
        rollout_sessions=cfg.ppo_rollout_sessions,
        t_max=t_max,
        dynamics=dynamics,
        sequential=sequential,
        archetypes=archetypes,
        mixture_prior=mixture_prior,
        seed=seed,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_vocab": vocab,
            "action_vocab": action_vocab,
            "hidden_sizes": ppo.parse_hidden_sizes(cfg.hidden_sizes),
        },
        tmp_pt,
    )
    return PPOPolicy(tmp_pt)


def train_fqi_policy_obj(
    rows: list[dict[str, Any]],
    cfg: argparse.Namespace,
    tmp_json: Path,
):
    """Train tabular conservative FQI and wrap it in the served TrainedPolicy."""
    from eval_policy_sim import TrainedPolicy  # local import (no torch needed)

    q = fqi.fitted_q_iteration(
        rows=rows, gamma=cfg.gamma, iters=cfg.fqi_iters,
        conservative_penalty=cfg.fqi_penalty,
    )
    policy, defaults = fqi.greedy_policy(q, rows)
    q_table = {f"{k[0]}|||{k[1]}": v for k, v in q.items()}
    tmp_json.write_text(json.dumps({
        "algorithm": "tabular_fqi_conservative",
        "policy": policy,
        "default_action_by_decision_point": defaults,
        "q_table": q_table,
    }), encoding="utf-8")
    return TrainedPolicy(tmp_json, temperature=0.0)


def eval_mean_reward(
    policy, archetypes, mixture_prior, dynamics, sequential, t_max, n_sessions, eval_seed,
) -> dict[str, float]:
    rng = random.Random(eval_seed)
    stats = run_policy(
        policy, archetypes, mixture_prior,
        n_sessions=n_sessions, rng=rng, t_max=t_max,
        dynamics=dynamics, sequential=sequential,
    )
    return {
        "mean_reward": stats["mean_reward_per_session"],
        "std_reward": stats["std_reward_per_session"],
        "purchase_rate": stats["purchase_rate"],
    }


# ---------------------------------------------------------------------------
# One grid cell
# ---------------------------------------------------------------------------

def run_cell(
    axis: str,
    value: float,
    seed: int,
    base_dynamics: dict[str, Any],
    archetypes: dict[str, Any],
    mixture_prior: dict[str, float],
    cfg: argparse.Namespace,
    tmp_dir: Path,
) -> dict[str, dict[str, float]]:
    """Run all policies for one (axis, value, seed) and return per-policy stats."""
    dynamics = dict(base_dynamics)
    t_max = cfg.t_max
    if axis == HORIZON_AXIS:
        t_max = int(value)
    else:
        dynamics[axis] = value

    rows = generate_dataset(
        archetypes, mixture_prior, dynamics, sequential=True,
        t_max=t_max, n_sessions=cfg.dataset_sessions, seed=seed,
    )

    tag = f"{axis}_{value}_{seed}_{uuid.uuid4().hex[:6]}"
    db_path = tmp_dir / f"bandit_{tag}.db"
    pt_path = tmp_dir / f"ppo_{tag}.pt"
    json_path = tmp_dir / f"fqi_{tag}.json"

    build_bandit_db(rows, db_path)
    bandit = BanditPolicy(db_path)
    ppo_policy = train_ppo_policy_obj(
        rows, archetypes, mixture_prior, dynamics, True, t_max, seed, cfg, pt_path
    )
    fqi_policy = train_fqi_policy_obj(rows, cfg, json_path)

    policies = {
        "bandit_v2": bandit,
        "ppo_v3": ppo_policy,
        "fqi_v3": fqi_policy,
        "uniform": BehaviorPolicy(),
        "no_op": NoOpPolicy(),
    }
    eval_seed = cfg.eval_seed_base + seed  # paired across policies within a cell
    out: dict[str, dict[str, float]] = {}
    for name, pol in policies.items():
        out[name] = eval_mean_reward(
            pol, archetypes, mixture_prior, dynamics, True,
            t_max, cfg.eval_sessions, eval_seed,
        )

    # tidy up temp artifacts
    for p in (db_path, pt_path, json_path):
        try:
            p.unlink()
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------
# Aggregation + stats
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _ci95(xs: list[float]) -> float:
    """Half-width of a pointwise two-sided 95% Student-t CI for the mean.

    Confirmatory runs use five seeds, where 1.96 materially understates
    uncertainty (the correct t critical value is 2.776 at four degrees of
    freedom). The small lookup table avoids adding SciPy as a runtime
    dependency; for more than 31 observations, the normal limit is adequate.
    """
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t_critical_975 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }.get(n - 1, 1.96)
    return t_critical_975 * sd / math.sqrt(n)


# ---------------------------------------------------------------------------
# Dependency-free SVG line chart (no matplotlib)
# ---------------------------------------------------------------------------

def svg_gap_chart(
    axis: str,
    xs: list[float],
    series: dict[str, tuple[list[float], list[float]]],
    title: str,
) -> str:
    """Render mean ± CI curves for each series to a standalone SVG string.

    ``series[name] = (means, ci_halfwidths)`` aligned with ``xs``.
    A dashed y=0 reference marks the "no gap" (V3 ties V2) line.
    """
    W, H = 760, 460
    ml, mr, mt, mb = 70, 170, 50, 60
    pw, ph = W - ml - mr, H - mt - mb

    all_y: list[float] = []
    for means, cis in series.values():
        for m, c in zip(means, cis):
            all_y.extend([m - c, m + c])
    all_y.append(0.0)
    ymin, ymax = min(all_y), max(all_y)
    if ymax - ymin < 1e-9:
        ymax += 1.0
        ymin -= 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    xmin, xmax = min(xs), max(xs)
    if xmax - xmin < 1e-9:
        xmax = xmin + 1.0

    def px(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def py(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * ph

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="system-ui,Segoe UI,Arial" font-size="13">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    parts.append(
        f'<text x="{ml}" y="26" font-size="16" font-weight="600">{title}</text>'
    )

    # axes
    parts.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333" stroke-width="1"/>'
    )

    # y gridlines / ticks
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        yy = py(yv)
        parts.append(
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" fill="#444">{yv:.2f}</text>'
        )

    # y=0 reference (V3 ties V2)
    y0 = py(0.0)
    parts.append(
        f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml+pw}" y2="{y0:.1f}" '
        f'stroke="#888" stroke-width="1.2" stroke-dasharray="5,4"/>'
    )
    parts.append(
        f'<text x="{ml+pw}" y="{y0-5:.1f}" text-anchor="end" fill="#888">V3 = V2</text>'
    )

    # x ticks
    for x in xs:
        xx = px(x)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{mt+ph}" x2="{xx:.1f}" y2="{mt+ph+5}" stroke="#333"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{mt+ph+20}" text-anchor="middle" fill="#444">{x:g}</text>'
        )
    parts.append(
        f'<text x="{ml+pw/2:.1f}" y="{H-14}" text-anchor="middle" fill="#222">{axis}</text>'
    )
    parts.append(
        f'<text x="18" y="{mt+ph/2:.1f}" text-anchor="middle" fill="#222" '
        f'transform="rotate(-90 18 {mt+ph/2:.1f})">mean reward gap / session</text>'
    )

    # series
    for idx, (name, (means, cis)) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        pts = " ".join(f"{px(x):.1f},{py(m):.1f}" for x, m in zip(xs, means))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        for x, m, c in zip(xs, means, cis):
            xx, yhi, ylo = px(x), py(m + c), py(m - c)
            parts.append(
                f'<line x1="{xx:.1f}" y1="{yhi:.1f}" x2="{xx:.1f}" y2="{ylo:.1f}" '
                f'stroke="{color}" stroke-width="1.2"/>'
            )
            parts.append(f'<circle cx="{xx:.1f}" cy="{py(m):.1f}" r="3.2" fill="{color}"/>')
        ly = mt + 10 + idx * 22
        parts.append(
            f'<line x1="{ml+pw+14}" y1="{ly}" x2="{ml+pw+34}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2.2"/>'
        )
        parts.append(f'<text x="{ml+pw+40}" y="{ly+4}" fill="#222">{name}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archetypes", default=str(_repo_root / "CustomerSimulation" / "config" / "archetypes.yaml"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "sweep_results"))
    p.add_argument("--axes", nargs="*", default=list(DEFAULT_AXES.keys()),
                   help="Which axes to sweep (default: all).")
    p.add_argument(
        "--axis-values", action="append", default=[], metavar="AXIS=V1,V2,...",
        help=(
            "Override one axis grid (repeatable), e.g. "
            "--axis-values fatigue_rate=0,0.6. Useful for declared ablations."
        ),
    )
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--dataset-sessions", type=int, default=4000)
    p.add_argument("--eval-sessions", type=int, default=3000)
    p.add_argument("--eval-seed-base", type=int, default=10_000)
    p.add_argument("--t-max", type=int, default=20, help="Horizon for non-horizon axes.")
    # PPO
    p.add_argument("--ppo-mode", choices=["online", "offline"], default="online")
    p.add_argument("--ppo-iterations", type=int, default=25)
    p.add_argument("--ppo-rollout-sessions", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-minibatch", type=int, default=256)
    p.add_argument("--hidden-sizes", default="128,64")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    # FQI
    p.add_argument("--fqi-iters", type=int, default=25)
    p.add_argument("--fqi-penalty", type=float, default=0.15)
    p.add_argument(
        "--expose-history-to-bandit", action="store_true",
        help=(
            "Expose interventions_shown_bucket to the bandit explicitly. "
            "This is the preregistered fatigue robustness ablation; the actual "
            "effective setting is recorded in run_manifest.json."
        ),
    )
    # presets
    p.add_argument("--smoke", action="store_true",
                   help="Tiny everything + 1 seed + 2 values/axis: validates wiring fast.")
    return p


def apply_axis_value_overrides(
    axes_grid: dict[str, list[float]], raw_overrides: list[str],
) -> None:
    """Apply validated ``--axis-values`` entries in place."""
    for raw in raw_overrides:
        if "=" not in raw:
            raise SystemExit(
                f"Invalid --axis-values {raw!r}; expected AXIS=V1,V2,..."
            )
        axis, values_text = raw.split("=", 1)
        axis = axis.strip()
        if axis not in DEFAULT_AXES:
            raise SystemExit(
                f"Unknown axis {axis!r}; choose from {sorted(DEFAULT_AXES)}"
            )
        try:
            values = [float(value) for value in values_text.split(",") if value.strip()]
        except ValueError as exc:
            raise SystemExit(f"Invalid numeric grid in --axis-values {raw!r}") from exc
        if not values:
            raise SystemExit(f"Empty grid in --axis-values {raw!r}")
        if axis == HORIZON_AXIS:
            if any(value <= 0 or not value.is_integer() for value in values):
                raise SystemExit("t_max overrides must be positive integers")
            axes_grid[axis] = [int(value) for value in values]
        else:
            if any(value < 0 for value in values):
                raise SystemExit(f"{axis} overrides must be non-negative")
            axes_grid[axis] = values


def configure_bandit_history_features(explicitly_expose: bool) -> bool:
    """Apply and return the effective expose-history robustness setting.

    The legacy environment variable is still honoured because it mutates
    ``POINT_FEATURES`` when the shared module is imported. The CLI switch makes
    the setting visible and reproducible in the run manifest.
    """
    if explicitly_expose:
        for features in POINT_FEATURES.values():
            if "interventions_shown_bucket" not in features:
                features.append("interventions_shown_bucket")
    return all(
        "interventions_shown_bucket" in features
        for features in POINT_FEATURES.values()
    )


def apply_smoke(cfg: argparse.Namespace) -> dict[str, list[float]]:
    cfg.seeds = [0]
    cfg.dataset_sessions = 500
    cfg.eval_sessions = 500
    cfg.ppo_iterations = 3
    cfg.ppo_rollout_sessions = 48
    cfg.ppo_epochs = 2
    cfg.ppo_minibatch = 128
    cfg.fqi_iters = 8
    return {
        "transition_coupling_strength": [0.0, 3.0],
    }


def main() -> None:
    cfg = build_arg_parser().parse_args()
    if torch is None:
        raise SystemExit("PyTorch is required for the sweep (PPO).")

    axes_grid = dict(DEFAULT_AXES)
    if cfg.smoke:
        if cfg.axis_values:
            raise SystemExit("--smoke and --axis-values cannot be combined")
        axes_grid = apply_smoke(cfg)
        cfg.axes = list(axes_grid.keys())
    else:
        apply_axis_value_overrides(axes_grid, cfg.axis_values)

    unknown_axes = [axis for axis in cfg.axes if axis not in axes_grid]
    if unknown_axes:
        raise SystemExit(
            f"Unknown --axes values {unknown_axes}; choose from {sorted(axes_grid)}"
        )
    if not cfg.axes:
        raise SystemExit("--axes must select at least one axis")

    history_exposed = configure_bandit_history_features(
        cfg.expose_history_to_bandit
    )
    if history_exposed and not cfg.expose_history_to_bandit:
        print(
            "[sweep] bandit history exposure is active via "
            "SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT; recording it in the manifest"
        )

    archetypes, mixture_prior, full_config = load_archetypes(cfg.archetypes)
    base_dynamics = dict(full_config.get("dynamics", {}) or {})
    # Force every knob off in the base; each axis turns exactly one on.
    for k in ("fatigue_rate", "delayed_reward_strength", "transition_coupling_strength"):
        base_dynamics[k] = 0.0

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    cache_dir = out_dir / "cells"
    cache_dir.mkdir(exist_ok=True)

    archetypes_path = Path(cfg.archetypes)
    spec = _resume_spec(cfg, axes_grid, archetypes_path)
    run_fingerprint, manifest_path = _initialise_resume_manifest(out_dir, spec)
    status_path = out_dir / "run_status.json"

    long_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "config": vars(cfg),
        "run_fingerprint": run_fingerprint,
        "ci_method": "pointwise two-sided 95% Student-t interval across seeds",
        "axis_roles": {
            axis: (
                "descriptive horizon sensitivity"
                if axis == HORIZON_AXIS else "confirmatory mechanism axis"
            )
            for axis in cfg.axes
        },
        "axes": {},
    }

    total_cells = sum(len(axes_grid[a]) for a in cfg.axes) * len(cfg.seeds)
    cell_i = 0
    completed_cells = 0
    t0 = time.time()
    print(f"[sweep] axes={cfg.axes} seeds={cfg.seeds}  ({total_cells} cells)")
    print(f"[sweep] durable manifest: {manifest_path}")
    _atomic_write_json(status_path, {
        "state": "running",
        "run_fingerprint": run_fingerprint,
        "expected_cells": total_cells,
        "completed_cells": 0,
        "failed_cells": [],
        "final_outputs_complete": False,
    })

    for axis in cfg.axes:
        values = axes_grid[axis]
        per_value_gaps: dict[float, dict[str, list]] = {
            v: {"ppo_minus_bandit": [], "fqi_minus_bandit": [],
                "ppo": [], "fqi": [], "bandit": [], "seeds": []} for v in values
        }
        for value in values:
            for seed in cfg.seeds:
                cell_i += 1
                t_cell = time.time()
                identity = _cell_identity(axis, value, seed)
                cache_path = _cell_cache_path(cache_dir, axis, value, seed)
                stats = _load_cached_cell(cache_path, identity, run_fingerprint)
                resumed = stats is not None
                if stats is None:
                    try:
                        stats = run_cell(axis, value, seed, base_dynamics,
                                         archetypes, mixture_prior, cfg, tmp_dir)
                    except Exception as exc:  # record failure, then keep the sweep alive
                        failure = {**identity, "error": repr(exc)}
                        failures.append(failure)
                        print(
                            f"[sweep]   CELL FAILED {axis}={value} seed={seed}: {exc!r}"
                        )
                        _atomic_write_json(status_path, {
                            "state": "running_with_failures",
                            "run_fingerprint": run_fingerprint,
                            "expected_cells": total_cells,
                            "completed_cells": completed_cells,
                            "failed_cells": failures,
                            "final_outputs_complete": False,
                        })
                        continue
                    # The cell becomes resumable before any aggregate output is
                    # touched. A kill after this point loses no completed work.
                    _save_cached_cell(
                        cache_path, identity, run_fingerprint, stats,
                    )

                bandit_m = stats["bandit_v2"]["mean_reward"]
                ppo_m = stats["ppo_v3"]["mean_reward"]
                fqi_m = stats["fqi_v3"]["mean_reward"]
                ppo_gap = ppo_m - bandit_m
                fqi_gap = fqi_m - bandit_m
                per_value_gaps[value]["ppo_minus_bandit"].append(ppo_gap)
                per_value_gaps[value]["fqi_minus_bandit"].append(fqi_gap)
                per_value_gaps[value]["ppo"].append(ppo_m)
                per_value_gaps[value]["fqi"].append(fqi_m)
                per_value_gaps[value]["bandit"].append(bandit_m)
                per_value_gaps[value]["seeds"].append(seed)

                for pname, pstats in stats.items():
                    long_rows.append({
                        "axis": axis, "value": value, "seed": seed,
                        "policy": pname, **pstats,
                    })
                gap_rows.append({
                    "axis": axis, "value": value, "seed": seed,
                    "bandit_v2": bandit_m, "ppo_v3": ppo_m, "fqi_v3": fqi_m,
                    "ppo_minus_bandit": ppo_gap, "fqi_minus_bandit": fqi_gap,
                })
                completed_cells += 1
                _atomic_write_csv(out_dir / "long_results.partial.csv", long_rows)
                _atomic_write_csv(out_dir / "gaps.partial.csv", gap_rows)
                _atomic_write_json(status_path, {
                    "state": "running" if not failures else "running_with_failures",
                    "run_fingerprint": run_fingerprint,
                    "expected_cells": total_cells,
                    "completed_cells": completed_cells,
                    "failed_cells": failures,
                    "final_outputs_complete": False,
                })
                source = "resumed" if resumed else f"{time.time()-t_cell:.0f}s"
                print(f"[sweep] [{cell_i}/{total_cells}] {axis}={value} seed={seed}  "
                      f"V2={bandit_m:.3f} PPO={ppo_m:.3f} FQI={fqi_m:.3f}  "
                      f"gap(PPO-V2)={ppo_gap:+.3f}  ({source})")

        # aggregate this axis
        # Anchor-adjusted primary contrast (preregistration_v2.md §C):
        # Δ(v) = gap(knob=v) − gap(knob=0), computed within-seed (seeds are
        # paired across cells) and then aggregated as mean ± 95% CI.  The
        # anchor gap itself is reported as a first-class quantity — it
        # estimates the residual online-vs-offline / optimizer asymmetry that
        # survives with zero sequential structure.
        anchor_value = 0.0 if 0.0 in values else values[0]
        anchor_g = per_value_gaps[anchor_value]
        anchor_ppo_by_seed = dict(zip(anchor_g["seeds"], anchor_g["ppo_minus_bandit"]))
        anchor_fqi_by_seed = dict(zip(anchor_g["seeds"], anchor_g["fqi_minus_bandit"]))
        summary.setdefault("anchor", {})[axis] = {
            "anchor_value": anchor_value,
            "anchor_gap_ppo_mean": _mean(anchor_g["ppo_minus_bandit"]),
            "anchor_gap_ppo_ci95": _ci95(anchor_g["ppo_minus_bandit"]),
            "anchor_gap_fqi_mean": _mean(anchor_g["fqi_minus_bandit"]),
            "anchor_gap_fqi_ci95": _ci95(anchor_g["fqi_minus_bandit"]),
            "n_seeds": len(anchor_g["seeds"]),
        }

        agg = []
        for value in values:
            g = per_value_gaps[value]
            delta_ppo = [
                gap - anchor_ppo_by_seed[s]
                for s, gap in zip(g["seeds"], g["ppo_minus_bandit"])
                if s in anchor_ppo_by_seed
            ]
            delta_fqi = [
                gap - anchor_fqi_by_seed[s]
                for s, gap in zip(g["seeds"], g["fqi_minus_bandit"])
                if s in anchor_fqi_by_seed
            ]
            for s, dp, df in zip(
                [s for s in g["seeds"] if s in anchor_ppo_by_seed], delta_ppo, delta_fqi
            ):
                delta_rows.append({
                    "axis": axis, "value": value, "seed": s,
                    "anchor_value": anchor_value,
                    "delta_ppo_minus_bandit": dp,
                    "delta_fqi_minus_bandit": df,
                })
            agg.append({
                "value": value,
                "complete": len(g["seeds"]) == len(cfg.seeds),
                "n_seeds": len(g["ppo_minus_bandit"]),
                "ppo_minus_bandit_mean": _mean(g["ppo_minus_bandit"]),
                "ppo_minus_bandit_ci95": _ci95(g["ppo_minus_bandit"]),
                "fqi_minus_bandit_mean": _mean(g["fqi_minus_bandit"]),
                "fqi_minus_bandit_ci95": _ci95(g["fqi_minus_bandit"]),
                "bandit_mean": _mean(g["bandit"]),
                "ppo_mean": _mean(g["ppo"]),
                "fqi_mean": _mean(g["fqi"]),
                "n_seeds_delta": len(delta_ppo),
                "delta_ppo_minus_bandit_mean": _mean(delta_ppo),
                "delta_ppo_minus_bandit_ci95": _ci95(delta_ppo),
                "delta_fqi_minus_bandit_mean": _mean(delta_fqi),
                "delta_fqi_minus_bandit_ci95": _ci95(delta_fqi),
            })
        summary["axes"][axis] = agg

        # Do not overwrite a plot with a partial axis. The authoritative
        # run_status.json marks all final outputs invalid until every cell is
        # present; partial CSVs remain available for progress inspection.
        axis_complete = all(a["complete"] for a in agg)
        if not axis_complete:
            print(f"[sweep] axis {axis} incomplete; final SVGs withheld")
            continue

        # plot
        xs = [a["value"] for a in agg]
        series = {
            "PPO (V3) − bandit (V2)": (
                [a["ppo_minus_bandit_mean"] for a in agg],
                [a["ppo_minus_bandit_ci95"] for a in agg],
            ),
            "FQI (tabular V3) − bandit (V2)": (
                [a["fqi_minus_bandit_mean"] for a in agg],
                [a["fqi_minus_bandit_ci95"] for a in agg],
            ),
        }
        svg = svg_gap_chart(axis, xs, series, f"V3 − V2 gap vs {axis}")
        (out_dir / f"gap_vs_{axis}.svg").write_text(svg, encoding="utf-8")

        # anchor-adjusted Δ(v) chart (primary contrast, prereg v2 §C)
        delta_series = {
            "Δ PPO − bandit (anchor-adj.)": (
                [a["delta_ppo_minus_bandit_mean"] for a in agg],
                [a["delta_ppo_minus_bandit_ci95"] for a in agg],
            ),
            "Δ FQI − bandit (anchor-adj.)": (
                [a["delta_fqi_minus_bandit_mean"] for a in agg],
                [a["delta_fqi_minus_bandit_ci95"] for a in agg],
            ),
        }
        svg = svg_gap_chart(axis, xs, delta_series, f"Anchor-adjusted Δ(v) vs {axis}")
        (out_dir / f"delta_vs_{axis}.svg").write_text(svg, encoding="utf-8")

    run_complete = completed_cells == total_cells and not failures
    summary["run_status"] = {
        "state": "complete" if run_complete else "incomplete",
        "expected_cells": total_cells,
        "completed_cells": completed_cells,
        "failed_cells": failures,
    }
    _atomic_write_csv(out_dir / "long_results.partial.csv", long_rows)
    _atomic_write_csv(out_dir / "gaps.partial.csv", gap_rows)
    if delta_rows:
        _atomic_write_csv(out_dir / "deltas.partial.csv", delta_rows)

    if run_complete:
        _atomic_write_csv(out_dir / "long_results.csv", long_rows)
        _atomic_write_csv(out_dir / "gaps.csv", gap_rows)
        _atomic_write_csv(out_dir / "deltas.csv", delta_rows)
        _atomic_write_json(out_dir / "summary.json", summary)

    _atomic_write_json(status_path, {
        "state": "complete" if run_complete else "incomplete",
        "run_fingerprint": run_fingerprint,
        "expected_cells": total_cells,
        "completed_cells": completed_cells,
        "failed_cells": failures,
        "final_outputs_complete": run_complete,
    })

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    if not run_complete:
        print(
            f"\n[sweep] INCOMPLETE after {time.time()-t0:.0f}s: "
            f"{completed_cells}/{total_cells} cells durable. Re-run the exact "
            "same command to resume."
        )
        print(f"  status: {status_path}")
        raise SystemExit(1)

    print(f"\n[sweep] complete in {time.time()-t0:.0f}s. Wrote:")
    print(f"  {out_dir / 'long_results.csv'}")
    print(f"  {out_dir / 'gaps.csv'}")
    print(f"  {out_dir / 'summary.json'}")
    for axis in cfg.axes:
        print(f"  {out_dir / f'gap_vs_{axis}.svg'}")


if __name__ == "__main__":
    main()
