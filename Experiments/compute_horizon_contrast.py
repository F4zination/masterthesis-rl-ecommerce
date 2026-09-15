#!/usr/bin/env python3
"""Paired seed-level horizon contrasts for the Section 5.1.5 myopic ablation.

Section 5.1.5 compares the anchor-adjusted gap under the discounted objective
(``gamma = 0.95``) against the same grid re-run with the discount removed
(``gamma = 0``), and reads the two supported mechanisms in opposite directions:
delayed assist credit inverts, action-dependent transition coupling does not
move. The first reading is safe -- a significant positive gap becomes a
significant negative one. The second was originally argued from the observation
that each myopic estimate falls inside the interval of its discounted
counterpart, and **overlapping intervals are not a test of equality**: two
estimates can each be individually uncertain while their difference is estimated
precisely, and conversely. A claim that the horizon contributes nothing needs the
difference itself, with its own interval.

That contrast is directly available. Both runs use seeds 10--14 and record
per-seed anchor-adjusted deltas in ``deltas.csv``, so for each grid point the
within-seed paired difference

    D_arm(v) = Delta_arm{gamma=0}(v) - Delta_arm{gamma=0.95}(v)

is computed per seed and aggregated across seeds exactly as every other quantity
in the sweep is. Pairing removes the seed-level variance the two runs share --
they use identical seeds, identical datasets and identical implementation hashes
-- so the contrast is materially better determined than either marginal estimate.
For the coupling axis this converts an overlap argument into a bound on how much
of the effect the optimization horizon can account for.

**Both arms carry a horizon.** ``run_sequentiality_sweep.py`` passes a single
``--gamma`` to the PPO trainer (``:340``) and to fitted-Q iteration (``:379``),
so the myopic run re-ran the *tabular* learner at ``gamma = 0`` as well. The
same contrast is therefore available within the tabular class, and it matters
more there than it does for PPO. Section 5.1.4 reads FQI's anchor-adjusted
contrast staying near zero as the tabular learner "gaining nothing" from delayed
credit, but those are different estimands: a flat *contrast* says FQI's
advantage relative to the bandit did not move, not that FQI's own reward failed
to rise. It did rise, by about +0.61 per session. ``D_fqi(v)`` is what the
section needed all along -- a manipulation of the horizon alone, holding the
representation fixed -- and because the implementation's offline pathology
(unseen-but-eligible actions retaining their zero initialisation while observed
ones are penalised, ``train_offline_policy.py:56,83``) sits on *both* sides of
the difference, it cancels rather than confounding the result.

Both analyses are **exploratory**, inheriting that status from the myopic run
they read (``preregistration_v3.md`` Section I): it was specified after the
confirmatory results were known and tests no pre-registered hypothesis. They add
no new run and no new seeds.

Statistical conventions match ``run_sequentiality_sweep.py`` exactly: the
within-seed contrast is formed first, then aggregated as mean plus or minus the
half-width of a pointwise two-sided 95% Student-t interval across seeds. The
convention is reimplemented here rather than imported, because importing the
sweep runner pulls in the full training pipeline; ``--self-check`` guards against
drift by recomputing each run's published per-cell values from its own
``deltas.csv`` and ``gaps.csv`` and asserting they reproduce that run's
``summary.json``. It additionally asserts the precondition the contrast needs:
that the bandit means are bit-identical across the two runs, so that the bandit
cancels exactly rather than approximately.

Usage::

    python Experiments/compute_horizon_contrast.py

Outputs, written beside the myopic run:
    horizon_contrast.csv                one row per (arm, axis, value)
    horizon_contrast.json               the same, plus provenance and the checks
    horizon_contrast_table.tex          Table 5.1.5 fragment (PPO)
    tabular_horizon_contrast_table.tex  the two arms' contrasts side by side
    absolute_change_table.tex           per-arm absolute change, for Sec 5.1.4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Confirmatory axis order, as tabulated in Section 5.1.5.
AXIS_ORDER = ["delayed_reward_strength", "transition_coupling_strength", "fatigue_rate"]

AXIS_LABEL = {
    "delayed_reward_strength": r"Delayed assist credit $\sigma$",
    "transition_coupling_strength": r"Action-dependent transitions $\kappa$",
    "fatigue_rate": r"Intervention fatigue $\rho$",
}

# Both learners are re-run at gamma = 0 by the myopic sweep, because one
# --gamma feeds both trainers. "ppo" is the neural arm of Section 5.1.5,
# "fqi" the tabular arm of Section 5.1.4.
ARMS = ("ppo", "fqi")

# gaps.csv column names for the raw per-seed means of each system.
ABS_COLUMN = {"ppo": "ppo_v3", "fqi": "fqi_v3", "bandit": "bandit_v2"}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _ci95(xs: list[float]) -> float:
    """Half-width of a pointwise two-sided 95% Student-t CI for the mean.

    Verbatim from ``run_sequentiality_sweep._ci95`` so that the paired contrast
    and the published marginals share one convention. At five seeds the critical
    value is 2.776 (four degrees of freedom), not 1.96.
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


Deltas = dict[tuple[str, float], dict[int, float]]


def load_deltas(run_dir: Path, arm: str) -> Deltas:
    """Per-seed anchor-adjusted deltas for one arm, keyed by (axis, value)."""
    path = run_dir / "deltas.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run the sweep first")
    column = f"delta_{arm}_minus_bandit"
    out: Deltas = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no column {column!r}")
        for row in reader:
            out[(row["axis"], float(row["value"]))][int(row["seed"])] = float(row[column])
    return dict(out)


def load_absolute(run_dir: Path) -> dict[tuple[str, float], dict[int, dict[str, float]]]:
    """Per-seed raw mean reward for each system, from gaps.csv.

    Needed because the anchor-adjusted contrast and the arm's own reward are
    different estimands, and Section 5.1.4 conflates them.
    """
    path = run_dir / "gaps.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run the sweep first")
    out: dict[tuple[str, float], dict[int, dict[str, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["axis"], float(row["value"]))][int(row["seed"])] = {
                name: float(row[col]) for name, col in ABS_COLUMN.items()
            }
    return dict(out)


def self_check(run_dir: Path, per_arm: dict[str, Deltas]) -> list[str]:
    """Assert the local convention reproduces this run's published summary.

    Guards the paired contrasts against silent divergence from the numbers the
    thesis already reports, for every arm the contrast will use.
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    worst = 0.0
    checked = 0
    for axis, records in summary["axes"].items():
        for rec in records:
            key = (axis, float(rec["value"]))
            for arm in ARMS:
                deltas = per_arm[arm]
                if key not in deltas:
                    continue
                vals = [deltas[key][s] for s in sorted(deltas[key])]
                for label, ours, theirs in (
                    ("mean", _mean(vals), rec[f"delta_{arm}_minus_bandit_mean"]),
                    ("ci95", _ci95(vals), rec[f"delta_{arm}_minus_bandit_ci95"]),
                ):
                    drift = abs(ours - theirs)
                    worst = max(worst, drift)
                    if drift > 1e-9:
                        raise SystemExit(
                            f"convention drift in {run_dir.name} {arm} {axis}={rec['value']} "
                            f"{label}: recomputed {ours!r} vs published {theirs!r}"
                        )
                checked += 1
    return [
        f"{run_dir.name}: reproduced every published delta mean and CI from deltas.csv "
        f"for {'/'.join(ARMS)} across {checked} cells (max drift {worst:.2e})"
    ]


def check_bandit_cancels(
    discounted_abs: dict[tuple[str, float], dict[int, dict[str, float]]],
    myopic_abs: dict[tuple[str, float], dict[int, dict[str, float]]],
) -> list[str]:
    """Assert the bandit is bit-identical across the two runs.

    Every contrast here is a difference of two arm-minus-bandit gaps, so the
    bandit cancels algebraically. That cancellation is only *exact* if the
    bandit means agree bit for bit -- which they should, since the discount
    reaches neither the behaviour policy nor the bandit aggregation. Verifying
    it is what licenses calling the contrast a within-arm manipulation with no
    property of the bandit in it.
    """
    shared = sorted(set(discounted_abs) & set(myopic_abs))
    compared = identical = 0
    worst = 0.0
    for key in shared:
        for seed in sorted(set(discounted_abs[key]) & set(myopic_abs[key])):
            a = discounted_abs[key][seed]["bandit"]
            b = myopic_abs[key][seed]["bandit"]
            compared += 1
            identical += a == b
            worst = max(worst, abs(a - b))
    if identical != compared:
        raise SystemExit(
            f"bandit differs across runs in {compared - identical}/{compared} cells "
            f"(max |diff| {worst:.3e}); the contrast is not a clean within-arm manipulation"
        )
    return [f"bandit mean bit-identical across both runs in all {compared} shared cells"]


def paired_contrast(discounted: Deltas, myopic: Deltas, arm: str) -> list[dict[str, Any]]:
    """D(v) = Delta_myopic(v) - Delta_discounted(v), paired within seed."""
    rows: list[dict[str, Any]] = []
    for axis in AXIS_ORDER:
        values = sorted(
            v for (a, v) in discounted if a == axis and (axis, v) in myopic and v != 0.0
        )
        for v in values:
            key = (axis, v)
            shared = sorted(set(discounted[key]) & set(myopic[key]))
            if not shared:
                continue
            d95 = [discounted[key][s] for s in shared]
            d0 = [myopic[key][s] for s in shared]
            paired = [myopic[key][s] - discounted[key][s] for s in shared]
            m, hw = _mean(paired), _ci95(paired)
            rows.append({
                "arm": arm,
                "axis": axis,
                "value": v,
                "n_seeds": len(shared),
                "seeds": shared,
                "delta_discounted_mean": _mean(d95),
                "delta_discounted_ci95": _ci95(d95),
                "delta_myopic_mean": _mean(d0),
                "delta_myopic_ci95": _ci95(d0),
                "paired_mean": m,
                "paired_ci95": hw,
                "paired_lo": m - hw,
                "paired_hi": m + hw,
                "paired_excludes_zero": abs(m) > hw,
            })
    return rows


def absolute_change(
    absolutes: dict[tuple[str, float], dict[int, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Within-seed change in each system's own reward, relative to v = 0.

    This is the quantity Section 5.1.4 needs and does not report. "FQI gains
    nothing" is true of the anchor-adjusted contrast and false of this.
    """
    rows: list[dict[str, Any]] = []
    for axis in AXIS_ORDER:
        anchor = absolutes.get((axis, 0.0))
        if anchor is None:
            continue
        for v in sorted(v for (a, v) in absolutes if a == axis and v != 0.0):
            shared = sorted(set(absolutes[(axis, v)]) & set(anchor))
            if not shared:
                continue
            row: dict[str, Any] = {"axis": axis, "value": v, "n_seeds": len(shared)}
            for name in ("ppo", "fqi", "bandit"):
                changes = [
                    absolutes[(axis, v)][s][name] - anchor[s][name] for s in shared
                ]
                m, hw = _mean(changes), _ci95(changes)
                row[f"{name}_abs_change_mean"] = m
                row[f"{name}_abs_change_ci95"] = hw
                row[f"{name}_abs_change_excludes_zero"] = abs(m) > hw
            rows.append(row)
    return rows


def fmt(mean: float, hw: float, bold: bool = False) -> str:
    body = f"{mean:+.3f} \\pm {hw:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def render_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "% Generated by Experiments/compute_horizon_contrast.py -- do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Anchor-adjusted gap $\Delta_{\text{PPO}}(v)$ under the discounted and the myopic "
        r"objective, with the paired within-seed horizon contrast "
        r"$D(v) = \Delta_{\gamma=0}(v) - \Delta_{\gamma=0.95}(v)$. Bold entries have a pointwise "
        r"95\,\% interval excluding zero.}",
        r"\label{tab:sweep-myopic}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Mechanism} & \textbf{$v$} & \textbf{$\gamma = 0.95$} & \textbf{$\gamma = 0$} "
        r"& \textbf{$D(v)$} \\",
        r"\midrule",
    ]
    for i, axis in enumerate(AXIS_ORDER):
        block = [r for r in rows if r["axis"] == axis]
        if not block:
            continue
        if i:
            out.append(r"\midrule")
        for j, r in enumerate(block):
            label = AXIS_LABEL[axis] if j == 0 else ""
            out.append(
                f"{label} & {r['value']:g} & "
                f"{fmt(r['delta_discounted_mean'], r['delta_discounted_ci95'], abs(r['delta_discounted_mean']) > r['delta_discounted_ci95'])} & "
                f"{fmt(r['delta_myopic_mean'], r['delta_myopic_ci95'], abs(r['delta_myopic_mean']) > r['delta_myopic_ci95'])} & "
                f"{fmt(r['paired_mean'], r['paired_ci95'], r['paired_excludes_zero'])} \\\\"
            )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def render_two_arm_table(by_arm: dict[str, list[dict[str, Any]]]) -> str:
    """The same horizon manipulation under both representations, side by side."""
    ppo = {(r["axis"], r["value"]): r for r in by_arm["ppo"]}
    fqi = {(r["axis"], r["value"]): r for r in by_arm["fqi"]}
    out = [
        "% Generated by Experiments/compute_horizon_contrast.py -- do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{The horizon manipulation under both representations. "
        r"$D_{\text{arm}}(v) = \Delta_{\gamma=0}(v) - \Delta_{\gamma=0.95}(v)$, paired "
        r"within seed. The bandit cancels from both columns exactly, and the tabular "
        r"learner's offline pathology sits on both sides of $D_{\text{FQI}}$ and cancels "
        r"with it. Bold entries have a pointwise 95\,\% interval excluding zero.}",
        r"\label{tab:horizon-by-representation}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Mechanism} & \textbf{$v$} & \textbf{$D_{\text{FQI}}(v)$ (tabular)} "
        r"& \textbf{$D_{\text{PPO}}(v)$ (neural)} \\",
        r"\midrule",
    ]
    for i, axis in enumerate(AXIS_ORDER):
        keys = [k for k in ppo if k[0] == axis]
        if not keys:
            continue
        if i:
            out.append(r"\midrule")
        for j, key in enumerate(sorted(keys, key=lambda k: k[1])):
            label = AXIS_LABEL[axis] if j == 0 else ""
            f, p = fqi.get(key), ppo[key]
            f_cell = (
                fmt(f["paired_mean"], f["paired_ci95"], f["paired_excludes_zero"])
                if f else "---"
            )
            out.append(
                f"{label} & {key[1]:g} & {f_cell} & "
                f"{fmt(p['paired_mean'], p['paired_ci95'], p['paired_excludes_zero'])} \\\\"
            )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def render_absolute_table(rows: list[dict[str, Any]]) -> str:
    """Each system's own reward change -- the estimand Section 5.1.4 needs."""
    out = [
        "% Generated by Experiments/compute_horizon_contrast.py -- do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Change in each system's \emph{own} mean reward per session relative to "
        r"the all-mechanisms-disabled configuration, paired within seed, under the "
        r"discounted objective. Distinct from the anchor-adjusted contrast, which measures "
        r"how each arm's gap \emph{against the bandit} moved. Bold entries have a pointwise "
        r"95\,\% interval excluding zero.}",
        r"\label{tab:absolute-change}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Mechanism} & \textbf{$v$} & \textbf{FQI} & \textbf{Bandit} & \textbf{PPO} \\",
        r"\midrule",
    ]
    for i, axis in enumerate(AXIS_ORDER):
        block = [r for r in rows if r["axis"] == axis]
        if not block:
            continue
        if i:
            out.append(r"\midrule")
        for j, r in enumerate(block):
            label = AXIS_LABEL[axis] if j == 0 else ""
            cells = " & ".join(
                fmt(
                    r[f"{name}_abs_change_mean"],
                    r[f"{name}_abs_change_ci95"],
                    r[f"{name}_abs_change_excludes_zero"],
                )
                for name in ("fqi", "bandit", "ppo")
            )
            out.append(f"{label} & {r['value']:g} & {cells} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--discounted-dir", type=Path, default=HERE / "sweep_results_confirmatory_v3")
    ap.add_argument("--myopic-dir", type=Path, default=HERE / "sweep_results_myopic_v3")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Defaults to --myopic-dir.")
    ap.add_argument("--no-self-check", action="store_true",
                    help="Skip reproducing each run's published summary values.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.myopic_dir
    discounted = {arm: load_deltas(args.discounted_dir, arm) for arm in ARMS}
    myopic = {arm: load_deltas(args.myopic_dir, arm) for arm in ARMS}
    discounted_abs = load_absolute(args.discounted_dir)
    myopic_abs = load_absolute(args.myopic_dir)

    checks: list[str] = []
    if not args.no_self_check:
        checks += self_check(args.discounted_dir, discounted)
        checks += self_check(args.myopic_dir, myopic)
        checks += check_bandit_cancels(discounted_abs, myopic_abs)
        for line in checks:
            print(f"[check] {line}")

    by_arm = {arm: paired_contrast(discounted[arm], myopic[arm], arm) for arm in ARMS}
    rows = [r for arm in ARMS for r in by_arm[arm]]
    if not rows:
        raise SystemExit("no overlapping (axis, value, seed) cells between the two runs")
    abs_rows = absolute_change(discounted_abs)

    for arm in ARMS:
        print(f"\n{arm.upper()}: D_{arm}(v) = Delta[gamma=0] - Delta[gamma=.95]")
        header = (
            f"{'axis':30s} {'v':>5s} {'gamma=.95':>16s} {'gamma=0':>16s} "
            f"{'paired D(v)':>17s}  excl.0"
        )
        print(header)
        print("-" * len(header))
        for r in by_arm[arm]:
            print(
                f"{r['axis']:30s} {r['value']:5g} "
                f"{r['delta_discounted_mean']:+8.3f}+/-{r['delta_discounted_ci95']:.3f} "
                f"{r['delta_myopic_mean']:+8.3f}+/-{r['delta_myopic_ci95']:.3f} "
                f"{r['paired_mean']:+8.3f}+/-{r['paired_ci95']:.3f}  "
                f"{'YES' if r['paired_excludes_zero'] else 'no'}"
            )

    print("\nABSOLUTE change in each system's own reward vs v=0 (discounted run)")
    header = f"{'axis':30s} {'v':>5s} {'FQI':>16s} {'bandit':>16s} {'PPO':>16s}"
    print(header)
    print("-" * len(header))
    for r in abs_rows:
        print(
            f"{r['axis']:30s} {r['value']:5g} "
            + " ".join(
                f"{r[f'{n}_abs_change_mean']:+8.3f}+/-{r[f'{n}_abs_change_ci95']:.3f}"
                for n in ("fqi", "bandit", "ppo")
            )
        )

    fields = [
        "arm", "axis", "value", "n_seeds",
        "delta_discounted_mean", "delta_discounted_ci95",
        "delta_myopic_mean", "delta_myopic_ci95",
        "paired_mean", "paired_ci95", "paired_lo", "paired_hi",
        "paired_excludes_zero",
    ]
    csv_path = out_dir / "horizon_contrast.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    abs_fields = ["axis", "value", "n_seeds"] + [
        f"{n}_abs_change_{s}"
        for n in ("fqi", "bandit", "ppo")
        for s in ("mean", "ci95", "excludes_zero")
    ]
    abs_csv_path = out_dir / "absolute_change.csv"
    with abs_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=abs_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(abs_rows)

    def fingerprint(run_dir: Path) -> Any:
        try:
            return json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            ).get("run_fingerprint")
        except OSError:
            return None

    json_path = out_dir / "horizon_contrast.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "Experiments/compute_horizon_contrast.py",
        "status": "exploratory (inherits preregistration_v3.md section I)",
        "estimands": {
            "paired_*": "D_arm(v) = Delta_arm,myopic(v) - Delta_arm,discounted(v), "
                        "paired within seed; the bandit cancels exactly",
            "*_abs_change_*": "change in that system's own mean reward vs v=0, "
                              "paired within seed; NOT a gap against the bandit",
        },
        "ci_method": "pointwise two-sided 95% Student-t interval across seeds",
        "arms": list(ARMS),
        "discounted_run": {
            "dir": str(args.discounted_dir), "run_fingerprint": fingerprint(args.discounted_dir),
        },
        "myopic_run": {
            "dir": str(args.myopic_dir), "run_fingerprint": fingerprint(args.myopic_dir),
        },
        "self_check": checks or ["skipped"],
        "cells": rows,
        "absolute_change": abs_rows,
    }, indent=2), encoding="utf-8")

    written = [csv_path, abs_csv_path, json_path]
    for path, body in (
        (out_dir / "horizon_contrast_table.tex", render_table(by_arm["ppo"])),
        (out_dir / "tabular_horizon_contrast_table.tex", render_two_arm_table(by_arm)),
        (out_dir / "absolute_change_table.tex", render_absolute_table(abs_rows)),
    ):
        path.write_text(body, encoding="utf-8")
        written.append(path)

    print("\nwrote " + "\n      ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
