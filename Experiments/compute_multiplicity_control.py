#!/usr/bin/env python3
"""Multiplicity control for the sequentiality sweep's confirmatory family.

The preregistered decision rule declares an effect significant exactly when its
own pointwise 95% interval excludes zero. Applied across the confirmatory grid
that is twenty-four such decisions -- twelve mechanism cells for each of the two
learners in Table `tab:sweep-delta` -- with no family-level control, so the
probability that at least one null cell clears the rule is far above 0.05. The
thesis disclosed this and did not correct it. This script supplies the
correction, at two levels.

**Per cell, with a false-discovery-rate adjustment.** For each cell the per-seed
anchor-adjusted deltas give a one-sample t statistic on df = n-1, hence a
two-sided p-value; Benjamini-Hochberg across the twenty-four then controls the
expected proportion of false discoveries among the cells declared significant.
BH rather than Bonferroni because the cells are positively dependent (shared
seeds, shared anchor, monotone knob grids), which is the regime BH is designed
for and where Bonferroni is needlessly conservative.

**Per axis, which is how the thesis actually argues.** Section `subsec:EvalRQ1`
supports a mechanism by the shape of its axis rather than by any single cell.
The corresponding statistic is the within-seed mean of Delta(v) over the axis's
non-zero settings: one number per seed, one interval per axis, three axes per
learner. Holm across the six axis-level tests is then almost free, because six
tests is not a multiplicity problem in the way twenty-four is. This is the
weaker-looking but more honest primary analysis, and it is the one the thesis
should lead with.

**Status.** This is a *post hoc* addition, not a preregistered analysis. The
preregistration fixed the per-cell rule and this script does not retrospectively
replace it: both are reported, the preregistered rule as the declared primary
decision and these adjustments as the multiplicity-aware reading alongside it.
`preregistration_v3.md` records the amendment. Where the two disagree the
disagreement is itself the finding and is reported as such.

The t-distribution tail is implemented here rather than imported, for the same
reason `compute_horizon_contrast.py` reimplements the interval convention:
`Experiments/requirements.txt` does not declare scipy, and an analysis that
carries a thesis claim should not depend on a package that is present in the
development environment by accident. `--self-check` validates the implementation
against closed-form values.

Usage::

    python Experiments/compute_multiplicity_control.py

Outputs, written beside the run:
    multiplicity_control.csv        per-cell p, BH q, and both decisions
    multiplicity_control.json       the same, plus axis-level tests and provenance
    multiplicity_table.tex          table fragment for Section 6.3.1
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

# The descriptive t_max axis is excluded: it tests no hypothesis and the thesis
# builds no claim on it, so it does not belong in the confirmatory family.
CONFIRMATORY_AXES = [
    "delayed_reward_strength",
    "transition_coupling_strength",
    "fatigue_rate",
]
ARMS = ("ppo", "fqi")
ALPHA = 0.05

AXIS_LABEL = {
    "delayed_reward_strength": r"Delayed assist credit $\sigma$",
    "transition_coupling_strength": r"Action-dependent transitions $\kappa$",
    "fatigue_rate": r"Intervention fatigue $\rho$",
}
ARM_LABEL = {"ppo": r"$\Delta_{\text{PPO}}$", "fqi": r"$\Delta_{\text{FQI}}$"}


# --------------------------------------------------------------------------
# Student-t tail, self-contained.
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided p-value for a t statistic. P(|T| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2)."""
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return _betai(0.5 * df, 0.5, df / (df + t * t))


# Verbatim from ``run_sequentiality_sweep._ci95``. The published intervals were
# computed with these three-decimal constants, so the preregistered
# interval-excludes-zero decision must be reproduced with them and not with a
# more precise critical value -- otherwise a borderline cell can flip relative
# to what the thesis reports, and the comparison against BH would be measuring
# that discrepancy rather than the multiplicity adjustment.
T_CRITICAL_975_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_published(df: int) -> float:
    """The critical value the sweep actually used, for reproducing its decisions."""
    return T_CRITICAL_975_TABLE.get(df, 1.96)


def t_critical(df: int, alpha: float = ALPHA) -> float:
    """Two-sided critical value: the t with ``t_sf_two_sided(t, df) == alpha``.

    Bisection on a decreasing function, so the invariant is that the tail at
    ``lo`` is above alpha and at ``hi`` below it.
    """
    if df <= 0:
        return float("nan")
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf_two_sided(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def one_sample_t(xs: list[float]) -> tuple[float, float, float, int]:
    """Return (mean, t, two-sided p, df) for H0: mean == 0."""
    n = len(xs)
    df = n - 1
    m = sum(xs) / n
    if df <= 0:
        return m, float("nan"), float("nan"), df
    var = sum((x - m) ** 2 for x in xs) / df
    se = math.sqrt(var / n)
    if se == 0.0:
        return m, float("inf") if m else 0.0, 0.0 if m else 1.0, df
    t = m / se
    return m, t, t_sf_two_sided(t, df), df


# --------------------------------------------------------------------------
# Multiplicity adjustments.
# --------------------------------------------------------------------------
def benjamini_hochberg(pvals: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """BH step-up. Returns (adjusted q-values, rejected flags) in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0
    # Enforce monotonicity by sweeping from the largest p downward.
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        q[i] = running
    return q, [q[i] <= alpha for i in range(m)]


def holm(pvals: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm step-down family-wise adjustment. Returns (adjusted p, rejected)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order, start=1):
        running = max(running, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running)
    return adj, [adj[i] <= alpha for i in range(m)]


# --------------------------------------------------------------------------
def load_deltas(run_dir: Path, arm: str) -> dict[tuple[str, float], dict[int, float]]:
    path = run_dir / "deltas.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run the sweep first")
    column = f"delta_{arm}_minus_bandit"
    out: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no column {column!r}")
        for row in reader:
            out[(row["axis"], float(row["value"]))][int(row["seed"])] = float(row[column])
    return dict(out)


def self_check() -> list[str]:
    """Validate the t tail against values with closed forms or published digits."""
    checks = []
    # df=1 is Cauchy: P(|T| >= 1) = 1/2 exactly.
    got = t_sf_two_sided(1.0, 1)
    assert abs(got - 0.5) < 1e-12, got
    checks.append(f"t tail df=1 at t=1 -> {got:.12f} (Cauchy, exact 0.5)")
    # The critical value the sweep uses: t_{.975, 4} = 2.776445, so p ~ 0.05.
    got = t_sf_two_sided(2.776445105, 4)
    assert abs(got - 0.05) < 1e-6, got
    checks.append(f"t tail df=4 at 2.776445 -> {got:.8f} (critical value, expect 0.05)")
    # df=2 has closed form: P(|T|>=t) = 1 - t/sqrt(2+t^2).
    for t in (0.5, 1.5, 3.0):
        want = 1.0 - t / math.sqrt(2.0 + t * t)
        got = t_sf_two_sided(t, 2)
        assert abs(got - want) < 1e-12, (t, got, want)
    checks.append("t tail df=2 matches closed form at t in {0.5, 1.5, 3.0}")
    # BH on a hand-worked vector. With m=6 and alpha=0.05 the step-up threshold
    # is p_(k) <= k*alpha/m, satisfied last at k=2 (0.008 <= 0.0167) and failed
    # at k=3 (0.039 > 0.025), so exactly the first two reject. The q-values are
    # the running minimum of m*p_(j)/j swept downward from j=m.
    p_demo = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06]
    q, rej = benjamini_hochberg(p_demo, 0.05)
    q_want = [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.06]
    assert all(abs(a - b) < 1e-12 for a, b in zip(q, q_want)), (q, q_want)
    assert rej == [True, True, False, False, False, False], (q, rej)
    checks.append("BH q-values and rejections match a hand-worked six-p-value example")
    # Holm on the same vector is strictly more conservative than BH.
    hp, hrej = holm(p_demo, 0.05)
    assert all(a >= b - 1e-12 for a, b in zip(hp, q)), (hp, q)
    assert sum(hrej) <= sum(rej)
    checks.append("Holm is no less conservative than BH on the same vector")
    # t_critical must invert the tail, and must agree with the 2.776 the sweep
    # hard-codes at df=4. Getting this backwards silently declares everything
    # significant, so it is checked rather than assumed.
    assert abs(t_critical(4, 0.05) - 2.7764451) < 1e-5, t_critical(4, 0.05)
    assert abs(t_critical(14, 0.05) - 2.1447867) < 1e-5, t_critical(14, 0.05)
    checks.append("t_critical reproduces t_{.975} at df=4 (2.776445) and df=14 (2.144787)")
    # The sweep's table is the rounded form of the same quantity; confirm the
    # rounding is all that separates them, so using it is a faithfulness choice
    # rather than an accuracy one.
    for df in (4, 9, 14):
        assert abs(t_critical_published(df) - t_critical(df, 0.05)) < 5e-4, df
    checks.append("sweep's published t-table agrees with the exact tail to <5e-4 at df in {4, 9, 14}")
    return checks


def check_against_published(
    run_dir: Path, cells: list[dict[str, Any]]
) -> list[str]:
    """Assert the recomputed half-widths reproduce the run's published intervals.

    The preregistered decision is interval-excludes-zero, so if this pass does
    not reproduce the intervals the run itself reported, the ``prereg`` column
    is measuring something else and the comparison against BH is meaningless.
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    published = {
        (arm, axis, float(rec["value"])): (
            rec[f"delta_{arm}_minus_bandit_mean"], rec[f"delta_{arm}_minus_bandit_ci95"]
        )
        for axis, records in summary["axes"].items()
        for rec in records
        for arm in ARMS
    }
    worst_m = worst_h = 0.0
    checked = 0
    for c in cells:
        key = (c["arm"], c["axis"], c["value"])
        if key not in published:
            continue
        pm, ph = published[key]
        worst_m = max(worst_m, abs(c["mean"] - pm))
        worst_h = max(worst_h, abs(c["ci95_halfwidth"] - ph))
        checked += 1
    if worst_m > 1e-9 or worst_h > 1e-6:
        raise SystemExit(
            f"recomputed values diverge from {run_dir.name}/summary.json "
            f"(max |dmean| {worst_m:.3e}, max |dhalfwidth| {worst_h:.3e})"
        )
    return [f"reproduced {checked} published means and intervals from summary.json "
            f"(max drift {max(worst_m, worst_h):.2e})"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, default=HERE / "sweep_results_confirmatory_v3")
    ap.add_argument("--out-dir", type=Path, default=None, help="Defaults to --run-dir.")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir

    checks = self_check()
    for line in checks:
        print(f"[check] {line}")

    per_arm = {arm: load_deltas(args.run_dir, arm) for arm in ARMS}

    # ---- per-cell family -------------------------------------------------
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        deltas = per_arm[arm]
        for axis in CONFIRMATORY_AXES:
            for v in sorted(x for (a, x) in deltas if a == axis and x != 0.0):
                seeds = sorted(deltas[(axis, v)])
                xs = [deltas[(axis, v)][s] for s in seeds]
                m, t, p, df = one_sample_t(xs)
                # The preregistered rule, recomputed here so the two decisions
                # are produced by one pass over one dataset. Validated below
                # against the run's own published intervals.
                half = 0.0
                if df > 0:
                    var = sum((x - m) ** 2 for x in xs) / df
                    se = math.sqrt(var / len(xs))
                    half = t_critical_published(df) * se
                cells.append({
                    "arm": arm, "axis": axis, "value": v, "n_seeds": len(xs), "df": df,
                    "mean": m, "ci95_halfwidth": half, "t": t, "p_raw": p,
                    "prereg_significant": abs(m) > half,
                })

    checks += check_against_published(args.run_dir, cells)
    print(f"[check] {checks[-1]}")

    qs, rejected = benjamini_hochberg([c["p_raw"] for c in cells], args.alpha)
    for c, q, r in zip(cells, qs, rejected):
        c["p_bh"] = q
        c["bh_significant"] = r

    # ---- axis-level family ----------------------------------------------
    axis_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        deltas = per_arm[arm]
        for axis in CONFIRMATORY_AXES:
            values = sorted(x for (a, x) in deltas if a == axis and x != 0.0)
            seeds = sorted(set.intersection(*(set(deltas[(axis, v)]) for v in values)))
            per_seed = [sum(deltas[(axis, v)][s] for v in values) / len(values) for s in seeds]
            m, t, p, df = one_sample_t(per_seed)
            axis_rows.append({
                "arm": arm, "axis": axis, "n_values": len(values), "n_seeds": len(seeds),
                "df": df, "mean": m, "t": t, "p_raw": p,
            })
    holm_p, holm_rej = holm([r["p_raw"] for r in axis_rows], args.alpha)
    for r, hp, hr in zip(axis_rows, holm_p, holm_rej):
        r["p_holm"] = hp
        r["holm_significant"] = hr

    # ---- report ----------------------------------------------------------
    print(f"\nPER-CELL FAMILY ({len(cells)} tests, BH at q={args.alpha})")
    hdr = f"{'arm':4s} {'axis':30s} {'v':>5s} {'mean':>9s} {'p':>9s} {'p_BH':>9s}  prereg  BH"
    print(hdr)
    print("-" * len(hdr))
    flips = 0
    for c in cells:
        flip = c["prereg_significant"] != c["bh_significant"]
        flips += flip
        print(f"{c['arm']:4s} {c['axis']:30s} {c['value']:5g} {c['mean']:+9.3f} "
              f"{c['p_raw']:9.4f} {c['p_bh']:9.4f}  "
              f"{'YES' if c['prereg_significant'] else ' no':6s}  {'YES' if c['bh_significant'] else ' no'}"
              f"{'   <-- differs' if flip else ''}")
    print(f"\ncells significant under preregistered rule: {sum(c['prereg_significant'] for c in cells)}")
    print(f"cells surviving BH:                        {sum(c['bh_significant'] for c in cells)}")
    print(f"decisions that change:                     {flips}")

    print(f"\nAXIS-LEVEL FAMILY ({len(axis_rows)} tests, Holm at alpha={args.alpha})")
    hdr = f"{'arm':4s} {'axis':30s} {'mean':>9s} {'p':>9s} {'p_Holm':>9s}  sig"
    print(hdr)
    print("-" * len(hdr))
    for r in axis_rows:
        print(f"{r['arm']:4s} {r['axis']:30s} {r['mean']:+9.3f} {r['p_raw']:9.4f} "
              f"{r['p_holm']:9.4f}  {'YES' if r['holm_significant'] else ' no'}")

    # ---- outputs ---------------------------------------------------------
    csv_path = out_dir / "multiplicity_control.csv"
    fields = ["arm", "axis", "value", "n_seeds", "df", "mean", "ci95_halfwidth",
              "t", "p_raw", "p_bh", "prereg_significant", "bh_significant"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)

    json_path = out_dir / "multiplicity_control.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "Experiments/compute_multiplicity_control.py",
        "status": "post hoc; reported alongside the preregistered per-cell rule, "
                  "not a retrospective replacement of it (preregistration_v3.md amendment)",
        "run_dir": str(args.run_dir),
        "alpha": args.alpha,
        "per_cell": {"family_size": len(cells), "method": "Benjamini-Hochberg", "cells": cells},
        "axis_level": {"family_size": len(axis_rows), "method": "Holm",
                       "statistic": "within-seed mean of Delta(v) over the axis's non-zero settings",
                       "axes": axis_rows},
        "self_check": checks,
    }, indent=2), encoding="utf-8")

    tex = [
        "% Generated by Experiments/compute_multiplicity_control.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{Axis-level effects with family-wise control. Each entry is the within-seed mean of "
        r"$\Delta(v)$ over the axis's non-zero settings, tested against zero and Holm-adjusted across the "
        r"six tests. This is the level at which Section~\ref{subsec:EvalRQ1} argues, and it replaces "
        r"twenty-four uncontrolled per-cell decisions with six controlled ones.}",
        r"\label{tab:multiplicity}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"\textbf{Arm} & \textbf{Mechanism} & \textbf{Axis mean} & \textbf{$p$} & \textbf{$p_{\text{Holm}}$} \\",
        r"\midrule",
    ]
    for i, arm in enumerate(ARMS):
        if i:
            tex.append(r"\midrule")
        for j, r in enumerate([x for x in axis_rows if x["arm"] == arm]):
            label = ARM_LABEL[arm] if j == 0 else ""
            body = f"{r['mean']:+.3f}"
            cell = f"$\\mathbf{{{body}}}$" if r["holm_significant"] else f"${body}$"
            tex.append(f"{label} & {AXIS_LABEL[r['axis']]} & {cell} & "
                       f"${r['p_raw']:.4f}$ & ${r['p_holm']:.4f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex_path = out_dir / "multiplicity_table.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")

    print(f"\nwrote {csv_path}\n      {json_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
