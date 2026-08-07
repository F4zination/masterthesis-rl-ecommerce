# Fresh-seed confirmatory sequentiality sweep (SUPERSEDED)

> **SUPERSEDED — pre-`33378f3` environment. Do not cite as confirmatory
> evidence.** Commit `33378f3` (2026-07-28) changed the simulator after this run
> completed. It altered rewards and features in every cell and, decisively, the
> all-knobs-zero **anchor** that every Δ in this file is computed against.
> `preregistration_v3.md` §A lists the five changes and §G fixes this
> directory's status: retained as a disclosed historical record, citable only as
> explicitly labelled pre-`33378f3` context, never pooled with or substituted
> for the `_v3` runs, and no figure may combine cells from both environments.
> The current confirmatory evidence is `sweep_results_confirmatory_v3/`. The
> `final_outputs_complete: true` flag below is true and is **not** sufficient
> authority to cite this run.

`run_status.json` is authoritative: **95/95 cells complete, zero failures,
`final_outputs_complete: true`**. The run fingerprint is
`5ee55664e82e07895eb02e65215d2e66e9abde759b1894e8a194e6f005940298`.
It matches `summary.json` and `run_manifest.json` (`fingerprint`).

Values below are means with pointwise two-sided 95% Student-t interval
half-widths across fresh seeds 5–9. The primary mechanism contrast is the
within-seed anchor-adjusted change in PPO−bandit gap.

| Setting | Raw PPO−bandit | Anchor-adjusted Δ |
|---|---:|---:|
| all mechanisms off | +0.172 ± 0.112 | 0 |
| delayed reward 0.25 | +0.516 ± 0.076 | **+0.344 ± 0.080** |
| delayed reward 0.5 | +0.292 ± 0.187 | +0.121 ± 0.239 |
| transition coupling 0.5 | +0.406 ± 0.100 | **+0.234 ± 0.180** |
| transition coupling 1.0 | +0.642 ± 0.113 | **+0.470 ± 0.095** |
| transition coupling 2.0 | +0.624 ± 0.107 | **+0.452 ± 0.193** |
| transition coupling 3.0 | +0.457 ± 0.214 | +0.285 ± 0.316 |
| fatigue 0.4 | +0.129 ± 0.083 | −0.043 ± 0.192 |
| fatigue 0.6 | +0.111 ± 0.063 | −0.061 ± 0.169 |

The result supports low delayed reward and low-to-moderate transition coupling,
with saturation/decline. It does not support a fatigue-driven gain. FQI does
not track PPO and is strongly negative relative to the bandit under coupling,
so the result cannot isolate sequential credit assignment from online data
access, interaction budget, function approximation, or optimization.

Integrity checks on 2026-07-21 found 95 unique axis/value/seed rows in both
`gaps.csv` and `deltas.csv`, 475 rows in `long_results.csv`, 95 atomic cell
records, and byte-identical final/partial CSV pairs. Temporary model/database
files from the interrupted legacy attempt and its stale `INCOMPLETE.md` marker
were removed after final-output validation.
