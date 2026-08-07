# Confirmatory sequentiality sweep (preregistration v3 §D)

**This is the confirmatory evidence for the thesis.** It supersedes
`sweep_results_confirmatory/`, which ran in the pre-`33378f3` environment.

`run_status.json` is authoritative: **95/95 cells complete, zero failures,
`final_outputs_complete: true`**. The run fingerprint is
`d4c312393ff2df6af2cabe42808eb9ec116a550a8287effbdd3016ad7b049bf3`.
It matches `summary.json` and `run_manifest.json` (`fingerprint`).

Executed 2026-08-05 at commit `98701f7` on fresh seeds 10–14, disjoint from
every prior run. Python 3.14.2, torch 2.12.0+cpu, 2788 s wall clock.

Values below are means with pointwise two-sided 95% Student-t interval
half-widths across the five seeds. The primary mechanism contrast is the
within-seed anchor-adjusted change in the PPO−bandit gap. Bold marks intervals
excluding zero.

| Setting | Raw PPO−bandit | Anchor-adjusted Δ |
|---|---:|---:|
| all mechanisms off | +0.141 ± 0.052 | 0 |
| delayed reward 0.25 | +0.494 ± 0.046 | **+0.353 ± 0.053** |
| delayed reward 0.5 | +0.366 ± 0.038 | **+0.225 ± 0.048** |
| delayed reward 1.0 | +0.258 ± 0.130 | **+0.117 ± 0.109** |
| delayed reward 2.0 | +0.261 ± 0.131 | **+0.120 ± 0.110** |
| transition coupling 0.5 | +0.357 ± 0.075 | **+0.216 ± 0.121** |
| transition coupling 1.0 | +0.576 ± 0.155 | **+0.434 ± 0.174** |
| transition coupling 2.0 | +0.432 ± 0.180 | **+0.291 ± 0.190** |
| transition coupling 3.0 | +0.391 ± 0.159 | **+0.250 ± 0.133** |
| fatigue 0.1 | +0.090 ± 0.052 | **−0.051 ± 0.028** |
| fatigue 0.2 | +0.042 ± 0.112 | −0.100 ± 0.112 |
| fatigue 0.4 | +0.194 ± 0.070 | +0.052 ± 0.076 |
| fatigue 0.6 | +0.200 ± 0.100 | +0.059 ± 0.084 |

The result supports delayed assist credit and transition coupling across their
whole grids, both peaking at a low-to-moderate value and declining thereafter —
the saturation the amended alternative hypothesis permitted in advance. It does
not support a fatigue-driven gain; at fatigue 0.1 the effect is significantly
*negative*.

`t_max` is descriptive only and is inactive above 30: the 30, 40 and 60 cells
are identical to ten decimal places in all three arms, so no session in the grid
runs beyond thirty steps.

**FQI does not track PPO.** Its anchor-adjusted change is null across the
delayed-reward axis (within [−0.016, +0.028], every interval covering zero) and
substantially negative across the coupling axis (−0.345 ± 0.122 at 0.5,
deteriorating to −0.799 ± 0.126 at 2.0, every interval excluding zero). Under
the predeclared reading rule this means sequential credit assignment alone is
**not** identified as the cause of PPO's advantage; deep function approximation
and/or online training are necessary in addition. Do not write "sequentiality
explains the gain".

The full post-result record, including the descriptive comparison against the
superseded run, is `preregistration_v3.md` §H.
