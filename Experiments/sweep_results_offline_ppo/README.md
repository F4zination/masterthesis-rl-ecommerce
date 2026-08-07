# Offline-PPO ablation interpretation (SUPERSEDED)

> **SUPERSEDED — pre-`33378f3` environment. Do not cite.** Commit `33378f3`
> (2026-07-28) changed the simulator after this run completed; see
> `preregistration_v3.md` §A and §G. Retained as a disclosed historical record
> only. The current version of this ablation is
> `sweep_results_offline_ppo_v3/`.

This directory contains the completed, predeclared **offline-PPO robustness
ablation**, not an additional confirmatory analysis. The immutable settings and
implementation hashes are recorded in `run_manifest.json`; completion is
authoritatively recorded in `run_status.json`.

Interpretation caveats:

- `summary.json` inherits generic axis-role labels that call these
  "confirmatory mechanism axes." In this directory those labels describe the
  grid variables only; the run's inferential role remains an ablation.
- The value-zero cells are deliberately identical across the three axes. Do not
  pool them or count them as independent replications.
- Confidence intervals are pointwise Student-t intervals over five seeds and
  are not adjusted for multiple comparisons.
- `ppo_mode` is `offline`: PPO repeatedly updates over each fixed simulated
  dataset. This is distinct from the on-policy confirmatory sweep and from the
  frozen PPO policy deployed in the Clickworker study.

The command used was the exact full-grid command recorded in
`Experiments/SequentialitySweepRunbook.md`. It completed 75 of 75 cells with no
failures and produced all final tables and plots.

## Result

Values are mean PPO−bandit reward gaps with pointwise 95% Student-t interval
half-widths; parentheses show the within-seed change from the zero-mechanism
anchor.

| Setting | Raw gap | Anchor-adjusted Δ |
|---|---:|---:|
| all mechanisms off | −0.017 ± 0.037 | 0 |
| delayed reward 0.25 | +0.279 ± 0.095 | **+0.296 ± 0.115** |
| delayed reward 0.5 | +0.235 ± 0.042 | **+0.252 ± 0.074** |
| delayed reward 1.0 | +0.158 ± 0.019 | **+0.175 ± 0.041** |
| delayed reward 2.0 | +0.155 ± 0.032 | **+0.172 ± 0.040** |
| transition coupling 0.5 | +0.211 ± 0.105 | **+0.228 ± 0.130** |
| transition coupling 1.0 | +0.495 ± 0.234 | **+0.512 ± 0.250** |
| transition coupling 2.0 | +0.418 ± 0.156 | **+0.435 ± 0.141** |
| transition coupling 3.0 | +0.223 ± 0.281 | +0.240 ± 0.283 |
| fatigue 0.4 | −0.013 ± 0.068 | +0.004 ± 0.062 |
| fatigue 0.6 | −0.042 ± 0.077 | −0.025 ± 0.058 |

The all-off advantage seen for online PPO disappears under fixed-dataset
training, while the delayed-reward and moderate-coupling changes remain close
to the online sweep's magnitudes. In this simulator, fresh online interaction
is therefore not necessary for those two mechanism effects. This ablation does
not by itself identify sequential credit assignment: PPO still differs from
the bandit in state/history access, function approximation, objective, and
optimization, and this fixed-dataset PPO implementation is not a validated
general-purpose offline-RL estimator.
