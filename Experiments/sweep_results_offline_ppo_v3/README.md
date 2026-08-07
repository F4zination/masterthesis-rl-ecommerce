# Offline-PPO ablation interpretation (preregistration v3 §F.2)

This directory contains the completed, predeclared **offline-PPO robustness
ablation**, not an additional confirmatory analysis. It supersedes
`sweep_results_offline_ppo/`, which ran in the pre-`33378f3` environment.

`run_status.json` records **75/75 cells complete with zero failures**; the
fingerprint is
`2f70cdc5672b90fab9b97e0bd0aff00725438d158e6fb2c0b3270240c6371956`. Executed
2026-08-05 at commit `98701f7` on seeds 10–14, 3524 s wall clock. The immutable
settings and implementation hashes are in `run_manifest.json`.

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
- Because `--ppo-mode offline` changes only the PPO arm, the bandit and FQI arms
  are identical to the §D run at the same seeds; the shared FQI−bandit anchor of
  −0.083 ± 0.055 confirms this.

## Result

Values are mean PPO−bandit reward gaps with pointwise 95% Student-t interval
half-widths; the second column is the within-seed change from the zero-mechanism
anchor. Bold marks intervals excluding zero.

| Setting | Raw gap | Anchor-adjusted Δ |
|---|---:|---:|
| all mechanisms off | +0.027 ± 0.098 | 0 |
| delayed reward 0.25 | +0.278 ± 0.069 | **+0.251 ± 0.082** |
| delayed reward 0.5 | +0.207 ± 0.115 | **+0.180 ± 0.149** |
| delayed reward 1.0 | +0.180 ± 0.060 | **+0.152 ± 0.106** |
| delayed reward 2.0 | +0.171 ± 0.072 | **+0.144 ± 0.118** |
| transition coupling 0.5 | +0.167 ± 0.167 | +0.140 ± 0.171 |
| transition coupling 1.0 | +0.360 ± 0.287 | **+0.333 ± 0.319** |
| transition coupling 2.0 | +0.198 ± 0.130 | **+0.171 ± 0.144** |
| transition coupling 3.0 | +0.214 ± 0.139 | **+0.187 ± 0.118** |
| fatigue 0.1 | +0.038 ± 0.035 | +0.011 ± 0.108 |
| fatigue 0.2 | +0.002 ± 0.102 | −0.025 ± 0.061 |
| fatigue 0.4 | +0.081 ± 0.094 | **+0.053 ± 0.044** |
| fatigue 0.6 | +0.035 ± 0.150 | +0.008 ± 0.083 |

The positive all-off anchor seen for online PPO collapses to an interval
covering zero under fixed-dataset training, so that anchor asymmetry is
substantially an artefact of online data collection. The delayed-reward and
coupling effects nonetheless survive the restriction. In this simulator, fresh
online interaction is therefore not necessary for those two mechanism effects.

Read together with the FQI result in the §D run — where the sequential
objective is present and gains nothing — this points at the capacity of the
function approximator over the state as the remaining distinguishing factor.
That reading is suggestive, not decisive: state access, objective and
optimization still differ between the arms after the restriction, this
fixed-dataset PPO implementation is not a validated general-purpose offline-RL
estimator, and the comparison is exploratory by declaration.

The fatigue 0.4 cell (+0.053 ± 0.044) is the only positive fatigue interval
excluding zero anywhere in the v3 confirmatory or ablation runs. It is small, it
carries the narrowest interval on its axis, and it appears in an exploratory
analysis. It is recorded in `preregistration_v3.md` §H.3 for completeness and no
claim is built on it.
