# History-exposed bandit robustness result (preregistration v3 §F.1)

This is the predeclared **robustness ablation**, not an additional confirmatory
axis. It supersedes `sweep_results_expose_history/`, which ran in the
pre-`33378f3` environment.

`run_status.json` is authoritative and records **25/25 cells complete with zero
failures**; the fingerprint is
`6d7aabe536f51bf69206da22c786293aab16fa30ccc25644142a4eaea0c8cf12`. Executed
2026-08-05 at commit `98701f7` on seeds 10–14, 630 s wall clock.

The generic runner's `summary.json` calls `fatigue_rate` a "confirmatory
mechanism axis" because that label describes the axis in the main grid; it does
not change this directory's ablation role.

| Setting | Raw PPO−bandit | Anchor-adjusted Δ |
|---|---:|---:|
| all mechanisms off | +0.159 ± 0.101 | 0 |
| fatigue 0.1 | +0.086 ± 0.079 | −0.074 ± 0.099 |
| fatigue 0.2 | +0.064 ± 0.081 | **−0.096 ± 0.050** |
| fatigue 0.4 | +0.220 ± 0.083 | +0.061 ± 0.158 |
| fatigue 0.6 | +0.247 ± 0.064 | +0.088 ± 0.102 |

This ablation exists to test the most obvious alternative explanation of the
fatigue null: intervention fatigue is a function of how many widgets a session
has already seen, and the bandit does not observe that count, so the mechanism
could be said never to have been given to the CB arm to fail at. Exposing the
intervention-count feature to the bandit does not produce a positive sequential
fatigue effect. Withholding the history is therefore **not** what produced the
null, and the result reinforces the main finding rather than qualifying it. It
does **not** establish that withholding history caused a PPO advantage.
