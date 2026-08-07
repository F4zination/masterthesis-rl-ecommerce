# History-exposed bandit robustness result (SUPERSEDED)

> **SUPERSEDED — pre-`33378f3` environment. Do not cite.** Commit `33378f3`
> (2026-07-28) changed the simulator after this run completed; see
> `preregistration_v3.md` §A and §G. Retained as a disclosed historical record
> only. The current version of this ablation is
> `sweep_results_expose_history_v3/`.

This is the predeclared **robustness ablation**, not an additional confirmatory
axis. `run_status.json` is authoritative and records 25/25 cells complete with
zero failures. The generic runner's `summary.json` calls `fatigue_rate` a
"confirmatory mechanism axis" because that label describes the axis in the
main grid; it does not change this directory's ablation role.

When the bandit receives the history feature, the PPO-minus-bandit gap does not
grow materially relative to the zero-fatigue anchor: the anchor-adjusted gaps
are +0.013 (95% CI half-width 0.099) at fatigue 0.4 and +0.014 (half-width
0.084) at fatigue 0.6. The main comparison without bandit history likewise
shows no positive anchor-adjusted fatigue effect. The robustness result
therefore reinforces the null fatigue finding; it does **not** establish that
withholding history caused a PPO advantage.
