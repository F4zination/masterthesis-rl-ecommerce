# Sequentiality sweep: run and ablation runbook

**Status (2026-08-05):** the `preregistration_v3.md` §D confirmatory run and both
§F ablations are complete on seeds 10–14 with zero failures:

| Run | Directory | Cells | Fingerprint |
|-----|-----------|-------|-------------|
| §D confirmatory | `sweep_results_confirmatory_v3` | 95/95 | `d4c312393ff2df6a…` |
| §F.1 expose-history | `sweep_results_expose_history_v3` | 25/25 | `6d7aabe536f51bf6…` |
| §F.2 offline-PPO | `sweep_results_offline_ppo_v3` | 75/75 | `2f70cdc5672b90fa…` |

**These three directories are the confirmatory evidence for the thesis.** The
post-result record is `preregistration_v3.md` §H; the thesis reports them in
Section `sec:ExpSweep` of `MasterThesis/Content/Experiments.tex`.

## Do not cite the pre-`33378f3` runs

`sweep_results_confirmatory/`, `sweep_results_expose_history/` and
`sweep_results_offline_ppo/` (no `_v3` suffix) completed on 2026-07-21 on seeds
5–9 and each still declares `final_outputs_complete: true`. **That flag is not
sufficient authority to cite them.** Commit `33378f3` ("fix: align policy
training and serving domains", 2026-07-28) changed the simulator afterwards, so
they describe an environment that no longer exists. `preregistration_v3.md` §A
enumerates the five changes and §G fixes their status: retained as a disclosed
historical record, citable only as explicitly labelled pre-`33378f3` context,
never pooled with or substituted for the `_v3` runs, and no figure may combine
cells from both environments.

The decisive change is that latent assist credit now accumulates
unconditionally, where it was previously gated on `delayed_reward_strength > 0`.
That alters the all-knobs-zero **anchor**, and since `Δ(v)` is defined by
subtracting the anchor, it propagates to every contrast. The 2026-07-13
interrupted attempt remains a disclosed historical event on the same terms.

**Before citing any output, check two things, not one:** that its own
`run_status.json` says `final_outputs_complete: true`, *and* that its
`run_manifest.json` `implementation_sha256` entries still match the working tree.
A run whose pinned files have moved is stale regardless of what its status file
says.

## Keep the environment frozen until submission

Any commit that modifies `CustomerSimulation/simulation/`,
`SharedSchema/shared_schema/features.py`, `shared_schema/constants.py`, the
trainers under `OfflineTraining/`, or `run_sequentiality_sweep.py` invalidates
the `_v3` runs exactly as `33378f3` invalidated their predecessors, and requires
a fourth run in fresh directories on fresh seeds plus a further preregistration
amendment. This is the single most likely way to lose the experiment before
submission. If a fix to those paths becomes unavoidable — for instance one
surfaced by the human study — treat re-running the sweep as part of the cost of
that fix and budget about two hours of compute for it.

## Running it

Make targets wrap the preregistered commands; the seeds and output directories
are fixed by `preregistration_v3.md` and should be overridden only for a
documented ablation:

```text
make sweep-v3-smoke          # wiring check, ~1 min, throwaway output
make sweep-v3-confirmatory   # §D, 95 cells
make sweep-v3-expose-history # §F.1, 25 cells
make sweep-v3-offline-ppo    # §F.2, 75 cells
make sweep-v3                # all three in the preregistered order
make sweep-v3-status         # print run_status.json for each _v3 directory
```

`sweep-v3` chains the three as prerequisites rather than a recursive `make`
call, so it is not affected by the unquoted `C:/Program Files (x86)` expansion
that breaks `study-policies` on the Windows dev machine. Run `sweep-v3-smoke`
first after any toolchain change: `33378f3` added a `ValueError` in
`_make_next_state` for cart/checkout transitions arriving with an empty cart,
and the smoke run exercises that path in about four seconds.

The equivalent explicit commands are in `preregistration_v3.md` §D and §F.

## Scope and measured runtime

The §D command contains 95 cells: 75 confirmatory mechanism cells (3 axes × 5
values × 5 seeds) and 20 descriptive `t_max` cells (4 values × 5 seeds). Each
cell simulates 6,000 dataset sessions, 10,240 online PPO rollout sessions, and
25,000 evaluation sessions, or about 3.92 million sessions for the complete grid.

Measured on the Windows dev machine (12 logical cores, Python 3.14.2,
torch 2.12.0+cpu) on 2026-08-05:

| Run | Cells | Wall clock | Per cell |
|-----|-------|-----------|----------|
| §D confirmatory | 95 | 2788 s (46 min) | ~29 s |
| §F.1 expose-history | 25 | 630 s (11 min) | ~25 s |
| §F.2 offline-PPO | 75 | 3524 s (59 min) | ~47 s |

About **two hours** for all three sequentially. Offline PPO is the slowest per
cell because it updates over the entire fixed dataset each iteration. These are
measurements on one machine, not promised bounds.

## Resume and fail-closed behaviour

The runner writes an immutable `run_manifest.json`, an authoritative
`run_status.json`, partial CSVs, and one atomic `cells/*.json` result after every
completed cell. Every run is safe to interrupt after a completed cell, and
re-running the identical command resumes from the durable cell cache. A code,
archetype, grid, or hyperparameter mismatch fails closed and requires a new
output directory rather than silently mixing environments.

For a smaller explicitly documented ablation, `--axis-values` can restrict each
axis, for example `--axis-values fatigue_rate=0,0.6`. Never place a restricted
and a full-grid run in the same output directory; the manifest prevents this.

## Interpretation guardrails

- `fatigue_rate`, `delayed_reward_strength`, and
  `transition_coupling_strength` are the confirmatory mechanism axes.
- `t_max` is descriptive only. The v3 run shows it is not merely weak but
  entirely inactive above 30: the 30, 40 and 60 cells are identical to ten
  decimal places, so no session in the grid runs beyond thirty steps.
- Confidence intervals are pointwise Student-t intervals across seeds. With five
  seeds this is materially wider than a 1.96 × SE normal approximation.
- The history and offline-PPO runs are exploratory robustness evidence, not
  extra confirmatory axes.
- **The FQI reading rule binds the conclusion.** In the v3 run FQI does not
  track PPO on either winning axis: its anchor-adjusted change is null across
  the delayed-reward axis and substantially *negative* across the coupling axis
  (−0.345 to −0.799, every interval excluding zero). Under the predeclared rule
  this means sequential credit assignment alone is **not** identified as the
  cause of PPO's advantage; deep function approximation and/or online training
  are necessary in addition. Do not write "sequentiality explains the gain".
- Offline PPO has a near-zero all-off gap (+0.027 ± 0.098, covering zero) but
  retains the delayed-reward and coupling effects. Fresh online interaction is
  therefore not necessary for those effects in this setup; this does not
  equalize state access, model class, objective, or optimization, and the
  fixed-dataset routine is not a validated general-purpose offline-RL estimator.
- Fatigue is unsupported. One isolated positive interval excluding zero exists
  (offline PPO, `fatigue_rate=0.4`, +0.053 ± 0.044); it is recorded in §H.3 and
  no claim is built on it.
