# Pre-registration v2 (amendment): When is full RL worth it over a contextual bandit?

**Status:** amendment to `preregistration.md` (v1, commit `03808d5`,
2026-06-25 19:51:21 +0200). This version (a) discloses the exploratory runs that
preceded and followed the v1 commit, (b) fixes the hyperparameters that were
*actually* used (v1 declared different values), and (c) defines a **confirmatory
run on fresh seeds** that were never used in any prior run. Everything in v1 not
explicitly amended here still applies (environment mechanisms, functional forms,
asymmetry construction, algorithms, metric, significance rule).

---

## A. Disclosure of exploratory runs (2026-06-25)

Full transparency on the timeline, because v1's blindness claim does not hold as
originally stated:

1. **Pilot sweep — `Experiments/sweep_results/` (written 19:44, i.e. ~7 minutes
   *before* the v1 commit).** 3 seeds `{0,1,2}`, `transition_coupling_strength`
   axis only, script defaults (dataset 4000, eval 3000, PPO iterations 25,
   rollout sessions 128, FQI iters 25). It showed a positive PPO−bandit gap
   growing with coupling. **Consequently, H1/H2 for the coupling axis were not
   blind at v1 commit time.** The other three axes had not been run.

2. **First full sweep — `Experiments/sweep_results_thesis/` (20:02–20:38, after
   the v1 commit).** Seeds `{0,1,2,3,4}`, all four axes, but with hyperparameters
   deviating from v1 §3: PPO iterations 25→40, rollout sessions 128→256, dataset
   4000→6000, eval 3000→5000, FQI iters 25→30.

Both runs are hereby reclassified as **exploratory/pilot**. Their outputs are
committed alongside this file for the record. The confirmatory evidence for the
thesis is the run defined in §B, on seeds disjoint from everything above.

**Pilot observations that inform the amended analysis plan (declared now, before
the confirmatory run):**

- **H0 was partially violated.** At all-knobs-0 the PPO−bandit gap was
  significantly positive on 2 of 4 axis anchors (+0.12 ± 0.05 and +0.09 ± 0.05;
  the anchors are independent PPO retrainings of the same configuration). The
  suspected cause is a residual training-regime asymmetry: PPO trains *online*
  against the simulator while the bandit and FQI train *offline* from a logged
  dataset. §C makes the anchor gap an explicit reported quantity and switches the
  primary contrast to be anchor-adjusted.
- **FQI−bandit was ≤ 0 almost everywhere** (strongly negative under coupling).
  Per v1 §8's pre-declared reading rule, if this replicates the conclusion is
  "function approximation and/or online training matter too" — the win may NOT
  be attributed to sequentiality alone.
- **The delayed-reward gap was non-monotone** (peak at strength 0.25, then
  decline as the bandit also benefits and the purchase-probability clamp
  saturates). H1 is amended accordingly (§C).
- **The `t_max` axis did not bind:** bandit and FQI results were bit-identical at
  t_max 30/40/60 because sessions terminate naturally well before 30 steps. The
  axis is demoted to descriptive-only (§B); no horizon claim will be made unless
  the knob is first made binding (out of scope for this amendment).

---

## B. Confirmatory run (fixed)

**Fresh seeds:** `{5, 6, 7, 8, 9}` — disjoint from all pilot seeds. Evaluation
remains paired per policy via `eval_seed_base = 10000`.

**Hyperparameters (as actually used in the 20:02 pilot, now the declared spec):**

| Parameter | Value |
|---|---|
| dataset sessions / cell | 6000 |
| eval sessions / cell | 5000 |
| `t_max` (non-horizon axes) | 20 |
| PPO mode | online (on-policy simulator rollouts) |
| PPO iterations | 40 |
| PPO rollout sessions / iteration | 256 |
| PPO epochs / minibatch | 4 / 256 |
| hidden sizes | (128, 64) |
| γ / GAE λ | 0.95 / 0.95 |
| learning rate | 3e-4 |
| clip / entropy / value coef / grad-clip | 0.2 / 0.01 / 0.5 / 0.5 (script defaults, unchanged from v1) |
| FQI iterations / conservative penalty | 30 / 0.15 |
| Bandit | Bayesian-smoothed greedy, `PRIOR_COUNT=5`, `PRIOR_MEAN=0` (unchanged) |

**Grid (unchanged from v1 §4):** `fatigue_rate` ∈ {0, 0.1, 0.2, 0.4, 0.6};
`delayed_reward_strength` ∈ {0, 0.25, 0.5, 1.0, 2.0};
`transition_coupling_strength` ∈ {0, 0.5, 1.0, 2.0, 3.0}. The `t_max` axis
{20, 30, 40, 60} may be run for completeness but is **descriptive only** (known
not to bind; see §A).

**Command:**

```
python Experiments/run_sequentiality_sweep.py \
  --out-dir Experiments/sweep_results_confirmatory \
  --seeds 5 6 7 8 9 \
  --dataset-sessions 6000 --eval-sessions 5000 \
  --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30
```

---

## C. Amended analysis plan

- **Anchor gap is a first-class reported quantity.** For each axis, report the
  all-knobs-0 PPO−bandit gap with its 95% CI. It estimates the residual
  online-vs-offline / optimizer asymmetry that survives with zero sequential
  structure.
- **Primary contrast (amended):** the **anchor-adjusted gap**
  `Δ(v) = gap(knob = v) − gap(knob = 0)` per axis, mean ± 95% CI across seeds
  (seeds are paired across cells, so the difference is computed within-seed).
  The unadjusted gap curves are still reported as in v1.
- **H0′:** at all knobs = 0, the *unadjusted* gap may be positive (pilot: ~+0.1);
  it is reported and attributed to the training-regime asymmetry, not to
  sequential structure.
- **H1′ (amended):** `Δ(v) > 0` for knob values above 0, growing over the low
  range of each mechanism knob; **saturation or decline at high values is
  permitted** and does not count against the hypothesis (mechanisms saturate,
  e.g. the purchase-probability clamp).
- **H2 (unchanged in spirit):** there is a threshold above which `Δ(v)` is
  significant (95% CI excludes 0).
- **FQI reading rule (unchanged from v1 §8):** if FQI tracks PPO, the win is
  sequentiality; if only PPO wins (as the pilot suggests), the reported
  conclusion is that deep function approximation and/or online training are
  necessary in addition to sequential credit assignment.

## D. Pre-declared follow-ups (not part of the confirmatory run)

1. **Expose-history robustness** (v1 §7, still owed): repeat the `fatigue_rate`
   axis with `SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT=1`.
2. **Online/offline ablation:** `--ppo-mode offline` at the anchor and at one
   elevated point per mechanism axis, to decompose the anchor gap into
   "online data access" vs. everything else.

## E. Implementation notes (2026-07-13, recorded before inspecting confirmatory results)

Declared here for transparency; none of these changes touch the environment
(simulator mechanisms, functional forms, grids, reward table, feature schema —
all unchanged from v1/v2):

1. **Δ(v) implemented.** `run_sequentiality_sweep.py` now computes the §C
   anchor-adjusted contrast within-seed and emits it as `deltas.csv`,
   `delta_vs_<axis>.svg`, per-value `delta_*` fields in `summary.json`, and a
   first-class `summary.anchor` block per axis.
2. **Reproducibility fixes.** PPO minibatch shuffling now uses the run's
   seeded RNG (was: unseeded global `random.shuffle`), and the FQI greedy
   argmax iterates actions in sorted order (was: hash-randomized set order,
   nondeterministic across processes). Two independent smoke-run processes now
   produce bit-identical `gaps.csv`. Verified 2026-07-13.
3. **Runs attempted 2026-07-13:** the §B confirmatory process was interrupted
   after writing the fatigue and delayed-reward plots and while processing
   coupling strength 3.0, seed 5. It wrote no seed-level tables or final
   `summary.json`, so it is not a completed confirmatory result. No result
   artifacts exist for the §D.1 expose-history or §D.2 offline-PPO runs; both
   remain outstanding.

## F. Post-interruption engineering note (2026-07-21)

This note was added after the partial plots above existed and is not represented
as a blind amendment to the hypotheses or grid.

1. `run_sequentiality_sweep.py` now atomically caches every completed cell and
   fingerprints the script, archetype file, grid, and effective settings.
   Exact-command reruns resume; mismatched settings fail closed. The legacy
   partial run has no cell cache and therefore must restart from cell one.
2. `--expose-history-to-bandit` now makes the history robustness setting
   explicit in the command and manifest. `--axis-values` supports documented
   reduced ablation grids. `summary.json` labels `t_max` descriptive rather
   than confirmatory.
3. Newly generated confidence intervals use pointwise Student-t critical values
   across seeds. The prior implementation used 1.96 × SE; with only five seeds
   that understated uncertainty. This reporting correction is explicitly
   post-interruption and may change whether a pointwise interval excludes zero.
4. A stored integration regression now verifies that two independent smoke-run
   processes produce byte-identical seed-level CSVs.
5. Commands, completeness checks, and evidence-based runtime estimates are in
   `Experiments/SequentialitySweepRunbook.md`.

## G. Post-result record (2026-07-21; not a preregistration amendment)

The exact §B restart completed all 95 expected cells on fresh seeds 5–9 with
zero failures. `sweep_results_confirmatory/run_status.json` records
`final_outputs_complete: true` and fingerprint
`5ee55664e82e07895eb02e65215d2e66e9abde759b1894e8a194e6f005940298`.
The intervals below are the predeclared within-seed estimates with the repaired
pointwise Student-t 95% half-widths.

- All-mechanisms-off raw PPO−bandit anchor: **+0.172 ± 0.112**.
- Delayed reward 0.25: raw gap +0.516 ± 0.076; anchor-adjusted
  **Δ=+0.344 ± 0.080**. At 0.5, 1.0, and 2.0, Δ declines and its intervals
  include zero.
- Transition coupling 0.5, 1.0, and 2.0: respectively
  **Δ=+0.234 ± 0.180**, **+0.470 ± 0.095**, and
  **+0.452 ± 0.193**. At 3.0, Δ=+0.285 ± 0.316.
- Every nonzero fatigue setting has a non-positive anchor-adjusted mean; the
  fatigue hypothesis is not supported.
- `t_max` changes the gap by approximately −0.004 ± 0.070 from 30 onward and
  remains descriptive-null.
- FQI does not track PPO: anchor FQI−bandit is −0.091 ± 0.057 and becomes
  substantially negative under transition coupling. Under the §C reading rule,
  sequentiality alone is therefore **not** identified as the cause of PPO's
  advantage.

The predeclared history-exposed-bandit fatigue robustness run also completed
25/25 cells with zero failures. Its high-fatigue anchor-adjusted PPO−bandit
changes are +0.013 ± 0.099 at 0.4 and +0.014 ± 0.084 at 0.6. It reinforces the
null fatigue result and does not show that withholding history caused a PPO
advantage.

## H. Offline-PPO ablation record (2026-07-21; exploratory robustness)

The full three-axis fixed-dataset PPO ablation completed 75/75 cells with zero
failures. `sweep_results_offline_ppo/run_status.json` records
`final_outputs_complete: true` and fingerprint
`906380ce9d5ba93ae1ae62b563620ab5354d70d9993e871189fbf60b31613e43`.

The all-mechanisms-off PPO−bandit gap is −0.017 ± 0.037, eliminating the
positive online-PPO anchor. Delayed reward retains positive anchor-adjusted
changes of +0.296 ± 0.115 at 0.25, +0.252 ± 0.074 at 0.5, +0.175 ± 0.041 at
1.0, and +0.172 ± 0.040 at 2.0. Transition coupling retains +0.228 ± 0.130 at
0.5, +0.512 ± 0.250 at 1.0, and +0.435 ± 0.141 at 2.0; the interval crosses
zero at 3.0. Fatigue remains null.

This post-result ablation shows that fresh online simulator interaction is not
necessary for the delayed-reward and moderate-coupling effects in this setup.
It does not identify sequential credit assignment alone: the fixed-dataset PPO
routine is not a validated general-purpose offline-RL estimator, and state
access, function approximation, objective, and optimization still differ from
the contextual bandit.
