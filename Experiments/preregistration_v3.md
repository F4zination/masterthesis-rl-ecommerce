# Pre-registration v3 (amendment): When is full RL worth it over a contextual bandit?

**Status:** amendment to `preregistration_v2.md`, which is itself an amendment to
`preregistration.md` (v1, commit `03808d5`, 2026-06-25 19:51:21 +0200).

This version exists for one reason: **commit `33378f3` ("fix: align policy
training and serving domains", 2026-07-28 19:30:03 +0200) changed the
environment.** v2 §E asserted that its implementation notes did *not* touch the
environment — "simulator mechanisms, functional forms, grids, reward table,
feature schema — all unchanged from v1/v2". That assertion is true of v2's own
changes and **false of `33378f3`**. The completed confirmatory run recorded in
v2 §G and the ablation in v2 §H therefore describe an environment that no longer
exists.

v3 (a) discloses exactly what changed and why, (b) supersedes the v2 §G/§H
records as confirmatory evidence, (c) defines a new confirmatory run on fresh
seeds disjoint from every prior run, and (d) states plainly that this run is
**not blind**. Everything in v1/v2 not amended here still applies: grids,
algorithms, the anchor-adjusted metric, the hypotheses, the FQI reading rule,
and the significance rule.

---

## A. Disclosure: what changed in the environment

All five changes are in commit `33378f3`. They were made to remove a
training/serving domain mismatch that blocked the Clickworker study release
(missing PPO vocabulary tokens; bandit contexts and arms with no trained
evidence). None was made in response to sweep results.

| # | File | Change | Effect on the sweep |
|---|---|---|---|
| 1 | `simulation/state.py` | `SimState.page_depth` default `0` → `1` | With thresholds (1, 3, 6) the landing-page `page_depth_bucket` moves from 0 to 1. Changes bandit context keys and PPO state tokens in every cell. |
| 2 | `simulation/simulator.py` | `page_depth` no longer increments when the next stage is `scroll_engagement` or `done` | Changes the `page_depth_bucket` trajectory within a session. |
| 3 | `simulation/simulator.py` | `device_type` drawn from `DEVICE_TYPES` (4 values, incl. `unknown`) instead of `["mobile","tablet","desktop"]` | Adds a fourth context value; changes the context distribution and the support the bandit and PPO see. |
| 4 | `simulation/simulator.py` | New `_ensure_cart_reachability_events`: a sampled transition into `cart`/`checkout` from an empty cart now materialises an `add_to_cart` event before reward computation | **Changes transition rewards and successor states.** `add_to_cart` carries reward, so per-session return changes. |
| 5 | `simulation/simulator.py` | `primed_credit` now accumulates unconditionally; previously gated on `delayed_reward_strength > 0` | **Changes the all-knobs-zero anchor.** The removed code carried the comment "so primed_credit stays 0 (and primed_credit_bucket constant) in the off/anchor configuration". |

**Motivation.** Changes 1, 2, and 4 make the simulator agree with production
reachability and browser semantics: production emits `cart`/`checkout` only
with a non-empty cart, a scroll opportunity does not load a new page, and the
browser's page counter is already 1 at the landing decision. Change 3 covers the
`unknown` device fallback that production emits. Change 5 matches production,
which reconstructs served-action history regardless of whether delayed reward is
active. The simulator is more faithful after `33378f3`, not merely different.

**Explicitly unchanged by `33378f3`:** the reward table (`EVENT_REWARD`,
`ACTION_COST` — the `constants.py` diff is purely additive), the functional forms
of all three sequential mechanisms, the bucket threshold *values* (moved from
inline literals into `FEATURE_BUCKET_THRESHOLDS`, value-identical), the grids,
the algorithms, and the metric definitions.

---

## B. Why the v2 confirmatory evidence is superseded

The primary contrast is the anchor-adjusted gap
`Δ(v) = gap(knob = v) − gap(knob = 0)`. Change 5 alters the **anchor itself**:
under v2 the all-knobs-zero configuration held `primed_credit_bucket` constant,
so the anchor policies operated on a strictly smaller state space than they do
now. Changes 1–4 additionally alter features, transitions, and rewards in every
cell, anchor included.

Consequently the v2 §G numbers — anchor +0.172 ± 0.112, delayed-reward
Δ=+0.344 ± 0.080 at 0.25, coupling Δ=+0.470 ± 0.095 at 1.0, null fatigue — and
the v2 §H offline ablation are **not** estimates of the current environment.
They are retained as a disclosed historical record (§G below) and are not
reported as confirmatory evidence for the thesis.

The sweep runner enforces this independently: `run_sequentiality_sweep.py`
fingerprints the script, archetype file, grid, and effective settings, so a
rerun into `sweep_results_confirmatory/` fails closed rather than silently
mixing environments.

---

## C. Blindness status (declared before the run)

**This run is not blind, and no claim to the contrary is made.** The v2 §G and
§H results are known and are quoted above. This mirrors the disclosure v2 §A
made about its own pilots.

Three commitments follow, made before the new run starts:

1. **Nothing downstream of the environment change is retuned.** Grid,
   hyperparameters, seeds count, metric, hypotheses, and the FQI reading rule
   are carried over from v2 §B/§C verbatim. The only deliberate changes are the
   seed values and the output directory.
2. **The amended H1′ stays as written in v2 §C**, including its allowance for
   saturation or decline at high knob values, even though that allowance was
   informed by pilots run in the superseded environment. It is not re-widened
   after seeing new results.
3. **Both outcomes are reported.** If the new results contradict v2 §G — in
   particular if the delayed-reward or coupling effects vanish, or if fatigue
   becomes non-null — that is reported as the confirmatory finding, with the old
   numbers alongside for comparison.

Any comparison between the superseded and the new environment is **descriptive
and exploratory**. No hypothesis is stated about the direction or size of the
difference between them.

---

## D. New confirmatory run (fixed)

**Fresh seeds:** `{10, 11, 12, 13, 14}` — disjoint from the v1 §A pilot `{0,1,2}`,
the first full sweep `{0,1,2,3,4}`, and the v2 confirmatory and ablation runs
`{5,6,7,8,9}`. Evaluation remains paired per policy via
`eval_seed_base = 10000`.

**Hyperparameters:** unchanged from v2 §B in every entry (dataset 6000, eval
5000, `t_max` 20, PPO online, 40 iterations, 256 rollout sessions, 4 epochs /
256 minibatch, hidden (128, 64), γ/λ 0.95/0.95, lr 3e-4, clip/entropy/value/
grad-clip 0.2/0.01/0.5/0.5, FQI 30 iterations / 0.15 penalty, Bayesian-smoothed
greedy bandit with `PRIOR_COUNT=5`, `PRIOR_MEAN=0`).

**Grid:** unchanged from v1 §4. `fatigue_rate` ∈ {0, 0.1, 0.2, 0.4, 0.6};
`delayed_reward_strength` ∈ {0, 0.25, 0.5, 1.0, 2.0};
`transition_coupling_strength` ∈ {0, 0.5, 1.0, 2.0, 3.0}. The `t_max` axis
remains descriptive-only.

**Code state:** commit `33378f3` or a later commit that does not further modify
`CustomerSimulation/simulation/`, `SharedSchema/shared_schema/features.py`,
`SharedSchema/shared_schema/constants.py`, or the trainers. The run manifest
records the effective fingerprint; if it differs from the fingerprint of the
first completed cell, the run restarts in a new directory.

**Command:**

```
python Experiments/run_sequentiality_sweep.py \
  --out-dir Experiments/sweep_results_confirmatory_v3 \
  --seeds 10 11 12 13 14 \
  --dataset-sessions 6000 --eval-sessions 5000 \
  --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30
```

---

## E. Analysis plan

Unchanged from v2 §C and reproduced here so that v3 is self-contained:

- **Anchor gap is a first-class reported quantity.** Per axis, the all-knobs-0
  PPO−bandit gap with its 95% CI.
- **Primary contrast:** `Δ(v) = gap(knob = v) − gap(knob = 0)` per axis, mean ±
  95% CI across seeds, computed within-seed because seeds are paired across
  cells.
- **H0′:** at all knobs = 0 the unadjusted gap may be positive; it is reported
  and attributed to the training-regime asymmetry, not to sequential structure.
- **H1′:** `Δ(v) > 0` above 0, growing over the low range of each knob;
  saturation or decline at high values is permitted.
- **H2:** there is a threshold above which `Δ(v)` is significant (95% CI
  excludes 0).
- **FQI reading rule (v1 §8):** if FQI tracks PPO, the win is sequentiality; if
  only PPO wins, the conclusion is that deep function approximation and/or
  online training are necessary in addition to sequential credit assignment.
- Confidence intervals use pointwise Student-t critical values across seeds
  (v2 §F.3).

---

## F. Follow-ups to re-run

Both v2 §D items completed in the superseded environment and are equally
invalidated. Each re-runs in its own new directory, after the §D confirmatory
run, on the same fresh seeds `{10, 11, 12, 13, 14}` and at the v2 §B
hyperparameters.

**1. Expose-history robustness** — `fatigue_rate` axis only, 25 cells, matching
the executed `sweep_results_expose_history` run:

```
python Experiments/run_sequentiality_sweep.py \
  --out-dir Experiments/sweep_results_expose_history_v3 \
  --seeds 10 11 12 13 14 --axes fatigue_rate --expose-history-to-bandit \
  --dataset-sessions 6000 --eval-sessions 5000 \
  --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30
```

**2. Online/offline ablation** — full three-axis grid, 75 cells. Note the
discrepancy being corrected here: v2 §D.2 declared "the anchor and one elevated
point per mechanism axis", but the run actually executed and reported in v2 §H
covered all three complete axes. The larger executed scope is adopted as the
declared scope so that the record and the specification agree:

```
python Experiments/run_sequentiality_sweep.py \
  --out-dir Experiments/sweep_results_offline_ppo_v3 \
  --seeds 10 11 12 13 14 \
  --axes fatigue_rate delayed_reward_strength transition_coupling_strength \
  --ppo-mode offline \
  --dataset-sessions 6000 --eval-sessions 5000 \
  --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30
```

Both remain **exploratory robustness analyses**, not confirmatory endpoints,
exactly as under v2. Their prior records (v2 §G paragraph 2, v2 §H) become
historical on the same terms as the main run.

---

## G. Status of the superseded records

`sweep_results_confirmatory/` (fingerprint
`5ee55664e82e07895eb02e65215d2e66e9abde759b1894e8a194e6f005940298`),
`sweep_results_expose_history/`, and `sweep_results_offline_ppo/` (fingerprint
`906380ce9d5ba93ae1ae62b563620ab5354d70d9993e871189fbf60b31613e43`) are
**retained, not deleted**, on the same basis as the v1 pilots: they are a
disclosed record of a completed run in a superseded environment.

They may be cited in the thesis only as historical context, explicitly labelled
as pre-`33378f3`. They may not be pooled with, averaged into, or substituted for
the §D results, and no figure may combine cells from both environments.

---

## H. Post-result record (2026-08-05; not a preregistration amendment)

All three runs were executed on 2026-08-05 at commit `98701f7`, which satisfies
the §D code-state condition: no commit after `33378f3` modifies
`CustomerSimulation/simulation/`, `SharedSchema/shared_schema/`, the trainers,
or the sweep runner, and the working tree was clean at those paths. The runtime
matches the one recorded in the superseded manifests (Python 3.14.2,
torch 2.12.0+cpu). The §C commitments held: only the seed values and the output
directories differ from v2 §B. Intervals throughout are the predeclared
pointwise Student-t 95% half-widths across the five seeds, and "excludes zero"
is the §E H2 significance rule.

### H.1 §D confirmatory run

The §D command completed all 95 expected cells on fresh seeds 10–14 with zero
failures in 2788 s. `sweep_results_confirmatory_v3/run_status.json` records
`final_outputs_complete: true` and fingerprint
`d4c312393ff2df6af2cabe42808eb9ec116a550a8287effbdd3016ad7b049bf3`.

- All-mechanisms-off raw PPO−bandit anchor: **+0.141 ± 0.052**.
- Delayed reward 0.25: raw gap +0.494 ± 0.046; anchor-adjusted
  **Δ=+0.353 ± 0.053**. Δ then declines but every interval on the axis excludes
  zero: **+0.225 ± 0.048** at 0.5, **+0.117 ± 0.109** at 1.0, and
  **+0.120 ± 0.110** at 2.0. The decline above 0.25 is the saturation H1′
  already permits.
- Transition coupling 0.5, 1.0, 2.0, and 3.0: respectively
  **Δ=+0.216 ± 0.121**, **+0.434 ± 0.174**, **+0.291 ± 0.190**, and
  **+0.250 ± 0.133**. All four exclude zero; the maximum is at 1.0.
- Fatigue is not supported. Δ=−0.051 ± 0.028 at 0.1 excludes zero **in the
  negative direction**; at 0.2 it is −0.100 ± 0.112, and at 0.4 and 0.6 the
  means turn weakly positive (+0.052 ± 0.076 and +0.059 ± 0.084) with intervals
  covering zero. No fatigue setting supports H1′.
- `t_max` remains descriptive-null: Δ=−0.001 ± 0.041 from 30 onward. The 30, 40,
  and 60 cells are identical to ten decimal places (bandit 1.4989287772, PPO
  1.6390650473, FQI 1.4221920122), so no session in the grid runs beyond 30
  steps and the horizon does not bind above that value. Only `t_max`=20 differs
  from them, by about 0.001 in Δ and 0.007 in the bandit mean, so the horizon
  binds marginally at 20 and not at all thereafter.
- FQI does not track PPO. The anchor FQI−bandit gap is −0.083 ± 0.055. On the
  delayed-reward axis Δ_FQI stays within [−0.016, +0.028] with every interval
  covering zero, and on the coupling axis it is substantially negative and
  excludes zero at every value: −0.345 ± 0.122, −0.702 ± 0.171, −0.799 ± 0.126,
  and −0.744 ± 0.303. Under the §E reading rule, only PPO wins, so sequential
  credit assignment alone is **not** identified as the cause of the advantage;
  deep function approximation and/or online training are necessary in addition.

### H.2 §F.1 expose-history robustness (exploratory)

Completed 25/25 cells with zero failures in 630 s.
`sweep_results_expose_history_v3/run_status.json` records
`final_outputs_complete: true` and fingerprint
`6d7aabe536f51bf69206da22c786293aab16fa30ccc25644142a4eaea0c8cf12`.

The anchor is +0.159 ± 0.101. Anchor-adjusted fatigue changes are
−0.074 ± 0.099 at 0.1, −0.096 ± 0.050 at 0.2 (excluding zero, negative),
+0.061 ± 0.158 at 0.4, and +0.088 ± 0.102 at 0.6. Exposing the intervention
history to the bandit does not produce a positive fatigue effect, so withholding
that history is not what created the PPO advantage. This reinforces the §H.1
fatigue result rather than qualifying it.

### H.3 §F.2 offline-PPO ablation (exploratory)

Completed 75/75 cells with zero failures in 3524 s.
`sweep_results_offline_ppo_v3/run_status.json` records
`final_outputs_complete: true` and fingerprint
`2f70cdc5672b90fab9b97e0bd0aff00725438d158e6fb2c0b3270240c6371956`.

The all-mechanisms-off gap is +0.027 ± 0.098, an interval covering zero, so the
positive online-PPO anchor is removed. Delayed reward retains anchor-adjusted
changes of +0.251 ± 0.082 at 0.25, +0.180 ± 0.149 at 0.5, +0.152 ± 0.106 at 1.0,
and +0.144 ± 0.118 at 2.0, all excluding zero. Transition coupling retains
+0.140 ± 0.171 at 0.5 (covering zero), +0.333 ± 0.319 at 1.0, +0.171 ± 0.144 at
2.0, and +0.187 ± 0.118 at 3.0. Because `--ppo-mode offline` changes only the
PPO arm, the bandit and FQI arms are identical to §H.1 at the same seeds, which
the shared FQI−bandit anchor of −0.083 ± 0.055 confirms.

Fatigue is again unsupported, with one exception recorded rather than smoothed
over: Δ=+0.053 ± 0.044 at 0.4 excludes zero. It is the only positive fatigue
interval excluding zero anywhere in v3, it is small, it carries the narrowest
interval on that axis, and it appears in an exploratory ablation rather than the
confirmatory run. It is reported as an isolated positive point, not as support
for the fatigue hypothesis, and no claim is built on it.

As in v2 §H, this ablation shows that fresh online simulator interaction is not
necessary for the delayed-reward and moderate-coupling effects in this setup. It
does not identify sequential credit assignment alone: the fixed-dataset PPO
routine is not a validated general-purpose offline-RL estimator, and state
access, function approximation, objective, and optimization still differ from
the contextual bandit.

### H.4 Descriptive comparison against the superseded §G record

Per §C this comparison is descriptive and exploratory; no hypothesis was stated
about its direction or size, and §C.3 required reporting the outcome either way.
The new results **do not contradict** §G. Direction, shape, and the identity of
the winning axes all reproduce in the corrected environment.

- The anchor moves from +0.172 ± 0.112 to +0.141 ± 0.052 — the same positive
  training-regime asymmetry, with an interval less than half as wide.
- Delayed reward at 0.25 is essentially unchanged (+0.353 ± 0.053 against
  +0.344 ± 0.080). The substantive difference is at 0.5, 1.0 and 2.0, where the
  intervals now exclude zero and previously did not, so the axis is supported
  across its whole range rather than at a single point.
- Coupling preserves its rise-to-1.0-then-decline shape, with a slightly lower
  maximum (+0.434 ± 0.174 against +0.470 ± 0.095). The 3.0 cell now excludes
  zero where it did not before.
- Fatigue is unsupported in both environments, but the sign pattern differs. v2
  reported a non-positive mean at every nonzero setting; v3 has a significantly
  negative point at 0.1 and weakly positive, non-significant means at 0.4 and
  0.6.
- The FQI reading rule returns the same verdict more strongly. Δ_FQI under
  coupling roughly doubles in magnitude, from the −0.216…−0.348 range in §G to
  −0.345…−0.799 here.

The one change that motivated v3 — change 5, which let `primed_credit`
accumulate in the all-knobs-zero configuration — therefore tightened the anchor
without reversing any conclusion drawn from it.

---

## I. Amendment: myopic-objective ablation (2026-08-06)

**Status: exploratory. This section is an amendment written after §H and after
the confirmatory results were known.** It states no hypothesis, tests none, and
its cells are never pooled with the §D grid. It is recorded here so that the
analysis reported in `Experiments.tex` §`subsec:SweepMyopic` has a documented
provenance, not to claim confirmatory standing it does not have.

### I.1 Why it was run

§H.1 concludes under the §E reading rule that sequential credit assignment alone
is not identified as the cause of the advantage, because only PPO wins. §H.2 and
§H.3 remove history exposure and online data collection as candidate
explanations, but neither varies the **optimization horizon** itself. The
confirmatory design cannot: the two arms differ in objective, model class and
state access simultaneously. Setting the discount to zero varies the horizon
directly while holding everything else fixed, which is the one contrast the grid
never contains.

### I.2 Change set relative to §D

Only the following differ from the §D command. Everything else — archetypes,
dataset and evaluation sessions, PPO iterations and rollout sessions, FQI
iterations, seeds 10–14, and every recorded `implementation_sha256` — is
identical, verified by diffing the two `run_manifest.json` files.

1. `--gamma 0` (was 0.95).
2. `--gae-lambda 0` (was 0.95).
3. `--axes fatigue_rate delayed_reward_strength transition_coupling_strength`,
   omitting the descriptive `t_max` axis. §H.1 established `t_max` is inactive
   above 30, and it is meaningless under a zero discount.
4. `--out-dir Experiments/sweep_results_myopic_v3`.

The runner supplies one discount to **both** learners, so the tabular arm is
myopic in this run as well. This is not worked around: a myopic FQI is a
one-step regressor over the discretised state and is not the quantity of
interest. Only Δ_PPO is read from this run.

### I.3 Disclosure

A two-cell smoke run (`--smoke`, seed 0, throwaway output directory) preceded the
full run to verify the pipeline executes under the changed configuration. Its
output was observed and is not used in, or reported as, any quantity here. The
change set above was fixed before the full run started, the §E analysis method
was carried over verbatim (within-seed contrast, mean ± Student-t 95% interval on
five seeds), and no result was retuned after inspection.

### I.4 Result

Completed 75/75 cells with zero failures.
`sweep_results_myopic_v3/run_status.json` records `final_outputs_complete: true`
and fingerprint
`293941b789a4350fc073507874b4a3ae05eb24b66e78c378db9685f89c405ea0`.

The anchor moves from +0.141 ± 0.052 to **+0.184 ± 0.023**, so the residual
system asymmetry is not an artefact of discounting.

- **Delayed reward inverts.** Δ_PPO is −0.347 ± 0.062 at 0.25, −0.509 ± 0.088 at
  0.5, −0.586 ± 0.085 at 1.0, and −0.583 ± 0.080 at 2.0. All four exclude zero in
  the negative direction. The swing against §H.1 is near-constant at ≈ −0.70.
- **Transition coupling is unchanged.** Δ_PPO is +0.176 ± 0.074, +0.412 ± 0.152,
  +0.262 ± 0.172, and +0.257 ± 0.129 at 0.5, 1.0, 2.0 and 3.0. Every value lies
  inside the interval of its §H.1 counterpart; differences range from −0.039 to
  +0.007.
- **Fatigue remains unsupported** under both objectives, a consistency check on
  the manipulation.

### I.5 Reading

Delayed assist credit is a horizon effect: nothing but the objective changed, and
the advantage inverted. Transition coupling is not: the advantage survives the
complete removal of lookahead, leaving function-approximation capacity as the
explanation, consistent with the direction §H.3 already pointed. Combined with
§H.1 — FQI has the objective and gains nothing — the horizon is **necessary but
not sufficient** for exploiting delayed structure.

Not explained: why the myopic policy falls *below* the bandit under delayed
credit rather than converging toward it. Stable across four knob values and five
seeds, so not noise, but no policy-behaviour diagnostic was run and no mechanism
is asserted.

### I.6 Standing

`sweep_results_myopic_v3/` is an exploratory ablation directory on the same terms
as `sweep_results_expose_history_v3/` and `sweep_results_offline_ppo_v3/`. Its
cells may not be pooled with, or substituted for, the §D confirmatory grid, and
no figure may combine cells from both.

## J. Amendment: multiplicity control (2026-08-08)

### J.1 What prompted it

External review of the 2026-08-07 build observed that multiplicity is
acknowledged in the thesis but not corrected: §E fixes a per-cell decision rule
(interval excludes zero) and the confirmatory grid applies it twenty-four times
— twelve mechanism cells for each of PPO and FQI in the Δ table — with no
family-level control. The objection is correct. Disclosure is not a correction.

### J.2 Status: post hoc, additive, non-replacing

**This amendment is post hoc and is not presented otherwise.** It was specified
after the confirmatory results were known. It therefore does *not* retroactively
replace the §E decision rule, which remains the declared primary analysis and
continues to determine which cells are set in bold. The adjustments below are
reported alongside it. Where the two disagree, the disagreement is reported as
the finding rather than resolved in favour of whichever is more convenient.

### J.3 The two adjustments

**Per cell.** For each of the twenty-four confirmatory cells, the per-seed
anchor-adjusted deltas give a one-sample t statistic on df = n−1 and hence a
two-sided p-value. Benjamini–Hochberg at q = 0.05 across the twenty-four
controls the false discovery rate. BH rather than Bonferroni: the cells share
seeds, share the anchor, and lie on monotone knob grids, which is the positively
dependent regime BH targets and where Bonferroni is needlessly conservative.

**Per axis.** The within-seed mean of Δ(v) over an axis's non-zero settings
gives one value per seed, hence one interval per (arm, axis): six tests, Holm
controlled. This matches the level at which the thesis argues its mechanism
claims — by the shape of an axis rather than by any single cell — and is the
analysis the Evaluation chapter now leads with.

The descriptive `t_max` axis is excluded from both families: it tests no
hypothesis and carries no claim.

### J.4 Result on the §D confirmatory grid

Of the fifteen cells the §E rule accepts, **twelve survive BH**. The three that
fall are the delayed-credit cells at σ = 1.0 and σ = 2.0 (lower endpoints +0.008
and +0.010, already flagged in the thesis as cells nothing should rest on) and
the tabular fatigue cell at ρ = 0.2 (the isolated positive on an axis where
nothing else clears). No cell that carries an argument is removed.

At axis level, both supported mechanisms survive family-wise correction:
delayed credit +0.204 (p_Holm = 0.0048) and transition coupling +0.298
(p_Holm = 0.0009) for PPO. The fatigue null is unchanged (p_Holm = 1.000). For
FQI, the coupling deterioration survives (−0.648, p_Holm = 0.0002) and the
delayed-credit contrast does not clear (p_Holm = 1.000).

### J.5 Implementation

`Experiments/compute_multiplicity_control.py`. It reproduces every published
per-cell mean and interval from the run's own `summary.json` before adjusting
anything, using the same three-decimal t-table constants the sweep runner uses
so that the reproduced decisions are bit-faithful to the published ones rather
than merely close. The Student-t tail is implemented in-file rather than
imported, because `Experiments/requirements.txt` does not declare scipy; the
implementation is validated in `--self-check` against the Cauchy and df = 2
closed forms and against the critical values the sweep hard-codes.

Outputs are written beside the run: `multiplicity_control.csv`,
`multiplicity_control.json`, `multiplicity_table.tex`.

### J.6 Scope

The axis-level statistic answers a coarser question than the per-cell one: it
establishes that an axis carries a non-zero average effect, not that any
particular setting does. It does not license reading individual cells that BH
rejects.

## K. Amendment: 15-seed replication (2026-08-08)

### K.1 Standing: a replication, not a replacement

The §D confirmatory grid on seeds 10–14 **remains the confirmatory evidence of
this thesis**. This amendment declares an additional, higher-powered replication
on seeds 10–24. It does not supersede §D, its cells are not pooled with §D, and
no confirmatory claim is restated on its numbers.

The reason for that boundary is specific and is recorded rather than left
implicit. The replication was specified after the §D results were known, and it
changes the reading of one preregistered null (§K.4). Adopting it as the
confirmatory basis would mean changing the seed count after observing that doing
so converts a null into a positive result, which is the precise manoeuvre a
preregistration exists to prevent — regardless of the 15-seed estimate being the
better one. It is therefore reported alongside §D, in full, including the part
that is inconvenient.

### K.2 What was run

`Experiments/run_seed15_sweeps.sh`, all four grids on seeds 10–24, into
`sweep_results_{confirmatory,myopic,offline_ppo,expose_history}_v3_s15/`.
Completed 2026-08-08 05:42Z, 7 h 48 m, zero failed cells (285 + 225 + 225 + 75).
Every parameter other than the seed list is identical to §D and §F.

The myopic grid was extended alongside the confirmatory one because `D(v)` and
`D_FQI(v)` are within-seed paired contrasts: extending one run alone would leave
nothing to pair against.

### K.3 Validation

Seeds 10–24 are a deliberate superset of 10–14 rather than a fresh set, so the
overlap is a reproducibility test. **All 270 shared cells across the four grids
are bit-identical** to their §D and §F counterparts. The environment did not
drift between the runs and the two sets of numbers are directly comparable.

### K.4 What it shows

**The two supported mechanisms tighten and nothing load-bearing moves.** The
horizon contrast on delayed credit is −0.700 (was −0.710) for PPO and −0.096
(was −0.099) for FQI, so the ratio between what the horizon is worth under the
two representations is 7.3 against 7.2. Delayed credit at axis level is
+0.215 ± 0.067 (was +0.204 ± 0.069); coupling +0.341 ± 0.061 (was
+0.298 ± 0.062); the FQI coupling deterioration −0.567 ± 0.066 (was
−0.648 ± 0.083). The anchor falls from +0.141 ± 0.052 to +0.098 ± 0.057 and
still excludes zero.

Three qualifications that §D required are removed at higher power: the
delayed-credit cells at σ = 1.0 and σ = 2.0 are no longer marginal
(+0.140 ± 0.080 and +0.145 ± 0.074), the horizon contribution at κ = 3.0 is
bounded at ±0.050 where §D could not bound it at all (±0.159), and all four
`D_FQI` delayed-credit cells exclude zero where three did narrowly. Under the
§J multiplicity correction, **all eighteen cells that clear the per-cell rule
also survive BH** — no decision changes, where three of fifteen fell at §D.

**The fatigue null does not survive the additional power, and this is the
finding of the replication.** In §D no fatigue setting produced an advantage the
decision rule would accept, and ρ = 0.1 favoured the bandit. At fifteen seeds:

| Cell | §D (5 seeds) | §K (15 seeds) |
|---|---|---|
| PPO ρ = 0.1 | −0.051 ± 0.028 (sig.) | −0.035 ± 0.061 (n.s.) |
| PPO ρ = 0.2 | −0.100 ± 0.112 | +0.005 ± 0.081 |
| PPO ρ = 0.4 | +0.052 ± 0.076 | **+0.114 ± 0.070** |
| PPO ρ = 0.6 | +0.059 ± 0.084 | **+0.117 ± 0.073** |
| FQI axis | +0.078 ± 0.078 | **+0.096 ± 0.048** (p_Holm = 0.002) |

The §F.1 history-exposure ablation moves the same way: ρ = 0.4 (+0.089 ± 0.075)
and ρ = 0.6 (+0.136 ± 0.074) are positive and exclude zero *with the exposure
count supplied to the bandit*, where at five seeds only ρ = 0.2 cleared and did
so negatively.

The coherent reading is that intervention fatigue does confer a sequential
advantage where the mechanism is strong enough to bite — the upper half of the
grid — and that five seeds could not resolve it. The PPO fatigue *axis* mean
remains non-significant (+0.050 ± 0.064, p_Holm = 0.235), so this is a
within-axis pattern concentrated at high ρ rather than a uniform axis effect.

### K.5 Consequence for the thesis's claims

The §D null is retained as the confirmatory result and is **relabelled**: it is
evidence that no fatigue effect was resolvable at five seeds, not evidence that
none exists. Every statement of the form "fatigue produces no sequential
advantage" is qualified accordingly wherever it appears, and the framing "two of
the three mechanisms produce an advantage" is scoped to the confirmatory grid
with the replication named alongside it.

No other confirmatory claim changes.

### K.6 Standing of the directories

The `_v3_s15` directories are declared replication artefacts. Their cells may not
be pooled with or substituted for the §D grid, and no figure may combine cells
from both, on the same terms as §I.6.
