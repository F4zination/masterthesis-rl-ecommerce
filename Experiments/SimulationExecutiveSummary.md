# Simulation Executive Summary

*Prepared for the supervisor meeting on 2026-07-16 and updated on 2026-07-21. Covers the shared
customer-session simulator, the three sequential-dynamics knobs, the experimental
design built on them, and the current state of results.*

Naming note: the thesis contrasts **CB** (the contextual-bandit system,
`DemoSiteV2`) with **SRL**, implemented here with PPO (`DemoSiteV3`). Tabular
FQI is an additional sequential baseline. Below, PPO refers to the tested SRL
learner.

---

## 1. The research question the simulation addresses

> **When is full sequential RL (tested here as PPO) worth it over a contextual bandit (CB)?**

Early experiments showed CB ≈ PPO. This is consistent with the original
environment being dominated by local, immediate effects: its funnel-stage
transition weights did not depend on the selected action, and it contained no
history-dependent fatigue or delayed assist mechanism. These properties make a
myopic policy a strong baseline, although they do not prove exact equivalence
between the implemented CB and PPO systems.

The experiment therefore varies the amount and type of sequential structure in
the simulator and measures the PPO-minus-CB performance curve. The June 25
results remain exploratory. The fresh-seed confirmatory run is now complete:
it supports a PPO advantage from low delayed reward and from low-to-moderate
transition coupling, but it does not support a fatigue-driven advantage.

---

## 2. The simulator in one paragraph

`CustomerSimulation` generates e-commerce sessions as transition trajectories.
A session starts at `landing` with randomly sampled device, traffic source, and
price attributes, then moves stochastically among `landing`, `scroll_engagement`,
`pdp`, `cart`, `checkout`, and `done`, including loops, returns to PDP, and early
exits.
At every decision the policy selects one of six actions: five widgets
(`frequently_bought_together`, `discount_banner`, `trust_badge`, `help_popup`,
`trending_carousel`) or `no-op`. Customer archetypes define stage transitions,
base-event probabilities, widget click/dismiss probabilities, action-event
lifts, and order-total ranges in `CustomerSimulation/config/archetypes.yaml`.
Rewards use the shared scoring function: a purchase earns 8 plus an order-total
bonus capped at 4, and every non-no-op action incurs `ACTION_COST`. Dataset
generation, PPO rollouts, and evaluation use the same simulator dynamics, so
there is no environment mismatch between policies.

## 3. Why the original simulator produced a near-tie

Four properties explain why a myopic bandit was competitive in the original
dynamics:

1. **No deliberately delayed assist payoff** — rewards were recorded on the
   current transition, including purchase at checkout.
2. **Funnel-stage weights were action-independent** — actions could affect
   emitted state through events such as add-to-cart, but could not directly
   steer the next funnel stage.
3. **No history-dependent fatigue/saturation** — response probabilities did not
   deteriorate with prior widget exposure. Immediate action costs already
   existed, however.
4. **No intervention-history observation** — the state already summarized page
   depth and cart contents, but not widgets previously shown or priming credit.

These conditions motivate an empirical near-tie anchor; they are not a proof
that the implemented policies must have identical return.

---

## 4. The three knobs (the core of the contribution)

The simulator introduces three qualitative, literature-motivated mechanisms,
each controlled by one scalar strength. The exact functional forms and values
are synthetic modeling choices rather than empirically calibrated consumer
parameters.

- All three mechanism strengths default to zero. The resulting anchor is the
  current, post-cleanup simulator with all mechanisms disabled; it is **not**
  byte-identical to the pre-sequential legacy implementation because
  action-event-lift validation and emitted state were also revised.
- The event reward function and action costs are unchanged across knob values;
  differences in return arise from changed behavior and transitions.
- Shared constants and the YAML configuration currently agree on priming
  actions, amount, and decay, although those values exist in both places and
  therefore are not a strict single source of truth.

### Knob 1 — `fatigue_rate`: intervention fatigue

*Literature basis: ad/banner fatigue, banner blindness, habituation.*

The state tracks `interventions_shown` (cumulative non-no-op actions) and
optionally an exponential-window count (`fatigue_window` > 0 →
`recent_interventions`). With exposure count `k`:

```
decay      = exp(-fatigue_rate · k)
p_click'   = p_click · decay                                  # response fades
p_dismiss' = p_dismiss + (p_dismiss_max − p_dismiss)(1 − decay)  # annoyance rises toward 0.6
lift'      = lift · decay                                     # action lifts fade too
```

**Sequential effect:** serving a widget imposes a delayed negative externality
on later widgets. A policy can use exposure and spacing summaries to ration or
defer interventions. With a nonzero fatigue window, however, the exact
`recent_interventions` value remains latent, so the emitted features only
partially observe the windowed dynamics. The sweep uses values from 0 to 0.6.

### Knob 2 — `delayed_reward_strength`: delayed-assist reward

*Literature basis: multi-touch attribution, assist conversions, delayed reward.*

Two designated **priming actions** (`help_popup`, `trust_badge`) accrue a
latent `primed_credit` (+1.0 per serve, decaying ×0.9 per step). At the
**cash-out point** (`checkout`), the purchase *probability* is boosted by
`delayed_reward_strength · primed_credit`, clamped to [0, 1]. The purchase
reward itself is unchanged; the mechanism changes whether a purchase occurs.

**Sequential effect:** the conversion reward arrives at checkout but is caused
by earlier priming actions. The logged bandit assigns transition reward to the
checkout action, whereas a γ-discounted return can propagate value backward.
The swept strength range is 0 to 2.0.

### Knob 3 — `transition_coupling_strength`: action-dependent transitions

*Literature basis: funnel/journey progression, nudges advancing shoppers.*

Each archetype defines `action_stage_lifts` — e.g., for FastBuyer,
`frequently_bought_together: {cart: +0.55, done: −0.20}` (pushes toward cart,
retains the user), `trust_badge: {checkout: +0.45, done: −0.15}`. The
next-stage sampling weights become:

```
w[next] = max(0, base_w[next] + strength · lift[action][next])   # then renormalized
```

**Sequential effect:** an action's value now includes the future stages it
steers toward. The bandit estimates immediate reward, while PPO and FQI optimize
a discounted return. The swept strength range is 0 to 3.0.

### Descriptive axis — `t_max` (baseline horizon sensitivity)

`t_max` was swept over 20/30/40/60 with all three mechanisms off. It therefore
tests baseline truncation only, not whether horizon amplifies fatigue, priming,
or coupling; those questions require mechanism × horizon interactions. Sessions
usually terminate naturally before step 30. Evaluated bandit and FQI returns are
identical from 30 onward, but PPO returns vary without a clear trend.

---

## 5. What is and is not matched

The evaluation environment, reward function, action space, number of evaluation
sessions, and evaluation seed are shared. The training comparison is a
**system-level comparison**, not a one-variable controlled treatment:

- The **CB bandit** keys its context on `POINT_FEATURES[decision_point]` only
  (device, traffic source, bucketed page depth / cart total / price / item
  count). It ignores any other key in the state dict.
- **PPO and FQI encode every key** in the state dict. PPO is feed-forward rather
  than recurrent; its temporal information comes from explicit summaries.
- With `sequential=True`, `SimState.to_state_dict` appends four bucketed
  **history features** (`HISTORY_FEATURES`): `interventions_shown_bucket`,
  `steps_since_widget_bucket`, `session_step_bucket`, `primed_credit_bucket` —
  deliberately **not** added to `POINT_FEATURES`.

CB and FQI train from the 6,000-session uniform-behavior dataset. Online PPO
uses that dataset to construct its state/action vocabularies, then learns from
40 × 256 = **10,240 fresh on-policy simulator sessions per cell**. PPO therefore
also differs in online data access, training volume, function approximation,
objective, and optimization.

The planned bandit-history robustness check exposes only intervention count and
applies only to fatigue; it does not remove every information asymmetry.
Primed-credit serving semantics still require reconciliation.

## 6. Safeguards against "tuning until RL wins"

The design has three useful safeguards, with one important disclosure:

1. **Literature motivation:** repeated-exposure wearout, multi-touch conversion
   attribution, and long-term recommendation effects are established concepts.
   The simulator operationalizes them synthetically; it does not claim that the
   chosen equations or parameter values are behaviorally calibrated.
2. **Transparent preregistration timeline:** v1 was committed before the full
   four-axis sweep, but about seven minutes **after** a three-seed coupling pilot.
   V2 discloses this, classifies all results for seeds 0–4 as exploratory, and
   defines a confirmatory run on fresh seeds 5–9.
3. **Spectrum reporting:** the full curve is reported, including near-tie,
   saturation, and decline regimes rather than only the best grid point.

The comparison includes CB, tabular FQI, and deep PPO. If both sequential
learners improve, that supports a sequentiality explanation. If only PPO
improves, the result cannot distinguish among online data access, data volume,
function approximation, optimization, or limitations of FQI. In particular,
this FQI is not support-constrained: eligible but unobserved actions retain an
implicit Q-value of zero and may outrank observed actions with negative values.
Its failure is therefore non-diagnostic. The completed offline-PPO run is a
fixed-dataset ablation, not an importance-weighted trainer; it removes fresh
online interaction but does not equalize state access, model class, objective,
or optimization.

---

## 7. The experiment: sequentiality sweep

Under the v2 confirmatory configuration, which matches the actual June 25
exploratory settings, each grid cell (one knob at a time, with the others off)
and seed runs through this pipeline:

1. Generate 6,000 sessions with the uniform behavior policy under that cell's
   dynamics (`sequential=True`).
2. Build the CB bandit using the same aggregation as the production
   `build_bandit_policy.py` script.
3. Train tabular FQI for 30 iterations. Use the same rows to construct PPO's
   vocabulary, then train PPO from 40 × 256 online rollouts.
4. Evaluate CB, FQI, PPO, uniform, and no-op policies on 5,000 fresh sessions
   through the same simulator. Each policy receives the same evaluation seed;
   this pairs runs by seed, not individual trajectories, because actions change
   RNG consumption and session paths.

The metric is **mean cost-adjusted reward per session**. The v2 primary contrast
is the within-seed **anchor-adjusted gap**
`Δ(v) = [PPO − CB](v) − [PPO − CB](0)`; the raw PPO − CB gap is secondary. The
repaired confirmatory runner reports pointwise Student-t 95% intervals over
five seeds, with no multiplicity adjustment. The stored June 25 run predates
automatic delta output and used normal-approximation intervals.

## 8. Results — and the honest caveats

**Confirmatory full sweep (2026-07-21, fresh seeds 5–9, 95/95 cells,
zero failures):**

All values below are means with the half-width of a pointwise two-sided 95%
Student-t interval. The anchor is the repeated all-mechanisms-off online-PPO
training cell: raw PPO−CB = **+0.172 ± 0.112**. It is already positive, so the
mechanism result is the within-seed anchor-adjusted change, not the raw gap.

| Axis | Selected raw PPO−CB gap | Anchor-adjusted Δ | Confirmatory reading |
|---|---:|---:|---|
| `delayed_reward_strength=0.25` | +0.516 ± 0.076 | **+0.344 ± 0.080** | positive low-strength effect; later values decline and their Δ intervals include zero |
| `transition_coupling_strength=0.5` | +0.406 ± 0.100 | **+0.234 ± 0.180** | positive effect begins at 0.5 |
| `transition_coupling_strength=1.0` | +0.642 ± 0.113 | **+0.470 ± 0.095** | strongest precise confirmatory effect |
| `transition_coupling_strength=2.0` | +0.624 ± 0.107 | **+0.452 ± 0.193** | positive, with saturation |
| `fatigue_rate=0.4` | +0.129 ± 0.083 | −0.043 ± 0.192 | no fatigue-driven improvement over anchor |
| `fatigue_rate=0.6` | +0.111 ± 0.063 | −0.061 ± 0.169 | no fatigue-driven improvement over anchor |
| `t_max=30/40/60` | approximately +0.168 ± 0.048 | approximately −0.004 ± 0.070 | no horizon signal; sessions naturally terminate |

The full curves matter. Delayed reward peaks at 0.25: its anchor-adjusted Δ is
+0.121 ± 0.239 at 0.5 and about +0.072 ± 0.151 at 1.0/2.0. Transition
coupling peaks around 1–2 and declines to +0.285 ± 0.316 at 3.0. Every
fatigue Δ is non-positive. These data confirm two mechanisms, not all three.

FQI does not reproduce the PPO gains. At the anchor FQI−CB is −0.091 ± 0.057;
under transition coupling it ranges from −0.356 to −0.439 at strengths
0.5–2.0. On its own, the main sweep therefore cannot attribute the PPO result
to sequential credit assignment.

The predeclared history-exposed-bandit fatigue robustness run is also complete
(25/25 cells, zero failures). Its high-fatigue anchor-adjusted PPO−CB changes
are +0.013 ± 0.099 at 0.4 and +0.014 ± 0.084 at 0.6. This agrees with the main
null fatigue result; it does not show that withholding history caused a PPO
advantage.

The fixed-dataset PPO ablation is complete (75/75 cells, zero failures). Its
all-off PPO−CB anchor is **−0.017 ± 0.037**, so the positive online-PPO anchor
disappears. Nevertheless, its anchor-adjusted effects remain close to the
online result: delayed reward gives +0.296 ± 0.115 at 0.25 and +0.252 ± 0.074
at 0.5; transition coupling gives +0.228 ± 0.130 at 0.5, +0.512 ± 0.250 at
1.0, and +0.435 ± 0.141 at 2.0. Fatigue remains null. Fresh online simulator
interaction is therefore not necessary for the delayed-reward and coupling
effects in this setup. The ablation narrows, but does not close, attribution:
explicit state/history access, deep function approximation, the PPO objective,
and optimization still differ from the bandit, and the fixed-dataset PPO
implementation is not a general-purpose offline-RL estimator.

**Exploratory full sweep (2026-06-25, seeds 0–4,
`sweep_results_thesis/`):**

For the three mechanism axes, the table shows the selected exploratory peak;
the horizon row summarizes all three non-anchor settings.

| Axis | Raw PPO−CB gap at anchor | Raw exploratory finding | Anchor-adjusted finding | Reading |
|---|---:|---:|---:|---|
| `delayed_reward_strength` | +0.12 ± 0.05 | **+0.49 ± 0.04** at 0.25 | **+0.370 ± 0.087** | strong, non-monotone exploratory signal |
| `transition_coupling_strength` | +0.08 ± 0.08 | **+0.72 ± 0.09** at 2.0 | **+0.642 ± 0.117** | strongest exploratory signal |
| `fatigue_rate` | +0.05 ± 0.09 | +0.14 ± 0.08 at 0.4 | +0.096 ± 0.090 | weak and statistically fragile |
| `t_max` | +0.09 ± 0.05 at 20 | no systematic trend | +0.011 / −0.018 / −0.060 at 30/40/60 | no baseline horizon signal |

The intervals above use the script's pointwise normal approximation. With only
five seeds, a Student-t interval is wider; in particular, the fatigue delta no
longer excludes zero. Peaks were also selected after viewing the exploratory
grid, so they should not be presented as confirmatory significance tests.
The repaired runner uses Student-t intervals for the completed fresh-seed run;
this post-interruption reporting correction is disclosed in
`preregistration_v2.md` §F.

Key caveats and remedies:

1. All four repeated all-off anchor estimates had positive means; their
   normal-approximation intervals excluded zero in two of four executions.
   Possible contributors are PPO's online interaction, larger data budget,
   function approximation, optimization, and extra history information. Anchor
   subtraction removes an additive
   baseline; it does not identify its cause.
2. FQI shows no robust positive advantage and is strongly negative under
   coupling. Given its implementation limitation above, these FQI results cannot
   isolate which difference between PPO and FQI caused the result.
3. The exploratory run deviated from v1 settings: PPO iterations 25→40,
   rollouts 128→256, dataset 4k→6k, evaluation 3k→5k, and FQI iterations 25→30.
   V2 discloses these deviations and fixes the confirmatory settings.
4. Seeded PPO shuffling and deterministic FQI tie-breaking are covered by a
   stored regression that compares all seed-level CSVs from two independent
   smoke-run processes byte-for-byte. This supports exact reproducibility for
   the tested path; the complete full-scale run has not been independently
   duplicated.

**Artifact status:** `sweep_results_confirmatory/run_status.json` records 95/95
cells complete, zero failures, and `final_outputs_complete: true`; the directory
contains seed-level tables, final gap/delta tables, plots, a summary, and an
input/code fingerprint. `sweep_results_expose_history/run_status.json` records
25/25 cells complete, and `sweep_results_offline_ppo/run_status.json` records
75/75 cells complete; both have zero failures. The old interrupted attempt is retained only as a
disclosed historical event in `preregistration_v2.md`; its partial plots are not
used as evidence.

---

## 9. Likely supervisor questions, with answers

- **"Didn't you just build an environment where RL wins?"** — The mechanisms
  are qualitatively motivated, their exact parameterizations are disclosed, all
  strengths have an off setting, and the entire curve is reported. However, the
  coupling pilot preceded v1 preregistration. The fresh-seed amendment supplies
  a completed confirmatory replication, but the parameterization is still
  synthetic and the system-level training asymmetries remain.
- **"Is the comparison fair?"** — Evaluation uses the same environment, reward,
  actions, session count, and seed. Training is intentionally not budget-matched:
  PPO learns online with more simulator interaction and a neural model, while
  CB and FQI are trained offline. Anchor subtraction and the completed
  fixed-dataset/history ablations quantify parts of this system-level
  asymmetry; they do not make every difference disappear.
- **"What did you find?"** — Fresh-seed results confirm anchor-adjusted PPO
  gains for low delayed reward and low-to-moderate transition coupling, reject
  a fatigue-driven gain, and show no baseline horizon effect. FQI does not
  reproduce the gains. Fixed-dataset PPO does reproduce delayed-reward and
  coupling effects while eliminating the positive anchor, so fresh online data
  is not necessary; state access, model class, objective, and optimization
  remain unresolved explanations.
- **"What's left?"** — Reconcile primed-credit serving semantics, train release policies with embedded
  provenance and complete production-token coverage, and obtain human evidence
  without treating the frozen A/B as a mechanism-identification test.
