# Pre-registration: When is full RL worth it over a contextual bandit?

**Status:** pre-registered **before** running the sweep. Commit this file (with a
git timestamp) prior to executing `run_sequentiality_sweep.py`. Everything below
is fixed in advance so the result is *measured*, not reverse-engineered.

**Research question.** In the shared intra-session funnel MDP, how large is the
V3-minus-V2 advantage (cumulative-session RL vs. myopic contextual bandit) as a
function of *how much sequential structure* the environment contains?

**Headline claim under test.** "V3 ≈ V2 when sequential structure is low (where a
realistic intra-session funnel sits), and V3 overtakes V2 once the structure
exceeds a threshold X." We report the whole curve — including the regime where
the bandit ties or wins.

---

## 1. Fixed environment dynamics (functional forms)

All three mechanisms live in the **shared** `CustomerSimulation` simulator and are
seen identically by every policy. Each has a single scalar knob that defaults to
**0/off**.

**Reward-model invariant (fixed).** An `action_event_lifts` entry may only
*modulate* an event that is actually possible at the current decision point
(i.e. already present in `base_events[dp]`); it may not manufacture an impossible
event. In particular a `checkout_submit`/`purchase` lift carried by
`trust_badge`/`discount_banner` does **not** fire at `landing`/`pdp`/`cart` —
only at the page where the event exists. (Without this gate, those widgets earn
a flat +2.0 "checkout submit" on every page, which collapses the learned policy
onto whichever cheap widget carries the lift, regardless of the priors.)
Funnel-progression effects of an intervention are modelled by Mechanism 3
(action-dependent transitions), not by immediate reward. With every knob off the
gated model reproduces the near-tie anchor H0.

### Mechanism 1 — Intervention fatigue
*Literature:* ad/banner fatigue, banner blindness, habituation, intervention
fatigue in consumer behaviour. With effective over-exposure count `k`
(cumulative `interventions_shown`, or an exponential window when
`fatigue_window > 0`) and `decay = exp(-fatigue_rate * k)`:

```
p_click'   = p_click   * decay
p_dismiss' = p_dismiss + (p_dismiss_max - p_dismiss) * (1 - decay)
lift'      = lift       * decay        # action_event_lifts also fade
```
Knob: `fatigue_rate` (off = 0). `p_dismiss_max = 0.6`.

### Mechanism 2 — Delayed / assist reward
*Literature:* multi-touch attribution, assist conversions, delayed reward in RL.
Priming actions `{help_popup, trust_badge}` accrue latent `primed_credit`
(`+priming_amount = 1.0` when served), decayed each step by
`delayed_reward_decay = 0.9`. At the cash-out point (`checkout`) the purchase
probability is lifted by `delayed_reward_strength * primed_credit` (clamped to
[0,1]). The reward **table is unchanged** — only *whether/when* the purchase
fires shifts, so the delayed payoff is endogenous. Knob:
`delayed_reward_strength` (off = 0).

*Note (pre-declared):* credit accrues when a priming action is **served** (not
conditioned on a click), so the mechanism carries enough signal to be learnable;
this is a deliberate, declared modelling choice.

### Mechanism 3 — Action-dependent transitions
*Literature:* funnel/journey progression, stage-advancing nudges, sequential
decision-making in marketing. Next-stage weights get additive per-archetype
`action_stage_lifts`, scaled by a global coupling and clamped ≥ 0:

```
w[next] = max(0, base_w[next] + transition_coupling_strength * action_stage_lifts[action].get(next, 0))
# renormalised by rand.choices
```
Lifts target next-stage keys already present in each point's `stage_transitions`.
The per-archetype `action_stage_lifts` values are fixed in
`CustomerSimulation/config/archetypes.yaml` (committed alongside this file).
Knob: `transition_coupling_strength` (off = 0).

### Horizon (amplifier, not a separate mechanism)
`t_max` is swept as its own axis (longer sessions let fatigue accumulate, widen
the priming→conversion gap, and compound steered transitions).

---

## 2. The asymmetry (held constant, by construction)

When the dataset is generated with `--sequential`, `SimState.to_state_dict`
appends bucketed history features (`interventions_shown_bucket`,
`steps_since_widget_bucket`, `session_step_bucket`, `primed_credit_bucket`) that
are **not** in `POINT_FEATURES`. Therefore:

- the **V2 bandit** keys its context only on `POINT_FEATURES` → ignores history
  (stays myopic);
- **V3** (PPO, FQI) encodes every key in the state dict → conditions on history.

Both see identical per-decision raw context and identical reward; the only
manipulated variable per curve is one dynamics knob. This is the legitimate
algorithm-class treatment, not a thumb on the scale.

---

## 3. Algorithms (fixed)

- **V2 — bandit:** Bayesian-smoothed greedy over `bandit_arm_stats`
  (`PRIOR_COUNT=5`, `PRIOR_MEAN=0`), context_key over `POINT_FEATURES`.
- **V3a — tabular FQI:** conservative fitted-Q (`gamma=0.95`, `iters=25`,
  `conservative_penalty=0.15`).
- **V3b — PPO (corrected):** on-policy rollouts in the simulator each iteration;
  `old_log_prob` = current policy's log-prob at collection time; GAE(λ) with
  `gamma=0.95`, `gae_lambda=0.95`; clipped objective `clip=0.2`, entropy `0.01`,
  value coef `0.5`, grad-clip `0.5`, `lr=3e-4`; `iterations=25`,
  `rollout_sessions=128`, `epochs=4`, `minibatch=256`, hidden `(128,64)`.

Reporting **bandit vs. tabular-FQI vs. deep-PPO** separates "sequential structure
matters" from "deep function approximation matters": if FQI *also* beats the
bandit as knobs rise, the win is sequentiality, not the neural net.

---

## 4. Sweep grid (fixed)

| Knob | Off (anchor) | Swept values |
|---|---|---|
| `fatigue_rate` | 0.0 | 0.0, 0.1, 0.2, 0.4, 0.6 |
| `delayed_reward_strength` | 0.0 | 0.0, 0.25, 0.5, 1.0, 2.0 |
| `transition_coupling_strength` | 0.0 | 0.0, 0.5, 1.0, 2.0, 3.0 |
| `t_max` (horizon) | 20 | 20, 30, 40, 60 |

Axes are swept **independently** (the other knobs held at 0). Seeds: `{0,1,2}`
(≥3; scale to 5–10 for the final thesis run). Dataset = 4000 sessions/cell,
eval = 3000 sessions/cell, evaluation paired on a shared seed per policy.

---

## 5. Primary metric and test

- **Primary:** V3−V2 gap in **mean reward per session**, i.e.
  `mean_reward(PPO) − mean_reward(bandit)`, with the
  `mean_reward(FQI) − mean_reward(bandit)` curve overlaid.
- **Secondary:** normalized lift vs. no-op, and purchase rate (exact, event-based).
- **Uncertainty:** mean ± 95% CI across seeds at each grid point.
- **Significance:** a knob value's gap is "significant" when its 95% CI excludes 0.

---

## 6. Hypotheses

- **H0 (anchor):** at all knobs = 0, gap ≈ 0 (CI contains 0) — reproduces today's
  near-tie and proves nothing else changed.
- **H1:** the gap increases monotonically (in expectation) as each knob rises.
- **H2:** there exists a threshold X beyond which the gap is statistically
  significant — V3 ≈ V2 below X, V3 > V2 above X.

---

## 7. Robustness checks (pre-declared)

- **Hidden-feature objection:** repeat the `fatigue_rate` sweep with
  `SEQUENTIAL_EXPOSE_HISTORY_TO_BANDIT=1` (exposes `interventions_shown_bucket`
  to the bandit too). A residual PPO win is genuine temporal credit assignment,
  not information asymmetry.
- **Optimizer confound:** the PPO correction (GAE + proper ratio reference +
  on-policy rollouts) is applied for *all* points, so a win is not an optimizer
  artifact. The `--ppo-mode offline` ablation is reported separately if used.

---

## 8. What would falsify the headline claim

- If the gap is already large at all-knobs-0 (H0 fails) → the asymmetry, not the
  dynamics, is doing the work; investigate before claiming anything.
- If the gap never becomes significant across the whole range → "bandits suffice
  even under strong sequential structure here" (a valid, reportable result).
- If FQI tracks PPO closely → the win is sequentiality, not deep nets (report as
  such); if only PPO wins → function approximation matters too.
