# Clickworker Experiment: Human Calibration of the Simulator and Frozen-Policy Evaluation

**Study title**: Human-in-the-Loop Calibration of Simulated Shopping Behaviour and Evaluation of Frozen Personalization Policies

**Thesis**: *Reinforcement Learning in E-Commerce: Comparing Single and Multi-Step Decision Optimization*

**Design version**: July 2026 draft; lock before recruitment

**Confirmatory specification**: [`ClickworkerPreregistration.md`](ClickworkerPreregistration.md)

> **Frozen-policy design.** V2 and V3 are simulation-pretrained and fixed during this study. Neither system learns from Clickworker participants. The experiment therefore does not test online learning, PPO convergence during deployment, or adaptation speed. It has two purposes: (1) assess how closely scenario-induced human sessions match policy-matched simulator sessions, and (2) directly estimate the randomized difference between the two deployed policy stacks in this instructed study population.

The second purpose is a direct randomized A/B evaluation, not action-level off-policy evaluation (OPE). With deterministic serving, action propensities are 0 or 1 and there is no action overlap for IPS, SNIPS, or doubly robust estimates of alternative actions.

---

## Decision hierarchy

1. **Primary: simulator calibration/fidelity.** Estimate human-minus-simulator differences in conversion, transition count, and maximum funnel depth under the same policy and persona scenario. Equivalence is assessed only for pre-specified, equal-cell aggregate estimands and tolerances.
2. **Key secondary: frozen-policy A/B.** Estimate the intention-to-treat (ITT), equal-persona difference in mean session reward between assignment to the V3 stack and assignment to V2.
3. **Secondary (cell-level calibration):** bound the mean absolute discrepancy across the 10 policy × persona cells for the same three metrics, so that offsetting per-archetype errors cannot pass as calibration. This is the archetype-level evidence.
4. **Exploratory:** policy-by-persona heterogeneity, conversion and engagement differences between policies, rank ordering across personas, trajectory diagnostics, and V3 policy-source/fallback summaries.

There are no co-primary policy and fidelity claims. Failure to find a V2/V3 difference is not evidence of equivalence unless a separate, adequately powered equivalence margin is pre-registered. Passing or failing a fidelity tolerance is informative either way; data collection is guaranteed, “validation” is not.

---

## What the study can and cannot establish

The randomized policy assignment permits a causal statement about assignment to the two **complete frozen systems** for the recruited, instructed population. The systems differ in policy architecture, history inputs, training procedure, training volume, and possible fallbacks. A V3 advantage therefore cannot be attributed specifically to delayed reward, action-dependent transitions, fatigue, or sequential credit assignment.

The persona manipulation tests whether the written scenarios evoke behaviour resembling the corresponding configured archetypes. Following the July 2026 supervisor feedback the prompts supply only a situational motive and never prescribe navigation, comparison, purchase, or duration, so the manipulation is deliberately weak: persona separation is an outcome to be observed rather than an instruction to be followed. The study does **not** show that people naturally fall into five stable archetypes, estimate the prevalence of those archetypes in organic traffic, or validate the simulator's `mixture_prior`.

---

## Experimental design

**Type**: between-subjects, 2 × 5 randomized factorial

**Factor A**: assigned policy stack — V2 frozen contextual bandit vs. V3 frozen PPO-only stack

**Factor B**: assigned persona scenario — five worker-facing tasks

**Assignment**: block-randomize 1:1 between policies within each persona; assign each participant to one persona and one policy; one participant contributes one study session.

| | Explorer | FastBuyer | DetailedComparator | DiscountHunter | WindowShopper |
|---|---|---|---|---|---|
| **V2 (Bandit)** | Cell 1 | Cell 2 | Cell 3 | Cell 4 | Cell 5 |
| **V3 (PPO-only stack)** | Cell 6 | Cell 7 | Cell 8 | Cell 9 | Cell 10 |

**Target**: at least 30 participants per cell (300 total). Forty per cell (400 total) is preferred if budget permits. A separate 20-session operational pilot may test recruitment, logging, completion codes, and data joins. Pilot outcomes are not pooled with the confirmatory sample; any design change is documented before main recruitment. Archive the pilot volumes and issue a fresh release with empty shop study tables and an empty dispatcher ledger before confirmatory recruitment.

### Target populations and units

- **Randomized ITT population:** every valid worker ID assigned through the study allocation table. This is the population for policy comparisons.
- **Attempted-session population:** randomized IDs with a recorded landing event. This is the primary population for human-to-simulator fidelity because every simulated session begins at landing.
- **Unit of assignment and analysis:** participant. If a worker generates duplicate sessions, the first assigned session is used; later sessions are retained only for an audit log.
- Administrative test IDs, malformed IDs, and documented duplicate platform accounts may be removed using rules applied without looking at policy outcomes.

---

## Simulator reference and weighting

Before recruitment, generate an aggregate diagnostic table for all 10 policy × persona cells and a primary reference for all 160 policy × persona × device-type × traffic-source strata using:

- the exact frozen V2 database and V3 checkpoint used in deployment;
- the same product catalogue, reward constants, feature schema, simulator configuration, horizon, and study release commit;
- enough simulated sessions per cell that Monte Carlo uncertainty is reported and is small relative to human uncertainty; and
- retained persona labels and session identifiers so all reference metrics can be reproduced.

Recruitment is balanced across personas, whereas `CustomerSimulation/config/archetypes.yaml` specifies a traffic prior of Explorer 30%, FastBuyer 20%, DetailedComparator 20%, DiscountHunter 15%, and WindowShopper 15%. The **primary fidelity estimand weights each of the 10 randomized cells equally** (20% per persona and 50% per policy), matching the experimental design. A YAML-prior-weighted result may be reported as a labelled sensitivity analysis, but it cannot validate organic prevalence because persona assignment is imposed by the experiment.

Device type and traffic source enter both policy states. The simulator samples
those contexts uniformly, but human participants use their actual device and
normally reach a shop through the dispatcher. Within each of the 10 randomized
cells, the primary analysis therefore standardizes the 16 locked simulator
context strata to the joint device/referrer distribution recorded at human
landing. The production normalizer's `unknown` device and `direct` traffic
categories are included. These are pre-behaviour context variables, not
post-treatment quality criteria. The unstratified 10-cell table remains a
diagnostic and planning summary; it is not substituted for the
context-standardized primary reference.

The simulator's session length is a count of emitted transitions/decision opportunities, not elapsed time. Human **transition count** is constructed with the same start, terminal, and truncation rules and is the quantity compared with the simulator. Human elapsed seconds and dwell time are reported separately; they are not called simulator session-length matches unless the simulator gains a calibrated clock model before preregistration is locked.

### Current exact-artifact diagnostic (not yet a study release)

The 20,000-session reference in [`study_policy_baselines/clickworker_release_20260802/`](study_policy_baselines/clickworker_release_20260802/) evaluates 2,000 sessions in every policy × persona cell with matched seeds. Equal-persona aggregation gives V2 conversion 7.49%, mean reward 1.477 (SD 3.297), and V3 conversion 7.69%, mean reward 1.484 (SD 3.343). Thus the current simulator prior is a near tie—V3 minus V2 is +0.20 percentage points for conversion and +0.007 reward units—not an expected PPO win. The predicted reward gap is about one hundredth of the mean-reward MDE at the preferred budget.

An earlier reference, generated before the policy domain-alignment fix, put the same two contrasts at −0.05 percentage points and −0.108 reward units. Both readings are a near tie; the sign changed when the reference was regenerated, and neither was ever a hypothesis.

The state-coverage mismatch reported against the July artifacts is resolved. Across the aggregate cells and all 160 context strata the reference now records 297,216 decisions with zero out-of-vocabulary token occurrences, zero unseen contexts, zero missing-feature decisions, and zero fallbacks. `make policy-artifacts-check` reports `valid: true` with the full 46-token state and 6-action domains covered and no missing entries.

The stratified reference also exposes a V2 cold-context problem: 29,495 of
147,912 decisions visit a completely unseen bandit context and 29,507 select an
arm with no trained row, primarily because the legacy training data never
covered `device_type=unknown`. No missing-feature fallback occurred, but an
untrained-prior choice is not evidence of successful offline pretraining. The
draft release manifest therefore blocks recruitment until both policies have
provenance-complete artifacts covering the production domains and the full
reference is regenerated with zero critical coverage failures.

---

## System setup

### Deployment

- Deploy DemoSiteV2 and DemoSiteV3 as separate publicly accessible instances.
- Seed both with identical product catalogues using the same seed.
- Replace real payment with a simulated **Order Confirmed** screen; never request real payment details.
- Embed condition and persona in a unique worker link, for example:
  - `https://<v2-url>/?wid=<worker_id>&persona=explorer`
  - `https://<v3-url>/?wid=<worker_id>&persona=fastbuyer`
- Store the allocation table independently of behavioural logs so ITT membership is recoverable even when a participant produces no post-landing events.

### Required frozen-policy settings

| Variable | V2 value | V3 value | Purpose |
|---|---|---|---|
| `FREEZE_POLICY` | `true` | `true` | Prevents policy updates; V2 serves greedily rather than with ε-exploration |
| `REQUIRE_BANDIT_POLICY` | `true` | not applicable | Fails V2 startup if the mounted database has no trained bandit rows |
| `LEARNER_ENABLED` | not applicable | `false` | Disables background PPO retraining |
| `POLICY_MODE` | not applicable | `ppo_only` | Makes the frozen PPO checkpoint the complete V3 treatment; no policy fallback is served |
| `REQUIRE_PPO_CHECKPOINT` | not applicable | `true` | Fails application startup if the PPO checkpoint is missing or incompatible |
| `TIMING_ENABLED` | not applicable | `false` | Gives both systems the same decision-opportunity structure |

Before launch, build and archive a fail-closed study-release manifest with [`build_study_release_manifest.py`](build_study_release_manifest.py). It records the Git commit and dirty state, exact configuration/dataset/database/checkpoint/baseline hashes, canonical catalogue digest, dependency files, runtime versions, immutable container/image identifiers, deployment settings, and embedded policy provenance. Recruitment may begin only when its `ready_for_recruitment` field is `true`; see [`study_releases/README.md`](study_releases/README.md). Run a scripted smoke test of both URLs and all five scenario links.

V2 must fail its readiness check if the mounted database lacks trained bandit rows; V3 must fail if the checkpoint is missing or incompatible. The intended PPO fallback rate is zero. Pause recruitment if monitoring detects a technical fallback, preserve sessions already assigned, and verify the frozen release before resuming. Runtime fallbacks do not justify excluding participants from ITT: if they occur, the randomized estimand remains the effect of the deployed V3 stack. Report their frequency and cause; do not condition on post-assignment fallback status to manufacture a PPO-only comparison.

### Session completion and compensation

- Display a six-digit reference code through the same end-session mechanism in every policy, persona, and purchase outcome. The final-question submit action persists the code with the attributed session before rendering the completion page; participants do not copy or send it manually.
- The end-session control is available without a minimum duration or page count in every cell; duration and page views do not gate the completion code.
- Participation in the convenience sample is voluntary and uncompensated; no payment, course credit, checkout-contingent reward, or purchase-contingent bonus is offered.
- Use the same post-session attention questions in every cell. Attention-check performance flags a sensitivity subset; it does not determine the primary ITT analysis or base compensation.

### Data collection

Retain and join, by worker and session ID:

- the allocation table and recruitment/completion records;
- `DecisionLog`: decision point, action, recorded propensity, context, and V3 `policy_source`;
- the event stream: widget clicks/dismissals, add-to-cart, checkout submit, purchase, scroll and dwell events; and
- timestamps needed to reconstruct session duration and ordering.

`BanditArmStat` should remain unchanged in frozen mode. Verify this before and after the study.

---

## Persona scenarios (worker-facing task instructions)

Each worker receives one description. The section headings below ("Scenario A — Gift Buyer *(Explorer)*") are internal cell labels for this document and the analysis code; neither the persona label nor the archetype name is shown to participants, who see the shared title in all five cells.

### Open-frame design rationale

An earlier draft gave each persona an explicit task ("compare **at least 3 different products**", "complete the purchase", "pay attention to prices, discounts, and any special offers", "after 8 minutes"). Supervisor feedback of July 2026 ([`FeedbackStefanClickworkerText.md`](../../MasterThesis/FeedbackStefanClickworkerText.md)) rejected that formulation: prompts concrete enough to prescribe the behaviour would **manufacture** the archetype pattern they are meant to compare against, so agreement with the simulator would partly measure instruction compliance rather than behavioural correspondence.

The prompts below therefore share one open frame, taken from that feedback:

> You are testing a new online shop. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved. When you feel you have explored the shop enough, end the session.

Every prompt now consists of **one situational opening sentence plus this frame, worded identically across all five cells**. The persona varies the participant's *reason for visiting*; it never states which pages to visit, how many products to view, whether to purchase, or how long to stay. The shared title ("Testing a new online shop") replaces the previous persona-suggestive headings ("Your Task — Gift Buyer", "— Efficient Buyer", …), and the modal's confirmation button reads "Start" rather than "Start Shopping".

Consequences to carry into the analysis and the write-up:

- The manipulation is **weaker by construction**. Persona separation in the observed data is now an empirical result, not an artefact of the instructions — which is what makes the fidelity comparison interpretable, but also makes null separation a live possibility.
- The expected patterns below are **predictions**, not compliance criteria. Do not treat a participant whose behaviour departs from the expected pattern as a protocol deviation, and do not exclude on that basis.
- Any manipulation check on persona separation is exploratory and must be labelled as such; it is not a gate on the primary equivalence analysis.
- The archetype parameters in `CustomerSimulation/config/archetypes.yaml` are **unchanged and were never derived from the prompts**. They are an a priori hand-specified behavioural model; the earlier task-prescriptive prompts were a second-person transcription of that model (`pdp→pdp 0.45` became "compare at least 3 different products", `discount_banner widget_click 0.42` became "pay attention to prices, discounts, and any special offers"). Removing the transcription is what turns cell-level agreement into a falsifiable prediction. The reference table must still be regenerated from the study release, but no archetype re-calibration is implied or required.

Fidelity conclusions remain conditional on these prompts.

### Scenario A — Gift Buyer *(Explorer)*

> **Testing a new online shop**
>
> A friend's birthday is coming up and you have not thought about a present yet. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved and no real order is placed.
>
> There is nothing you have to find or do. When you feel you have explored the shop enough, end the session.

Expected pattern: multiple category visits, broad product exploration, and conversion between FastBuyer and WindowShopper.

### Scenario B — Efficient Buyer *(FastBuyer)*

> **Testing a new online shop**
>
> Your headphones broke last week, so an electronics accessory has been on your list. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved and no real order is placed.
>
> There is nothing you have to find or do. When you feel you have explored the shop enough, end the session.

Expected pattern: short funnel path, high cart-to-checkout conversion, and relatively little exploration.

### Scenario C — Careful Researcher *(DetailedComparator)*

> **Testing a new online shop**
>
> Something for your home has been on your mind lately — a kitchen gadget, a desk item, that sort of thing. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved and no real order is placed.
>
> There is nothing you have to find or do. When you feel you have explored the shop enough, end the session.

Expected pattern: many product-detail visits, deeper funnels, and longer human elapsed time.

### Scenario D — Deal Seeker *(DiscountHunter)*

> **Testing a new online shop**
>
> Money is a little tight this month. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved and no real order is placed.
>
> There is nothing you have to find or do. When you feel you have explored the shop enough, end the session.

Expected pattern: comparatively high engagement with discounts and selective conversion.

### Scenario E — Casual Browser *(WindowShopper)*

> **Testing a new online shop**
>
> You have a few free minutes and nothing in particular in mind. Have a look around and see what the shop has to offer. If a product appeals to you, you are welcome to "buy" it — no real money is involved and no real order is placed.
>
> There is nothing you have to find or do. When you feel you have explored the shop enough, end the session.

Expected pattern: low conversion and relatively frequent early termination.

The deployed wording lives in the `_scenarios` block of `DemoSiteV2/app/templates/home.html` and `DemoSiteV3/app/templates/home.html`; `Experiments/tests/test_study_prompt_parity.py` enforces that the two are byte-identical.

---

## Outcomes and estimands

### Primary fidelity endpoints

For metric (m), policy (p), persona (a), and joint device/traffic
context (c), let μᴴₚₐₘ be the human attempted-session mean, μˢₚₐcₘ the
locked simulator stratum mean, and wₚₐc the observed context share among
recorded landings in that randomized cell. Define
μˢ*ₚₐₘ = Σc wₚₐc μˢₚₐcₘ. The equal-cell aggregate discrepancy is:

$$
Δ_m = \frac{1}{10}\sum_{p \in \{V2,V3\}}\sum_{a=1}^{5}(\mu^H_{pam}-\mu^{S*}_{pam}).
$$

| Endpoint | Common human/simulator definition | Equivalence margin |
|---|---|---:|
| Conversion | At least one purchase event in the session | ±0.05 absolute probability |
| Transition count | Number of emitted decision transitions under harmonized start/end rules | ±0.50 transition |
| Maximum funnel depth | Deepest stage reached: landing=1, PDP=2, cart=3, checkout=4, purchase=5 | ±0.50 stage |

For each endpoint, use two one-sided tests (TOST) at α=0.05, equivalently requiring the two-sided 90% confidence interval for Δₘ to lie wholly inside the margin. “Aggregate calibration within the pre-specified tolerance” may be claimed only if **all three** endpoint criteria pass. This intersection-union rule does not require a multiplicity adjustment for that global claim. Otherwise report which endpoints passed, failed, or were inconclusive; do not replace them with an omnibus “validated/not validated” statement.

**Mean absolute cell error is a pre-specified secondary endpoint, not a diagnostic** (`ClickworkerPreregistration.md` §6.3 and §7.1). Because Δₘ averages signed discrepancies across the 10 cells, offsetting per-archetype errors can pass it while every archetype is individually miscalibrated; MACEₘ removes that cancellation and carries a tolerance of 0.10 for conversion and 1.00 for transition count and funnel depth, tested against the upper limit of a one-sided 95% bootstrap interval. Its null-calibration floor and the 10 signed cell discrepancies are reported alongside it. An archetype-level calibration claim requires both the aggregate and the cell-level endpoint to pass.

The following remain descriptive fidelity diagnostics, not validation gates: empirical distributions, event rates per transition, and Spearman rank correlations across the five personas. With only five ranks, rank-order evidence is necessarily weak. Human elapsed seconds, order value, and raw dwell seconds have no direct simulator equivalent and remain human-only descriptions.

### Key-secondary policy endpoint

Session reward is the sum of event rewards minus action costs under the release-locked `SharedSchema` constants:

$$R_i = \sum_t r_{it} - \sum_t c_{a_{it}}.$$

The key-secondary estimand is the equal-persona ITT mean difference:

$$
τ_R = \frac{1}{5}\sum_{a=1}^{5}\left[E(R\mid A=V3,a)-E(R\mid A=V2,a)\right].
$$

This is the effect of assignment to the entire V3 deployment stack versus V2, not a PPO-only or sequential-mechanism effect.

### Exploratory endpoints

- Equal-persona policy risk difference in conversion.
- Funnel depth, transition count, widget click/dismiss ratio, and event counts per transition.
- Average order value among purchasers, explicitly labelled as conditional on a post-treatment event.
- V3 `policy_source` and fallback rates.
- Policy × persona interaction contrasts and trajectory-pattern summaries.

---

## Statistical analysis

### Primary fidelity analysis

1. Compute reference means from the locked 160-cell policy × persona × device × traffic simulation table.
2. Within every randomized cell, standardize the simulator to that cell's recorded joint device/traffic distribution; then estimate each Δₘ with equal weight across the 10 policy × persona cells, never with the YAML persona mixture.
3. Form participant-level stratified bootstrap confidence intervals, resampling human participants within policy × persona cells and recomputing their context weights. In each replicate, independently draw each required simulator stratum mean from a normal distribution using its recorded mean and Monte Carlo standard error (`SD / sqrt(n)`), and report the aggregate simulator Monte Carlo standard error. The simulator uses independent deterministic seeds across the 160 context cells.
4. Apply the endpoint-specific TOST criteria above. Report estimates and both 90% and 95% confidence intervals.

### Key-secondary A/B analysis

Estimate τᴿ by averaging the five within-persona V3-minus-V2 mean differences with equal weight. This can be implemented as standardized marginal means from a saturated `reward ~ policy * persona` model; policy and persona main effects are included, while the interaction terms allow the five stratum effects to differ. Use heteroskedasticity-robust standard errors and report the average mean difference with a two-sided 95% confidence interval. A participant-level bootstrap stratified by policy × persona and a randomization test that permutes policy labels within persona are pre-specified robustness analyses. The omnibus interaction test itself remains exploratory.

The mean is the estimand, so Mann–Whitney U is not the primary test for reward. A rank-based result may be reported only as an explicitly different, exploratory distributional estimand.

Estimate conversion with the analogous equal-persona risk difference and confidence interval. Persona-specific policy contrasts and the policy × persona interaction are exploratory; approximately 30 participants per cell is not enough to treat each cell as an independently powered confirmatory experiment.

### ITT, missingness, and quality sensitivity

- Include every randomized valid worker ID in the policy ITT analysis according to assigned policy, even after short sessions, low page counts, failed attention checks, crossovers, or V3 fallbacks.
- If an assigned participant never produces a landing or later event, code event-derived ITT outcomes (reward, conversion, and event counts) as zero and report no-start rates by arm. Duration remains missing. Report a sensitivity analysis restricted to recorded landings.
- The primary fidelity analysis includes all attempted sessions, including early exits; it does not condition on 90 seconds, three pages, purchase, or attention-check success.
- Report the former quality-filter subset (duration ≥90 seconds, page views ≥3, passed attention check, and at least one dwell or dismissal event) only as a sensitivity analysis. These variables can be affected by policy, so filtering on them cannot replace ITT.
- Report missingness and logging failures by randomized cell. Do not silently drop incomplete traces.

---

## Power and precision

The previous 25% baseline was unsupported. Current configured-mixture simulations are near 8% aggregate conversion; Explorer and FastBuyer together are about 11%, while only FastBuyer alone is near 25%. Power planning therefore uses an **8% baseline**.

Using the stated equal-arm normal approximation, two-sided α=0.05, and 80% power:

| Conversion contrast | Approximate required participants per policy arm |
|---|---:|
| 8% → 18% | 177 |
| 8% → 13% | 589 |
| 8% → 10% | 3,213 |

With 300 total participants (150 per policy), the conversion MDE is approximately 11.0 percentage points (8.0% → 19.0%). With 400 total (200 per policy), it is approximately 9.3 points (8.0% → 17.3%). The experiment is therefore powered only for very large conversion effects.

For reward, using the current simulator planning standard deviation of about 3 reward units, 150 participants per arm gives an approximate two-sided 80%-power MDE of **0.97 reward units**; 200 per arm gives **0.84**. These are planning values, not guarantees. Report the pilot/blinded pooled standard deviation and the corresponding standardized MDE; do not claim that 30 participants per cell is sufficient for a near-zero policy gap.

For the primary equivalence endpoints, `calculate_clickworker_power.py` uses the exact 10-cell aggregate reference SDs. Under the planning assumptions of zero true aggregate human-simulator bias and equal human/simulator cell variances, endpointwise power at 30 participants/cell is approximately 92.8% for conversion, 99.7% for transition count, and >99.9% for funnel depth; at 40/cell it is approximately 98.0%, >99.9%, and >99.9%. These are pre-context planning approximations, not a replacement for the locked context-standardized analysis. They do not supply joint power without an endpoint-dependence assumption and do not justify the chosen margins substantively.

The current exact-artifact simulator contrast is −0.108 reward units, roughly one eighth of the preferred-budget MDE and in the opposite direction from a PPO-superiority hypothesis. Unless human behaviour produces a much larger stack-level effect than the simulator predicts, the A/B reward result will be imprecise and likely null; that limitation must be stated before recruitment.

The fidelity equivalence margins are substantive tolerances, not effect sizes inferred from the pilot. If the supervisor rejects a margin, revise and timestamp the preregistration before observing confirmatory human outcomes.

External benchmarks place organic e-commerce conversion near 2% and the instructed-participant inflation factor well below the roughly fourfold gap that a 7.5% simulator aggregate implies. `ClickworkerPreregistration.md` §11.1 therefore records, before data collection, that **conversion is the equivalence endpoint most likely to fail**, that its failure would be an interpretable finding about the FastBuyer purchase parameters that dominate the aggregate, and that this expectation licenses neither a wider margin nor any adjustment to `archetypes.yaml`. The same argument, with citations, is written up in `MasterThesis/Content/Experiments.tex` §"Conversion-Rate Context and the Power-Planning Baseline".

---

## Interpretation and relation to the simulation work

- A randomized reward difference is direct evidence about these two frozen deployments in this instructed population; it need not be routed through OPE.
- Deterministic human logs do not support action-level IPS, SNIPS, or doubly robust evaluation of unchosen actions. Such analyses require a separately designed stochastic logging policy with known non-zero propensities and are outside this study.
- A policy difference cannot identify delayed reward, transition coupling, fatigue, or any other simulator mechanism. Mechanism claims require a mechanism-specific or micro-randomized design.
- Human-to-simulator agreement is conditional on the five prompts and the frozen study release. It does not establish organic archetype prevalence or general consumer realism.
- Human sessions may later inform new training or model development, but any retraining is post-study exploratory work and is not part of the frozen-policy A/B estimand.

---

## Pre-launch checklist

- Lock [`ClickworkerPreregistration.md`](ClickworkerPreregistration.md) with a timestamp before confirmatory recruitment.
- Lock the open-frame scenario wording and verify that the shared frame is byte-identical across all five cells in both templates (`test_study_prompt_parity.py`).
- Generate and archive the policy-matched 2 × 5 simulator reference table.
- Archive the study-release manifest and hashes for both policy artifacts.
- Confirm balanced block randomization and a durable assignment registry.
- Verify fixed compensation and identical completion-code access across outcomes.
- Smoke-test event ordering, worker/session joins, reward reconstruction, and harmonized transition counts.
- Verify both policies remain frozen and V3 readiness fails without a valid checkpoint.
- Confirm no real payment details are requested.
- Decide ethics/privacy, consent, retention, and platform requirements before collecting worker IDs or behavioural logs.

---

## File references

| File | Role |
|---|---|
| `Experiments/ClickworkerPreregistration.md` | Locked confirmatory estimands and analysis rules |
| `Experiments/evaluate_study_policy_baseline.py` | Exact-artifact 2 × 5 simulator reference generator |
| `Experiments/analyze_clickworker_study.py` | Participant-level ITT, fidelity equivalence, and audit pipeline |
| `Experiments/study_policy_baselines/` | Versioned simulator reference artifacts and diagnostics |
| `DemoSiteV2/app/config.py` | V2 frozen-policy configuration |
| `DemoSiteV2/app/services/decision.py` | V2 serving and write-freeze logic |
| `DemoSiteV3/app/config.py` | V3 frozen-policy and learner configuration |
| `DemoSiteV3/app/services/decision.py` | PPO-only fail-closed serving and policy-source logging |
| `DemoSiteV2/app/routers/shop.py` / `DemoSiteV3/app/routers/shop.py` | Worker/persona capture and completion flow |
| `DemoSiteV2/app/seed.py` / `DemoSiteV3/app/seed.py` | Identical catalogue seeding |
| `SharedSchema/shared_schema/constants.py` | Release-locked reward and action-cost definitions |
| `CustomerSimulation/config/archetypes.yaml` | Scenario-conditional synthetic behaviour and YAML traffic prior |
