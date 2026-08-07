# Human Study Preregistration

*(Filenames retain the historical "Clickworker" prefix; the recruited sample is a convenience sample as of Design Revision 1, below.)*

**Status**: draft; commit and timestamp before confirmatory recruitment

**Operational protocol**: [`ClickworkerExperiment.md`](ClickworkerExperiment.md)

**Study release commit**: `TO_BE_FILLED`

**Reference-data manifest**: `TO_BE_FILLED`

**Planned recruitment date**: `03.08.2026`

**Recruitment end date (hard stop)**: `24.08.2026`

This document freezes the confirmatory analysis of the human study. Fields marked `TO_BE_FILLED` must be completed before recruitment. Changes after recruitment begins are amendments and must state their date, rationale, whether outcomes had been inspected, and which analyses they affect.

### Design Revision 1 — 2026-08-02, pre-recruitment, no human outcome observed

*(Earlier wording-only revisions — the open-frame prompt softening and the Section 2.1 digest lock — are incremental and recorded in git history; this is the first revision of the design itself. The unrelated `preregistration_v3.md` belongs to the simulation sweep, not this study.)*

After supervisor consultation the study was redesigned before any data collection:

1. **Sample.** Recruitment moves from the Clickworker platform to an uncompensated convenience sample (personal and academic network). Rationale: supervisor decision; the aim is as many participants as possible inside a fixed window rather than a paid quota.
2. **Personas.** Three of the five scenario cells are recruited — `fastbuyer`, `detailedcomparator`, `windowshopper` — chosen **on the primary endpoint** as the triple with the widest simulator transition-progression span at the best directional power (FastBuyer and WindowShopper have the tightest transition variance). The choice was made against the locked simulator reference before recruitment. It was **not** made to improve any conversion endpoint; the recruited triple's configured conversion (≈10.7%) is in fact *further* from the expected human level than the rejected Explorer variant (≈3.7%), and Section 11.1 records that consequence.
3. **Primary endpoint.** The TOST equivalence family is replaced as primary by a **directional transition-ordering endpoint** (Section 7): the achievable sample cannot pass conversion equivalence even under perfect calibration (the ±0.05 margin sits below the null-calibration floor beneath ≈10 participants per cell), whereas the ordering claim is answerable and is closer to what the transition model asserts. Equivalence on transition count and funnel depth remains as a secondary family; conversion calibration is demoted to exploratory. The confirmatory contrast set was fixed against the regenerated locked reference (`clickworker_release_20260803`) by the selection rule stated in Section 7, which excluded one pair an earlier draft had listed on the strength of a per-step proxy; see the note in Section 7. All of this predates recruitment and any human data.
4. **Dates.** Recruitment 03.08.2026 to 24.08.2026 (three weeks, hard stop), driven by the 16.09.2026 thesis submission deadline.

The superseded design (2 × 5 cells, 300–400 crowdworkers, equivalence primary) is preserved in this file's git history. Nothing in this revision was informed by human data; the simulator reference and power calculations that motivated it are committed artifacts.

## 1. Study purpose and hierarchy

The systems are simulation-pretrained and frozen during deployment. Neither learns from participant data.

1. **Primary (transition ordering):** test whether the simulator's predicted ordering of conditional funnel-advance proportions across the recruited personas holds in humans, via the three pre-specified directional contrasts of Section 7 under an intersection-union rule.
2. **Key secondary:** estimate the equal-persona ITT difference in mean session reward for assignment to the frozen V3 deployment stack versus frozen V2.
3. **Secondary (aggregate calibration):** TOST equivalence of human against policy-matched simulated sessions for transition count and maximum funnel depth over the six recruited cells (Section 7.1).
4. **Secondary (cell-level calibration):** bound the mean absolute discrepancy across the 6 recruited policy × persona cells for the same two metrics (Section 7.2). The aggregate estimand averages *signed* cell discrepancies, so offsetting errors across personas can satisfy it while every persona is miscalibrated; this endpoint closes that loophole.
5. **Exploratory:** conversion calibration (aggregate, cell-level, and all ordering contrasts beyond the confirmatory three), all persona-specific policy effects, policy-by-persona interactions, conversion effects between policies, rank ordering, engagement/trajectory analyses, and fallback-stratified descriptions.

The study has no co-primary A/B claim. It does not test online learning or identify a sequential mechanism.

## 2. Design, assignment, and sample

- Between-subjects 2 policy × 3 instructed-persona factorial (`fastbuyer`, `detailedcomparator`, `windowshopper`).
- Block randomization 1:1 between V2 and V3 within each persona; personas assigned by the dispatcher with equal target allocation. The restriction to three cells is operational configuration (`STUDY_PERSONAS`), not a site change: the two unrecruited scenarios stay deployed and digest-locked but are never assigned, and the dispatcher fails closed on any persona name outside the locked set.
- **Sample:** an uncompensated convenience sample recruited from the researcher's personal and academic network. Participation is voluntary, anonymous at analysis time, and carries no payment or course credit. This population is not crowdworkers and not organic traffic; Sections 11.1 and 12 record what that does and does not license.
- Persona is manipulated by an **open-frame prompt**: one situational opening sentence followed by a shared frame that is worded identically in all recruited cells (browse freely, purchase only if something appeals, no real money, end the session when you feel you have explored enough). No prompt prescribes navigation, number of products viewed, comparison behaviour, purchase, or session duration. This wording replaces the earlier task-prescriptive drafts after July 2026 supervisor feedback; the exact deployed strings are reproduced in [`ClickworkerExperiment.md`](ClickworkerExperiment.md) and are locked by digest in Section 2.1 below.
- One assigned study session per participant.
- **Sample size:** as many participants as possible inside the recruitment window. There is no per-cell quota; `TARGET_PER_CELL` is set high enough that the dispatcher never closes a cell. For analyzability planning (Section 11): ≥10 participants per persona in a contrast's conditioning set is the floor below which that contrast is reported untested; ≈20 per persona-condition cell (~120 total) gives ≈94% power on the weakest confirmatory contrast and ≈91% jointly; ≈10 per persona-condition (~60 total) gives ≈73% and ≈60% respectively.
- Recruitment stops at the hard end date above, regardless of counts. Stopping earlier than the end date is not permitted except for the technical pause conditions in Section 3; stopping is never conditioned on observed outcomes.

A separate operational pilot of up to 20 sessions is excluded from confirmatory analyses. Any change motivated by the pilot is made before main recruitment and documented as a preregistration revision. After the pilot, archive its shop and dispatcher volumes and create a fresh release with empty event, decision, order, cart, discount, and assignment tables; the release manifest must verify that clean boundary.

### 2.1 Scenario wording lock

The five scenario prompts are the study's only manipulation and are locked as of this document. Three of the five cells are recruited (Design Revision 1); the `explorer` and `discounthunter` scenarios remain deployed and locked but receive no assignments, so the lock below covers the full set unchanged. The strings are not transcribed by hand: they are extracted from the Jinja `_scenarios` literal in `DemoSiteV{2,3}/app/templates/home.html` by [`audit_persona_prompt_quality.py`](audit_persona_prompt_quality.py), which also asserts that both condition sites carry byte-identical wording. Each digest is the SHA-256 of the rendered modal text (`title`, `body`, `note`, joined by blank lines).

| Cell | SHA-256 (first 16) | Length (chars) |
|---|---|---|
| `explorer` | `e45a0def8c86b980` | 382 |
| `fastbuyer` | `e1189cee0cc54c21` | 387 |
| `detailedcomparator` | `6ff75714e313aac4` | 409 |
| `discounthunter` | `ed56d5b61eee5d6a` | 339 |
| `windowshopper` | `a476202b552266a5` | 366 |

Set digest (all five joined in the order above, NUL-separated): `b94ced0ca08ba763836db5e4d7eb574265022fd6065b3d3402d0cb83738eff77`.

Two structural properties of the locked set are recorded here because they are the mechanical part of cross-cell fairness and are checkable without a judge: the heading and the closing frame are **identical strings in all five cells**, so the cells differ only in the situational premise; and the rendered lengths span 339–409 characters, a spread of 70 characters, so no cell is materially longer or more demanding to read than another.

The wording was additionally evaluated with an LLM-as-a-judge rubric (clarity, behavioural specificity, neutrality/demand characteristics, absence of hypothesis leakage, cross-cell fairness) against the superseded task-prescriptive draft, scored blind and in randomised presentation order over repeated passes. Judge model, rubric text and hash, per-prompt hashes, shuffle seed, and all raw scores are recorded in `persona_prompt_quality.json`; the methodology is written up in the thesis (Experiment 2, *Scenario Wording: Development and Judge-Based Validation*). Judge scores document the instrument only and enter no inferential analysis — they are not evidence that participants behave as the corresponding simulated archetype, which is the study's empirical question.

**Re-lock rule.** Any edit to a scenario string invalidates the digests above. Such an edit requires re-running the audit, updating this table, and recording the change as a preregistration revision before recruitment resumes. Wording must not change once recruitment has begun.

## 3. Policy and simulation freeze

Before recruitment, record in the reference-data manifest:

- repository commit and dirty/clean state;
- V2 database hash and V3 checkpoint hash;
- all training-data/configuration hashes needed to reproduce the policies;
- container/image versions, Python/dependency lock, and environment settings;
- product-catalogue seed and database hash;
- reward/schema version;
- simulator config, horizon, seed list, sessions per cell, and output hashes;
- the exact 10-cell policy × persona aggregate diagnostic table (the reference
  evaluates all five archetypes even though six cells are recruited, so the
  unrecruited cells remain available as diagnostics);
- all 160 policy × persona × device-type × traffic-source reference strata
  used for the context-standardized calibration comparison; and
- the pooled per-persona `funnel_progression` values over the recruited
  personas — the locked Section 7 ordering prediction.

Required deployment settings are `FREEZE_POLICY=true` on V2 and V3, `REQUIRE_BANDIT_POLICY=true` on V2, and `LEARNER_ENABLED=false`, `POLICY_MODE=ppo_only`, `REQUIRE_PPO_CHECKPOINT=true`, and `TIMING_ENABLED=false` on V3. Both policy readiness tests must pass before recruitment. Intended V3 runtime fallback rate: 0%. If monitoring detects a technical fallback, pause recruitment, preserve already assigned sessions, and verify the frozen release before resuming.

## 4. Populations and data rules

### 4.1 Randomized ITT population

All valid participant IDs in the allocation registry (stored as `worker_id` in the logs, a naming holdover from the superseded platform design) are analyzed by assigned policy. Short sessions, low page counts, attention-check failures, crossovers, and fallback use are not exclusions. For assigned IDs with no logged events, event-derived outcomes are zero; duration is missing. No-start rates are reported by cell.

### 4.2 Attempted-session population

All randomized IDs with a recorded landing event. This is the population for the primary ordering analysis and the calibration analyses, because simulated sessions begin at landing. Early exits remain included.

### 4.3 Administrative exclusions

Only malformed IDs, internal test IDs, and duplicate registrations by the same person identified under outcome-blind rules may be removed. For repeated sessions from one valid participant, use the first assigned session. Record every removal and duplicate in a flow table.

### 4.4 Sensitivity quality subset

The following previously proposed criteria define a sensitivity subset only: at least 90 seconds, at least three page views, passed attention check, and at least one dwell or widget-dismissal event. Because these variables may be caused by policy assignment, they do not define the ITT or primary fidelity sample.

## 5. Compensation

Participation is voluntary and uncompensated (Design Revision 1): no payment, bonus, credit, or reward of any kind, and therefore nothing that could differ across policy, persona, purchase, or checkout outcomes. Completion-code access follows the same procedure in every randomized cell so the participant-facing flow is identical across cells.
The end-session control is immediately available; neither elapsed time nor page count gates completion-code access.

## 6. Outcomes

### 6.1 Primary outcome: conditional funnel progression

For each session, **maximum funnel depth** is coded landing=1, PDP=2, cart=3, checkout=4, purchase=5. Because the funnel is strictly ordered, "ever reached stage k" is exactly "maximum depth ≥ k", so for each adjacent stage pair the **conditional funnel-advance proportion**

$$
\pi_{a}(k) = P(\text{depth} \ge k+1 \mid \text{depth} \ge k, \text{persona}=a)
$$

is a deterministic function of the session-level depth distribution. Both sides compute it identically: the simulator reference emits per-cell funnel-depth histograms (`funnel_progression` in [`evaluate_study_policy_baseline.py`](evaluate_study_policy_baseline.py)), and the human side derives the same statistic from the event stream (`transition_ordering_analysis` in [`analyze_clickworker_study.py`](analyze_clickworker_study.py)). This is the session-level funnel advance, **not** the per-step Markov parameter in `archetypes.yaml`; per-step probabilities are not identifiable from human page streams without step-alignment conventions the simulator does not share, and no such convention is introduced. What the depth statistic deliberately ignores — revisit loops before advancing — is measured by the transition-count outcome below, so the two outcomes are complementary rather than redundant.

The primary analysis pools the two policy conditions within each persona: the ordering claim is about personas, and both frozen policies face the same participants under the same randomization.

### 6.2 Secondary calibration outcomes

1. **Transition count:** number of emitted decision transitions using the same start, terminal, and truncation rules in human and simulator data.
2. **Maximum funnel depth:** as coded above, compared as a cell mean.

Human elapsed duration is measured in seconds and is not interchangeable with simulator transition count. Elapsed time, dwell time, raw page views, event rates, empirical distributions, and persona rank order are descriptive fidelity diagnostics.

### 6.3 Key-secondary outcome

**Session reward:** total release-locked event reward minus total action cost over the assigned session. No alternate reward definition will replace it after outcomes are observed.

### 6.4 Cell-level calibration outcome

**Mean absolute cell error (MACE)** for each of the two metrics in Section 6.2, computed over the 6 recruited policy × persona cells and the same context-standardized simulator reference. No additional data collection is required; this endpoint re-uses the Section 7.1 quantities without the sign cancellation.

### 6.5 Exploratory outcomes

**Conversion** (indicator for at least one purchase event) in all its uses: aggregate calibration, cell-level calibration, its ordering contrasts beyond the confirmatory four, and the policy risk difference. Also widget engagement, event rates per transition, order value conditional on purchase, policy-source/fallback rate, and policy × persona contrasts.

## 7. Primary estimand and test: transition ordering

The primary claim is directional: the simulator predicts a strict ordering of the conditional funnel-advance proportions π_a(k) (Section 6.1) across the three recruited personas, and the study tests whether that ordering holds in humans. The prediction — `fastbuyer` > `detailedcomparator` > `windowshopper` on every stage pair — is fixed a priori by the locked simulator reference: the reference's pooled per-persona `funnel_progression` values are the predicted quantities, recorded in the reference-data manifest before recruitment. Simulator Monte Carlo error at 2,000 sessions per cell is negligible against human cells of tens and the prediction is treated as fixed.

**The locked prediction.** Pooled per-persona conditional advance proportions from the locked reference `clickworker_release_20260803` (both policies pooled, 4,000 sessions per persona), which are *the* predicted quantities:

| Persona | landing→PDP | PDP→cart | cart→checkout | checkout→purchase |
|---|---:|---:|---:|---:|
| `fastbuyer` | 0.8672 | 0.5956 | 0.7473 | 0.5984 |
| `detailedcomparator` | 0.8403 | 0.4171 | 0.5913 | 0.4270 |
| `windowshopper` | 0.5272 | 0.0882 | 0.3280 | 0.1311 |

The predicted ordering `fastbuyer` > `detailedcomparator` > `windowshopper` holds on all four stage pairs in the reference.

**Confirmatory contrasts.** Three ordered persona pairs, each written (stage pair; higher > lower), selected by the rule that a confirmatory pair must be separated widely enough in the reference for a null to be interpretable at the achievable sample:

| # | Stage pair | Contrast | Reference gap |
|---|---|---|---:|
| C1 | landing→PDP | `fastbuyer` > `windowshopper` | 0.340 |
| C2 | landing→PDP | `detailedcomparator` > `windowshopper` | 0.313 |
| C3 | PDP→cart | `fastbuyer` > `windowshopper` | 0.507 |

Every other pair is **descriptive**: it is recorded in advance that a null on a descriptive pair is *not* evidence against the simulator. These are landing→PDP `fastbuyer` vs `detailedcomparator` (gap 0.027), PDP→cart `detailedcomparator` vs `windowshopper` (gap 0.329 but reported descriptively, see below), PDP→cart `fastbuyer` vs `detailedcomparator` (gap 0.178), and every contrast on cart→checkout and checkout→purchase, where WindowShopper's conditioning set is ~4.6% of its sessions and therefore never estimable at this scale.

**A note on PDP→cart `fastbuyer` vs `detailedcomparator`.** This pair is descriptive despite the configured per-step probabilities being far apart (0.55 against 0.18). The endpoint is the session-level *ever advanced* proportion, and DetailedComparator revisits product pages repeatedly, so it accumulates many opportunities to reach the cart; its session-level proportion is 0.417 against FastBuyer's 0.596, a gap of 0.178 that would need roughly 360 participants. This is the clearest illustration of why Section 6.1 insists the endpoint is not the per-step Markov parameter, and it is recorded because an earlier draft of this document listed the pair as confirmatory on the strength of the per-step gap before the reference was regenerated. The demotion was made on the pre-stated selection rule, before recruitment and before any human data existed.

**Consequence for the claim.** With C4 demoted the confirmatory set separates `windowshopper` from each of the other two personas, and does not confirmatorily separate `fastbuyer` from `detailedcomparator`. The permitted global claim is correspondingly narrower and is stated in those terms below.

**Test.** For each confirmatory contrast, human proportions are pooled over the two policy conditions within each persona and compared with an unpooled one-sided two-proportion z-test in the predicted direction, α = 0.05 one-sided. Wilson 95% intervals are reported per proportion and a Newcombe hybrid-Wilson 95% interval for the difference. A contrast whose conditioning set (participants who reached the source stage) is below 10 on either side is reported untested. The analysis is implemented and committed before recruitment in [`analyze_clickworker_study.py`](analyze_clickworker_study.py) (`transition_ordering_analysis`; the three contrasts are the `CONFIRMATORY_ORDERING_CONTRASTS` constant, and the descriptive pairs are emitted alongside them from the same reference-derived orientation).

**Decision rule.** The phrase "the simulator correctly orders the recruited personas on funnel advance where it separates them" is permitted only if **all three** confirmatory contrasts are estimable and reject at one-sided α = 0.05. No claim of the form "the full predicted ordering holds" is permitted, because `fastbuyer` versus `detailedcomparator` is not confirmatorily tested at this sample size. This is an intersection-union decision, so no multiplicity adjustment is required. If some but not all hold, report per-contrast results with no global ordering claim; named single-contrast statements ("FastBuyer advanced PDP→cart more often than WindowShopper, as predicted") are permitted for the contrasts that individually reject. If any confirmatory contrast is untestable at the achieved sample, the global claim is reported as not analyzable, not as a failure.

**Population and pooling.** The population is the attempted-session population (Section 4.2). Pooling the policy conditions within a persona is pre-specified and legitimate because the ordering claim is about personas: both frozen policies face participants from the same randomization, and policy assignment is independent of persona by design. Per-policy progression tables are reported descriptively; a policy-dependent ordering difference would be an exploratory finding.

**Context note.** The ordering prediction uses the reference's aggregate cells without device/traffic standardization: a direction, unlike a level, is robust to modest context reweighting, and both personas in any contrast are drawn from the same participant population so context shifts hit both sides of the comparison. The context-standardization machinery below (Section 7.1) continues to apply to the level-based calibration endpoints, where the confound argument is real.

### 7.1 Secondary aggregate calibration (equivalence)

Device type and traffic source are inputs to both frozen policies. The
simulator otherwise samples them uniformly, whereas participants arrive with
their actual device and dispatcher/browser referrer. To avoid treating this
known context-distribution difference as archetype miscalibration, let
`c=(device type, traffic source)`, let `wₚₐc` be its empirical share among
recorded landings in randomized cell `(p,a)`, and let μˢₚₐcₘ be the locked
simulator mean in that context stratum. Define the context-standardized
simulator mean as:

$$
\mu^{S*}_{pam}=\sum_c w_{pac}\mu^S_{pacm}.
$$

The weights use the production-normalized joint context recorded at landing
(`unknown`/`direct` are valid production categories). Device and initial
referrer are measured before policy behaviour. They are not quality filters
or additional exclusion criteria.

For metric (m), policy (p), and recruited persona (a), define the human cell
mean as μᴴₚₐₘ. The estimand is:

$$
Δ_m = \frac{1}{6}\sum_{p \in \{V2,V3\}}\sum_{a=1}^{3}(\mu^H_{pam}-\mu^{S*}_{pam}).
$$

The 6 recruited cells are equally weighted, matching equal target allocation. The YAML `mixture_prior` is not used in any confirmatory analysis; a YAML-weighted estimate may be a labelled sensitivity analysis and cannot validate organic archetype prevalence.

Pre-specified equivalence margins:

| Endpoint | Tier | Lower margin | Upper margin |
|---|---|---:|---:|
| Transition count | Secondary | −0.50 | +0.50 |
| Maximum funnel depth | Secondary | −0.50 | +0.50 |
| Conversion | Exploratory | −0.05 | +0.05 |

For each endpoint, perform TOST at α=0.05; equivalently, require the two-sided 90% confidence interval to fall wholly within the margin. Confidence intervals use a participant-level bootstrap stratified by the 6 randomized cells. Each replicate re-estimates the joint context weights from its resampled participants. Simulator Monte Carlo uncertainty is incorporated by independently drawing every required context-stratum mean from a normal distribution with the recorded mean and standard error (`SD / sqrt(n)`) inside every bootstrap replicate. Context-stratum simulations use independent deterministic seeds; the release reference covers all policy × persona × device × traffic strata. The resulting aggregate simulator Monte Carlo standard error is reported for every endpoint.

The phrase "aggregate calibration within the pre-specified tolerances" is permitted only if both secondary endpoints pass (intersection-union; conversion, as exploratory, does not gate it but is reported alongside with the same machinery). Equivalence verdicts are additionally conditional on achieved sample size: report each endpoint's realized 90% CI width against its margin, and where the CI could never have fallen inside the margin at the achieved n, report the endpoint as inconclusive-by-design rather than failed. Also report 95% confidence intervals, cell gaps, and mean absolute cell error.

Spearman rank correlations across the three recruited personas are descriptive. The study tests scenario-induced conditional fidelity; it neither estimates nor validates organic persona prevalence.

### 7.2 Secondary cell-level calibration estimand and test

Using the same μᴴₚₐₘ and context-standardized μˢ*ₚₐₘ over the recruited cells, define the mean absolute cell error:

$$
\mathrm{MACE}_m = \frac{1}{6}\sum_{p \in \{V2,V3\}}\sum_{a=1}^{3}\left|\mu^H_{pam}-\mu^{S*}_{pam}\right|.
$$

Unlike Δₘ, MACEₘ does not allow a positive discrepancy in one persona to cancel a negative discrepancy in another. It is the quantity that speaks to whether the recruited archetype configurations in `CustomerSimulation/config/archetypes.yaml` describe per-archetype behaviour, as opposed to whether the simulator is unbiased on average.

**Tolerances.** Each cell-level margin is set at twice the corresponding aggregate margin in the table above:

| Endpoint | Tier | MACE tolerance |
|---|---|---:|
| Transition count | Secondary | 1.00 |
| Maximum funnel depth | Secondary | 1.00 |
| Conversion | Exploratory | 0.10 |

The factor of two is a feasibility choice, not a substantive claim: per-cell precision is roughly √6 times worse than the 6-cell aggregate, and a tolerance equal to the aggregate margin would be indistinguishable from the sampling floor below. These tolerances are substantive and fixed; they are not to be revised after observing human outcomes.

**Test.** MACEₘ is non-negative, so TOST does not apply. Compute the one-sided upper limit of a 95% percentile bootstrap interval using the identical resampling scheme as the Section 7.1 analysis — participant-level resampling stratified by the 6 cells, context weights recomputed within each replicate, and each required simulator stratum mean redrawn from its Monte Carlo distribution. The endpoint passes if that upper limit falls below the tolerance.

**Null-calibration floor (mandatory reporting).** MACE is biased upward: because E|X| ≥ |EX|, sampling noise alone yields a positive MACE even under perfect calibration. Before unblinding, compute the expected MACEₘ under the null of zero true discrepancy in every cell by parametric bootstrap from the realized cell standard errors, and report it beside every observed MACEₘ. If a tolerance is not comfortably above its floor, that endpoint has little power to detect miscalibration and must be reported as inconclusive rather than as a pass. Report the floors computed at the achieved sample size, not only the planning values below.

Floors are computed by [`calculate_mace_floors.py`](calculate_mace_floors.py) (`--personas "FastBuyer,DetailedComparator,WindowShopper"`) from the recruited cells of the locked reference `study_policy_baselines/clickworker_release_20260803/`, combining human sampling error with simulator Monte Carlo error at 2,000 sessions per cell; the record is `mace_floors_recruited.json` alongside the baseline. Human cell variance is assumed equal to the simulator cell variance, and conversion uses the Bernoulli SD. "H0 upper" is the one-sided 95% limit under zero true discrepancy — the value the observed limit must exceed for a failure to mean anything. Because there is no recruitment quota, floors are tabulated across the plausible range of achieved cell sizes:

| Endpoint | n/cell | Floor | H0 upper | Tolerance | Headroom |
|---|---:|---:|---:|---:|---:|
| Transition count | 10 | 0.447 | 0.714 | 1.00 | 1.4× |
| Transition count | 20 | 0.317 | 0.506 | 1.00 | 2.0× |
| Transition count | 30 | 0.259 | 0.414 | 1.00 | 2.4× |
| Transition count | 40 | 0.225 | 0.359 | 1.00 | 2.8× |
| Funnel depth | 10 | 0.270 | 0.422 | 1.00 | 2.4× |
| Funnel depth | 20 | 0.191 | 0.300 | 1.00 | 3.3× |
| Funnel depth | 30 | 0.156 | 0.245 | 1.00 | 4.1× |
| Funnel depth | 40 | 0.136 | 0.213 | 1.00 | 4.7× |
| Conversion | 10 | 0.063 | 0.104 | 0.10 | 1.0× |
| Conversion | 20 | 0.045 | 0.074 | 0.10 | 1.4× |
| Conversion | 30 | 0.037 | 0.060 | 0.10 | 1.7× |
| Conversion | 40 | 0.032 | 0.052 | 0.10 | 1.9× |

Both secondary tolerances clear their floors from about 10 participants per cell upward, with transition count the thinner of the two at small n. Conversion at 10 per cell cannot pass even under perfect calibration (headroom 1.0×) — this is part of why conversion is exploratory in this design — and remains marginal through the plausible range. Recompute all floors at the achieved sample size before unblinding, per the mandatory-reporting rule above.

**Known limitation.** MACE is a mean over 6 cells, so one badly miscalibrated persona contributes only a sixth of its error and can be diluted below the tolerance by five well-calibrated cells — less dilution than the superseded 10-cell design, but still real. The mandatory signed-cell table is where a single failing persona becomes visible, and it must be inspected before any calibration claim is made; the simulator's per-cell conversion spread is already extreme (WindowShopper ≈0.002, FastBuyer ≈0.23), so a large error confined to one persona is a realistic outcome.

**Sign pattern (mandatory reporting).** Report all 6 signed cell discrepancies per metric together with the heterogeneity statistic Q_m = Σₚ Σₐ (Δ̂ₚₐₘ / SEₚₐₘ)², referred to χ² on 6 degrees of freedom. Q is a detection test, not an equivalence test: a large Q is evidence of cell-level miscalibration, whereas a small Q is not evidence of calibration. Its role is to make offsetting per-persona errors visible whenever Δₘ passes.

## 8. Key-secondary policy estimand and analysis

The estimand is the equal-persona ITT mean-reward contrast over the recruited personas:

$$
τ_R = \frac{1}{3}\sum_{a=1}^{3}\left[E(R\mid A=V3,a)-E(R\mid A=V2,a)\right].
$$

Estimate τᴿ by averaging the three within-persona V3-minus-V2 mean differences with equal weight. Implement this as standardized marginal means from a saturated `reward ~ policy * persona` model. The policy and persona main effects are included; interaction terms permit stratum effects to differ without changing the pre-specified equal-persona average. Report the average mean difference and a heteroskedasticity-robust two-sided 95% confidence interval. Pre-specified robustness analyses are:

1. participant bootstrap, resampling within policy × persona cells; and
2. randomization inference, permuting policy labels within persona blocks.

The mean is the target, so Mann–Whitney U is not substituted as the main test. Persona-specific effects, the omnibus policy × persona interaction, and rank-based comparisons are exploratory.

The analogous policy conversion estimand is the equal-persona risk difference. Report its 95% confidence interval as exploratory; do not claim each persona cell is independently powered.

## 9. Missing data and protocol deviations

- Reconstruct outcomes from raw logs using one version-locked pipeline.
- Report missing fields, broken joins, and incomplete event streams by randomized cell.
- ITT event outcomes for randomized no-starts are zero; report a recorded-landing sensitivity analysis.
- Do not impute elapsed duration for no-starts.
- Analyze V3 fallbacks under assigned V3 in ITT. Report fallback frequency and causes. Do not remove fallback sessions to manufacture a PPO-only effect.
- Crossovers remain assigned to the randomized arm. An as-treated result, if shown, is exploratory.

## 10. Multiplicity and reporting

The primary transition-ordering claim uses the intersection-union rule in Section 7: all four confirmatory contrasts at one-sided α=0.05, no adjustment required. The key-secondary reward contrast is one pre-specified two-sided test at α=0.05, reported regardless of the primary result. It remains secondary and is not promoted if the ordering claim fails. The Section 7.1 equivalence family and the Section 7.2 MACE family are each intersection-union within themselves over their two secondary endpoints (transition count, funnel depth); conversion accompanies both families as exploratory and gates nothing. All other analyses emphasize estimates and confidence intervals and are labelled exploratory; no selective subgroup claim is made from unadjusted p-values.

The cell-level endpoint is reported in full regardless of the primary result, and no endpoint gates another — an ordering pass with a calibration failure, and the reverse, are both reportable outcomes and both are informative. In particular, the ordering claim passing while the level-based equivalence fails is a coherent and expected pattern: it would say the simulator ranks the personas correctly while miscalibrating their absolute levels.

**Claim wording is constrained as follows, and this constraint is the reason the endpoint exists:**

| Result | Permitted claim |
|---|---|
| Δ passes, MACE passes | Aggregate **and** cell-level calibration within tolerance. Only this combination supports a general statement that the archetype configurations are calibrated. |
| Δ passes, MACE fails | Aggregate calibration only. Report explicitly that per-archetype errors offset one another, name the cells and their signs, and make no archetype-level validation claim. |
| Δ fails, MACE passes | Report both; a systematic shift affecting all cells in one direction is the likely reading. |
| Either inconclusive | Report endpoint-specific results with no global validation statement. |

No phrasing along the lines of “the archetypes were validated” is permitted on the strength of the aggregate endpoint alone. This table governs the calibration claims only; the permitted wording for the primary ordering claim is defined in Section 7.

Report participant flow, cell counts, all pre-specified endpoints, null and adverse results, protocol deviations, and all preregistration amendments.

## 11. Power assumptions

**Primary (transition ordering).** One-sided two-proportion normal approximations at α=0.05 for the three confirmatory contrasts, using the locked reference gaps of Section 7 and the reference's own conditioning rates (the PDP→cart denominator is the landing→PDP advancers, which differ per persona). n is the *pooled* per-persona count (both policy conditions; per-condition cells are half of this, total ≈ 3n):

| Contrast | Gap | n=20 | n=40 | n=60 | n=80 |
|---|---:|---:|---:|---:|---:|
| C1 landing→PDP `fastbuyer` > `windowshopper` | 0.340 | 81% | 97% | >99% | >99% |
| C2 landing→PDP `detailedcomparator` > `windowshopper` | 0.313 | 73% | 94% | 99% | >99% |
| C3 PDP→cart `fastbuyer` > `windowshopper` | 0.507 | 97% | >99% | >99% | >99% |

C2 is the binding contrast. Under independence the joint intersection-union power is approximately 60% at pooled n=20 (~60 participants total), 91% at n=40 (~120 total), and >98% at n=60 (~180 total); the contrasts share persona cells, so this is an approximation rather than a bound.

For reference, the demoted pair PDP→cart `fastbuyer` > `detailedcomparator` reaches only 28% at pooled n=20, 44% at n=40 and 83% at n=120 (~360 participants total). Retaining it as confirmatory would have capped the joint power of the whole primary endpoint at those values, which is why the pre-stated selection rule excludes it. It is reported descriptively with its interval, and a null there is uninformative.

**Key-secondary reward contrast.** With planning reward SD ≈3.3 (locked reference), the approximate mean-reward MDE at two-sided α=0.05 and 80% power is ≈1.4 units at 30 participants per policy-persona cell (~180 total) and ≈1.0 units at 60 per cell (~360 total). The 20,000-session exact-artifact diagnostic at the locked release (all five archetypes, equal weight) predicts V3−V2 mean reward of +0.0069 and conversion risk difference of +0.0020 under equal persona weighting over the recruited triple. These values are planning evidence of a near tie, not an alternative hypothesis and not a basis for a directional test. The predicted gap is orders of magnitude below any achievable MDE, so the key-secondary contrast is expected to be reported as an uninformative null; that is a known and accepted property of this design, not a reason to re-scope it.

An earlier version of this section recorded −0.108 and −0.0005 from the superseded `clickworker_pre_recruitment_20260721` reference, which predated the policy domain-alignment fix. The sign of both quantities changed when the reference was regenerated. Neither value was ever a hypothesis, and this revision is dated before recruitment and before any human outcome was observed.

**Secondary equivalence endpoints.** Planning power for the Section 7.1 TOST family is a function of the achieved cell size and is generated by `calculate_clickworker_power.py` against the locked reference `study_policy_baselines/clickworker_release_20260803/` (`power_analysis.json`); recompute at the achieved cell size and report alongside the results. At 20 participants per cell the transition-count and funnel-depth endpoints retain high endpointwise power under zero true bias; conversion equivalence is not powered at any plausible size of this sample (Section 7.2 floors), which is why it is exploratory.

### 11.1 Pre-data expectation for the conversion level

The simulator-side conversion baseline is a property of the recruited design, not a forecast of participant behaviour. Over the three recruited personas the equal-persona aggregate conversion of the frozen artifacts is 10.53% (V2) and 10.90% (V3), dominated by one archetype: FastBuyer converts at 22.55% and 23.65% in the two arms, while WindowShopper converts at 0.25% and 0.15%. This is *higher* than the 7.5% of the superseded five-persona design because the recruited triple was selected on the transition endpoint (Design Revision 1), and that selection consequence is recorded here rather than hidden: the rejected Explorer variant would have had a simulator baseline of ≈3.7%, far closer to the expected human level. Choosing personas to flatter the conversion comparison would have been the same circularity that item 3 below forbids for parameter tuning, so the conversion cost was accepted knowingly.

External evidence indicates the human side of the comparison will land far below 10.7%:

- **Organic level.** IRP Commerce reports a session conversion rate of 2.03% for the general e-commerce market in June 2026 (1.85% a year earlier), computed as transactions ÷ sessions from first-party platform data; its sector figures relevant to this catalogue span roughly 1.6–2.8%. The Contentsquare Digital Experience Benchmark 2026 (99 billion sessions, 6,500+ sites) reports channel-dependent rates between 0.7% and 2.8%.
- **Instructed-participant inflation, amplified by the sample.** Demand characteristics (Orne 1962; Zizzo 2010; de Quidt, Haushofer & Roth 2018) predict upward pressure relative to organic traffic, and a personal-network convenience sample is the strongest plausible version of that pressure: participants know the researcher, may infer that "success" means buying, and may wish to be helpful. The open-frame wording of Section 2 attenuates but cannot eliminate this. Murphy, Allen, Stevens & Weatherhead (2005) report a median hypothetical-to-actual ratio of 1.35 across 28 studies and 83 paired observations; the ratio concerns stated valuations, not purchase incidence, and transfers only as an order-of-magnitude anchor.

The organic anchor (~2%) and the inflation channel pull in opposite directions and neither is quantifiable for this population, so no point expectation is recorded — the honest statement is an interval: human aggregate conversion plausibly lands anywhere between ~3% and ~8%, and every value in that interval leaves the human−simulator discrepancy **outside** the ±0.05 exploratory margin against a 10.7% baseline. Conversion calibration is therefore expected to fail in this design, which is one of the two reasons (with the Section 7.2 floors) it is exploratory rather than confirmatory. Transition count and maximum funnel depth carry no analogous external evidence of a large offset and remain the secondary calibration endpoints.

This expectation is recorded for interpretation and is subject to the following constraints, which exist to prevent it from being used as a licence:

1. It is **not** an alternative hypothesis, not a directional test, and not a prediction that will be scored against the data.
2. It does **not** justify widening the ±0.05 margin. That margin is a substantive tolerance for what counts as adequate calibration for policy pretraining; expecting to miss a substantive tolerance is precisely what such a tolerance is for. Any margin change remains a dated, documented pre-recruitment decision under Section 7.
3. It does **not** justify adjusting `archetypes.yaml`, retuning the FastBuyer purchase parameters, or regenerating the reference to close the gap. The archetype parameters are specified a priori (Section 12); tuning them toward an anticipated human result would reintroduce the circularity that the open-frame prompt revision removed.
4. A conversion discrepancy is to be reported as an interpretable finding about the archetype configuration — most directly about the FastBuyer purchase parameters that dominate the recruited aggregate — and not as an operational defect of the study.

If the observed conversion instead lands near 10.7%, that is equally reportable and would indicate that instructed participants converge on the configured levels more closely than the external literature predicts.

## 12. Interpretation constraints

- **Population.** The sample is an uncompensated personal-network convenience sample. Results generalize to no other population: not to crowdworkers, not to organic shop traffic, and not to "users" in general. Every claim in this document is a claim about scenario-instructed sessions from this sample. Participants may know the researcher; the demand-characteristic implications are recorded in Section 11.1, and no statistical correction for them is possible or attempted.
- A confirmed transition ordering says the simulator ranks the recruited personas correctly on funnel advance; it does **not** say the simulator's absolute levels are calibrated (that is the separate Section 7.1/7.2 question), and it does not validate the two unrecruited archetypes at all.
- The primary endpoint is the session-level conditional funnel advance. It is not the per-step Markov transition matrix, and no claim about per-step transition probabilities is licensed beyond what the ordering of session-level advances implies.
- The A/B estimates the effect of the two complete frozen deployment stacks in this instructed sample.
- It does not test online policy learning, deployment-time PPO convergence, or adaptation.
- It cannot attribute a policy difference to delayed reward, transition coupling, fatigue, history features, or sequential credit assignment.
- Deterministic actions have no alternative-action overlap. IPS, SNIPS, and doubly robust action-level OPE are not valid on these deployment logs and are not part of the confirmatory analysis.
- Persona prompts test whether instructions evoke configured patterns; they do not establish natural archetypes or their population mixture.
- The open-frame prompts supply a motive, not a task. The manipulation is intentionally weak, so persona separation is an empirical outcome rather than instructed compliance. Absent or small separation between persona cells is an interpretable result about the archetypes, not a protocol failure, and is not grounds for exclusion, reweighting, or a post hoc prompt revision. A failed confirmatory ordering contrast is exactly such a result: it is evidence about the archetype model (or about the weak manipulation), and which of the two cannot be separated by this design.
- Any persona manipulation check is exploratory, reported with estimates and intervals, and does not gate the Section 7 analyses.
- Simulator archetype parameters are specified a priori in `CustomerSimulation/config/archetypes.yaml` and are not derived from, or adjusted to, the participant-facing prompts. Under the open-frame wording the prompts no longer restate those parameters as instructions, so cell-level human-simulator agreement is a genuine prediction of the archetype model rather than a check on instruction compliance.

## 13. Approvals and pre-recruitment sign-off

### 13.1 Approvals

The author attests on **2026-08-02**, before recruitment, that the following were obtained:

| Approval | Obtained | Reference / date of record |
|---|---|---|
| Ethics and privacy approval for an uncompensated personal-network sample | Yes | `RECORD_REFERENCE_PENDING` |
| Supervisor sign-off on Design Revision 1 (sample, three personas, directional primary endpoint) | Yes | `RECORD_REFERENCE_PENDING` |

The supporting documents are held outside this repository. `RECORD_REFERENCE_PENDING` marks the identifier or date of each record, to be filled from the approval correspondence; these are bookkeeping fields and do not affect any analysis. Should either approval be qualified or conditioned in a way that changes the protocol, that is a preregistration amendment under the rule at the head of this document.

### 13.2 Sign-off checklist

- [ ] Fill all `TO_BE_FILLED` fields (study release commit; reference-data manifest path). Both are derived from the final release cut and are therefore filled last.
- [x] Regenerate the simulator reference from the release commit with the current `evaluate_study_policy_baseline.py`, so every cell carries `funnel_depth_histogram`/`funnel_progression`, and record the reference's pooled per-persona progression values as the locked Section 7 prediction (done 2026-08-02, `clickworker_release_20260803`). The predicted ordering holds on all four stage pairs in the regenerated reference.
- [x] Compute the Section 7.2 null-calibration floors from the locked reference over the recruited cells across the plausible achieved-n range (done 2026-08-02, `mace_floors_recruited.json`; conversion does not clear its floor at small n and is exploratory accordingly). Recompute at the achieved n before unblinding.
- [x] Lock dependencies reproducibly for the container platform (done 2026-08-02, `deploy/lock-dependencies.sh`, five `requirements.lock` files resolved for Python 3.12 on Linux).
- [x] Obtain the required ethics/privacy approvals for an uncompensated personal-network sample (see Section 13.1).
- [x] Obtain supervisor sign-off on Design Revision 1 (see Section 13.1).
- [ ] Set `STUDY_PERSONAS=fastbuyer,detailedcomparator,windowshopper` in the deployment environment and verify via the dispatcher that only the three recruited cells receive assignments.
- [ ] Archive any pilot/test volumes and deploy clean shop and dispatcher ledgers.
- [ ] Obtain immutable image identifiers for all five deployment images.
- [ ] Verify checkpoint fail-fast and zero fallback in smoke tests.
- [ ] Verify block randomization and the assignment registry.
- [ ] Verify participation is uncompensated and the participant-facing flow is identical across cells.
- [ ] Generate the final release manifest and archive it alongside the policy artifacts.
- [ ] Commit and timestamp this preregistration with `Status` changed from draft to locked.
- [ ] Run and freeze `evaluate_study_policy_baseline.py` and `analyze_clickworker_study.py` before unblinding.
