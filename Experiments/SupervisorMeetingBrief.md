# Supervisor Meeting Brief: Frozen-Policy Human Study

**Prepared:** 2026-07-21  
**Current release status:** `DRAFT_NOT_READY` — do not recruit yet

## Bottom line

The colleague's central framing is correct: both deployed systems are frozen,
so the Clickworker experiment is not a test of online learning or PPO
convergence. It can provide external evidence about simulator calibration and
the performance of two complete pretrained deployment stacks.

Three corrections materially change the recommendation:

1. The 25% conversion prior is unsupported. The all-mechanisms-off configured
   mixture converts at 7.92%; FastBuyer alone is 22.82%, and the configured
   Explorer/FastBuyer sub-mixture is 10.83%.
2. The frozen V2/V3 comparison should remain **key secondary**, not co-primary.
   The current exact artifacts predict an almost null, slightly adverse V3−V2
   result (reward −0.108; conversion −0.0005), so 300–400 participants cannot
   reliably detect the expected gap.
3. A stack-level A/B cannot identify delayed reward, fatigue, transition
   coupling, or sequential credit assignment. It can show whether assignment
   to frozen V3 changes outcomes relative to frozen V2 in this instructed
   sample. A mechanism claim needs a targeted intervention or
   micro-randomized design.

Recommended hierarchy:

- **Primary:** aggregate human-to-simulator calibration for conversion,
  transition count, and maximum funnel depth, with pre-agreed equivalence
  margins.
- **Key secondary:** equal-persona ITT mean-reward difference, V3 minus V2,
  reported with its 95% confidence interval even if null or adverse.
- **Exploratory:** policy conversion difference, persona-specific effects,
  interactions, rank ordering, engagement, trajectories, and mechanisms.

## Validation of the colleague's claims

| Claim | Verdict | Evidence / correction |
|---|---|---|
| Neither deployed policy learns | Correct | V2 is frozen greedy bandit serving; V3 is frozen deterministic PPO-only serving. The study cannot test deployment-time convergence. |
| The study validates offline pretraining | Mostly correct | More precisely, it externally evaluates the two frozen stacks and checks simulator calibration. It does not validate the training algorithm in general. |
| Archetype ordering is face-valid | Correct | FastBuyer has PDP→cart 0.55 and checkout purchase 0.60; WindowShopper purchase 0.18 and exit intent 0.38; DiscountHunter discount-banner click 0.42. |
| Archetype magnitudes are calibrated | Not established | Values and sequential mechanisms are synthetic, hand-authored choices. Human agreement must be tested; failure is a result, not a failed study. |
| Aggregate conversion is roughly 8–10% | Correct | The configured-mixture all-mechanisms-off audit is 7.92%; stored datasets range roughly 7.9–9.8%. |
| A 25% baseline justifies about 176 per condition | Incorrect | About 177 per arm corresponds to 8%→18%. Under the repository's stated normal approximation, 25%→35% requires about 329 per arm. |
| Prompted personas validate natural archetypes | Incorrect boundary | They test whether instructions induce the intended conditional patterns. They cannot estimate natural cluster existence or organic traffic prevalence. |
| Fidelity and A/B can use one collection | Correct | The same 2×5 randomized sample supports both analyses, with a declared hierarchy. |
| Promote both to co-primary because the study “cannot lose” | Not defensible | Equivalence can fail, and a null underpowered A/B is inconclusive. Co-primary status adds claim and multiplicity pressure without adding power. |
| The human A/B tests whether sequential mechanisms exist | Incorrect | It compares full stacks. It cannot isolate the reason for a policy difference. |

## Archetype magnitude audit

All mechanisms were disabled and archetypes were simulated separately with a
uniform behavior policy (10,000 sessions each). The locked output is
[`archetype_face_validity_20260721.json`](archetype_face_validity_20260721.json).

| Archetype | Conversion | Mean transitions |
|---|---:|---:|
| Explorer | 2.84% | 3.298 |
| FastBuyer | 22.82% | 2.903 |
| DetailedComparator | 9.41% | 4.260 |
| DiscountHunter | 3.97% | 2.466 |
| WindowShopper | 0.21% | 1.940 |
| Configured mixture | 7.92% | — |

The ordering is plausible, but the spread is extreme—especially 0.21% versus
22.82% conversion—and has not been empirically calibrated. Report cell-level
gaps and rank ordering descriptively; the confirmatory claim is only the
pre-specified equal-cell aggregate calibration claim.

## Simulation result relevant to the thesis

The fresh-seed confirmatory sweep completed 95/95 cells with zero failures.
At all mechanisms off, online PPO already exceeds CB by +0.172 ± 0.112 reward,
so raw gaps cannot be attributed to sequential structure.

- Delayed reward at 0.25 increases the PPO−CB gap relative to anchor by
  **+0.344 ± 0.080**; the effect declines at larger values.
- Transition coupling increases it by **+0.234 ± 0.180** at 0.5,
  **+0.470 ± 0.095** at 1.0, and **+0.452 ± 0.193** at 2.0.
- Fatigue produces no positive anchor-adjusted effect at any nonzero value.
- FQI does not reproduce the PPO gains and is strongly worse under coupling.
- Exposing fatigue history to the bandit does not change the null fatigue
  conclusion.
- Fixed-dataset PPO removes the all-off advantage (−0.017 ± 0.037) but retains
  delayed-reward Δ=+0.296 ± 0.115 at 0.25 and coupling
  Δ=+0.512 ± 0.250 at 1.0, with similar effects across the supported range.

Therefore the simulator supports PPO under two synthetic mechanisms, not all
three. The fixed-dataset result shows that fresh online simulator interaction
is not necessary for those effects. It still does **not** identify sequential
credit assignment alone because PPO differs from the bandit in state/history
access, model class, objective, and optimization, and the fixed-dataset PPO
routine is an ablation rather than a validated general-purpose offline-RL
estimator.

## Power and precision

Using two-sided α=0.05 and 80% normal approximations:

| Contrast | Approximate required n per policy |
|---|---:|
| Conversion 8%→18% | 177 |
| Conversion 8%→13% | 589 |
| Conversion 8%→10% | 3,213 |
| Conversion 25%→35% | 329 |

At 150 participants per policy, the conversion minimum detectable increase is
about 8.0%→19.0% and the mean-reward MDE is about 0.97 reward units (assuming
SD=3). At 200 per policy they are about 8.0%→17.3% and 0.84 reward units.
The current frozen-artifact reward prediction of −0.108 is only about one
eighth of the preferred-budget MDE.

Planning calculations suggest the proposed aggregate fidelity margins may have
good endpointwise power at 30–40 participants per cell under zero true bias,
but they do not establish joint power and do not justify the margins. Margin
choice is a substantive supervisor decision.

## Design correction already implemented

Device and traffic source are policy inputs. The simulator originally sampled
them uniformly, while real workers arrive with their actual device and a
dispatcher/browser referrer. Comparing those aggregates would confound
archetype fidelity with a known context-distribution mismatch.

The release reference now includes all 160 policy × persona × device × traffic
strata. Within each randomized human cell, the primary analysis standardizes
the simulator to the recorded joint device/referrer distribution, then weights
the 10 randomized cells equally. The unstratified 10-cell table remains a
planning diagnostic.

That expanded audit also found that the legacy V2 bandit is fully cold in
29,495 of 147,912 reference decisions and selects an untrained arm in 29,507,
mainly for `device_type=unknown`. The legacy V3 checkpoint has an OOV token in
50,840 of 148,448 decisions. Both are now explicit release blockers.

## Decisions needed from the supervisor

1. Approve or revise the hierarchy: fidelity primary, reward A/B key secondary,
   all mechanism and persona-specific claims exploratory.
2. Approve equivalence margins: conversion ±0.05, transition count ±0.50, and
   funnel depth ±0.50—or provide substantively justified alternatives before
   outcomes are seen.
3. Choose the budget target: 300 minimum or 400 preferred. The larger sample
   improves precision but still does not power the predicted near-zero A/B.
4. Decide whether the thesis needs a separate mechanism-identification study.
   If yes, design it explicitly; do not retrofit that claim onto this A/B.
5. Approve persona wording, fixed compensation, stopping date, privacy/data
   retention, ethics, and platform-compliance details.
6. Approve the release-policy rebuild: both policy artifacts need embedded
   provenance, and the PPO training choice must cover the full production state
   vocabulary before recruitment.

## Operational next steps

1. Archive the completed confirmatory and ablation manifests with the thesis
   evidence package; preserve their predeclared inferential roles.
2. Rebuild V2 and V3 release artifacts with complete provenance. Cover V2's
   unknown-device contexts and train V3 on every production token, including
   `device_type=unknown`, item-count bucket 3, widget-gap bucket 3, and
   primed-credit buckets 1–3.
3. Regenerate the exact aggregate and context-stratified simulator reference;
   require zero PPO out-of-vocabulary decisions and zero fallbacks.
4. Lock Python 3.12 dependencies, build immutable images, fill and commit the
   preregistration, and generate a clean release manifest.
5. Run up to 20 operational pilot sessions. Archive the pilot volumes, then
   create fresh empty V2/V3/dispatcher ledgers for confirmatory recruitment.
6. Recruit only when the manifest says `READY_FOR_RECRUITMENT` and all smoke
   tests pass.

## Ready-to-send supervisor message

> I audited the frozen-policy study design and reran the simulator checks. The
> core framing is sound: neither V2 nor V3 learns during deployment, so the
> Clickworker study evaluates simulator calibration and the external behaviour
> of two pretrained frozen stacks—not online PPO convergence.
>
> I found three changes we should agree before recruitment. First, the 25%
> conversion prior was wrong: the configured mixture is about 8%, and the
> current frozen artifacts predict an almost null V3−V2 gap. Second, I recommend
> keeping simulator fidelity primary and the randomized reward comparison key
> secondary, rather than co-primary; 300–400 participants cannot reliably
> detect the predicted policy gap. Third, the A/B compares complete stacks and
> cannot identify delayed reward or sequential credit assignment.
>
> The fresh-seed 95-cell simulation sweep is complete. It confirms a PPO
> advantage under low delayed reward and low-to-moderate transition coupling,
> but not under fatigue. Fixed-dataset PPO reproduces the delayed-reward and
> coupling effects while its all-off advantage disappears, so fresh online
> interaction is not necessary; state access, model class, objective, and
> optimization remain possible explanations. I have also corrected the human
> fidelity analysis to match simulator references to the observed
> device/referrer mix.
>
> In the meeting I would like approval on the outcome hierarchy, equivalence
> margins, 300 versus 400 participants, whether a separate mechanism study is
> required, and the final ethics/compensation/stopping rules. I will not start
> recruitment until the policy-provenance/state-coverage issues are fixed and a
> fail-closed release manifest reports ready.
