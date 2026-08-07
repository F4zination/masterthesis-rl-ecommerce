# Clickworker Study Readiness Plan

**Goal:** collect analyzable data for the frozen-policy 2 × 5 study specified
in [`ClickworkerExperiment.md`](ClickworkerExperiment.md) and
[`ClickworkerPreregistration.md`](ClickworkerPreregistration.md).

**Current status (2026-07-21):** the study application and analysis scaffolding
are implemented, but the candidate release is deliberately **not ready for
recruitment**. The authoritative gate is
[`study_releases/clickworker_draft_20260721/release_manifest.json`](study_releases/clickworker_draft_20260721/release_manifest.json).

## Completed implementation

- V2 and V3 persist worker/persona attribution on events and orders and in
  decision contexts; the schema migration is restart-safe.
- The dispatcher stores consent and assignments and now balances personas
  first, then randomizes V2/V3 within persona.
- Purchase and non-purchase paths both lead to the same attention-check and
  completion-code procedure.
- A correctly matched completion code is always valid for compensation;
  duration, navigation, engagement, and attention flags are sensitivity-only.
- The end-session control is immediately available in both systems. Duration,
  page count, attention, dwell, and dismissals are measured but never used to
  deny compensation or exclude the ITT population.
- V2 freezes its bandit parameter tables and validates that trained rows exist
  before database initialization. V3 uses `POLICY_MODE=ppo_only`, disables
  learning and timing, validates its checkpoint before database initialization,
  and fails requests rather than silently serving another policy.
- A new dispatcher worker receives a fresh shopping session; refreshes and
  resumptions by the same worker retain the existing session.
- The V2 and V3 persona prompts and session-end controls are parity-tested.
- Simulator JSONL/CSV retains `user_type`, and summaries include per-archetype
  metrics.
- The exact-artifact 2 × 5 aggregate diagnostic, full device/traffic-stratified
  simulator reference, participant-level context-standardized analysis,
  release-manifest builder, and resumable sequentiality sweeps are scripted and
  tested.

## Current blockers

1. **Policy state coverage:** the full reference finds OOV tokens in 50,840 of
   148,448 V3 decisions and fully unseen contexts in 29,495 of 147,912 V2
   decisions. The latter is driven mainly by the valid unknown-device state;
   29,507 V2 decisions select an arm with no trained row. Select and train
   provenance-complete policies whose training data cover the production
   domains, then regenerate the aggregate diagnostic and all context strata.
2. **Policy provenance:** the existing V2 DB and V3 checkpoint predate embedded
   dataset/config/code/seed metadata. Rebuild both release artifacts with the
   updated trainers; do not overwrite the current local databases until their
   diagnostic data has been archived.
3. **Release reproducibility:** create and use a dependency lock built for the
   Python 3.12 container platform, build images, and record immutable image
   IDs/digests.
4. **Preregistration:** obtain supervisor agreement on hierarchy, equivalence
   margins, sample/budget, scenario wording, ethics/privacy, retention, and
   stopping date. Fill every `TO_BE_FILLED` field, commit a clean release, and
   timestamp the document before confirmatory recruitment.
5. **Operational validation:** run the 20-session pilot, verify the assignment
   ledger and all joins, compare database/catalog hashes, confirm zero PPO
   fallback/OOV and zero decision-request errors, and run one full flow in
   every cell. Archive those volumes and create clean shop/dispatcher ledgers;
   the release manifest now blocks confirmatory recruitment if pilot/test rows
   remain.

## Evidence already available

The current 100,000-session reference contains 20,000 aggregate sessions and
80,000 context-stratum sessions. Its aggregate diagnostic predicts:

| Frozen policy | Conversion | Mean reward | Reward SD | Mean transitions | Mean max depth |
|---|---:|---:|---:|---:|---:|
| V2 bandit | 7.50% | 1.110 | 2.992 | 2.954 | 2.240 |
| V3 PPO | 7.45% | 1.001 | 2.988 | 2.935 | 2.233 |

This is a near tie, with V3−V2 reward `−0.108`; it does not justify a
directional PPO-superiority expectation. At 300–400 participants, the planned
human A/B is only sensitive to a much larger reward or conversion effect.

The fresh-seed simulator sweep is complete (95/95 cells). Online PPO shows
positive anchor-adjusted effects for low delayed reward and low-to-moderate
transition coupling, but not fatigue. The history robustness run is complete
(25/25), and the fixed-dataset PPO ablation is complete (75/75): it removes
the positive all-off anchor while retaining delayed-reward and coupling
effects. Fresh online interaction is not necessary for those synthetic effects,
but state/model/objective/optimization differences remain.

## Release sequence

1. Agree the design choices with the supervisor.
2. Train/build clean, provenance-complete policy candidates.
3. Generate a new policy-matched aggregate diagnostic and all 160 context
   strata; require zero serving mismatch.
4. Lock dependencies and build immutable images.
5. Complete and commit the preregistration.
6. Generate a new release manifest; proceed only when it reports
   `READY_FOR_RECRUITMENT`.
7. Run the operational pilot, make any outcome-blind fixes, then create a new
   clean release if code or protocol changed.
8. Recruit the confirmatory sample and run
   `analyze_clickworker_study.py` without altering the frozen estimands.

## Verification commands

```text
make study-baseline STUDY_BASELINE_ID=<new-release-id>
make study-release-manifest STUDY_RELEASE_ID=<new-release-id> STUDY_IMAGE_ARGS="..."
make study-analysis STUDY_BASELINE=<locked-baseline.json>
```

The release builder refuses to overwrite an existing release directory and
keeps every unresolved condition visible as a blocker.
