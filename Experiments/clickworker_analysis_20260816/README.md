# Human study analysis — 16 August 2026 extract

Aggregate outputs backing Section `subsec:HumanResults` (Experiments) and
`subsec:EvalRQ3` (Evaluation) of the thesis.

Recruitment opened 2026-08-03; this is the extract taken 2026-08-16, at which
point the dispatcher ledger held 31 randomized participants and the most recent
assignment was dated 2026-08-12. The preregistered recruitment window runs to
2026-08-24, so re-run the commands below against a later extract if further
participants arrive, and update the thesis numbers from the regenerated files.

Source data is the study backup `Backups/20260816T103004Z/study-data.tar.gz`
(V2 shop, V3 shop, dispatcher ledger). Participant-level output
(`session_level.csv`) is deliberately not stored here; regenerate it locally
from the backup when needed.

The worker identifiers appearing in the `audit_events` block of
`analysis_results.json` are pseudonymized as `audit-worker-1..3`; the platform
identifiers they replace are held with the study data and are not published.
A local regeneration therefore reproduces every reported quantity, but writes
the original identifiers into that block.

## Regenerate

```bash
python Experiments/analyze_clickworker_study.py \
  --db v2=DemoSiteV2/data/demosite.db \
  --db v3=DemoSiteV3/data/demosite.db \
  --dispatcher-db StudyDispatcher/data/dispatcher.db \
  --baseline Experiments/study_policy_baselines/clickworker_release_20260803/policy_archetype_baseline.json \
  --personas fastbuyer,detailedcomparator,windowshopper \
  --out-dir Experiments/clickworker_analysis_20260816
```

`--personas` restricts every cell-averaged estimand to the three personas
Design Revision 1 recruits. Without it the secondary calibration family is
unresolvable, because it averages over cells the design never assigned.

Null-calibration floors at the achieved five participants per cell — the floors
file requires recomputing them at the achieved sample rather than reading the
planning table:

```bash
python Experiments/calculate_mace_floors.py \
  --baseline Experiments/study_policy_baselines/clickworker_release_20260803/policy_archetype_baseline.json \
  --personas FastBuyer,DetailedComparator,WindowShopper \
  --n-per-cell 5 6 \
  --out Experiments/clickworker_analysis_20260816/mace_floors_achieved_n5.json
```

## Outcome

- **Primary (transition ordering): NOT ANALYZABLE.** Two of three confirmatory
  contrasts estimable, neither rejecting (one-sided p = 0.057 and 0.458); the
  third below the pre-registered floor of ten (WindowShopper PDP→cart, n = 8).
- **Secondary (calibration): NOT ESTABLISHED.** Conversion +0.374, transition
  count +7.513, funnel depth +1.447; every interval outside its margin, every
  MACE above the null ceiling, all 18 cell gaps of one sign.
- **Exploratory (frozen-policy ITT):** V3−V2 session reward +2.312
  (p = 0.438), conversion +0.167 (p = 0.429). Indistinguishable at this N.
- **Serving integrity:** 174/174 V3 decisions served by the PPO policy (0%
  fallback), both arms frozen, V3 timing gate disabled.
