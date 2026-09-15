# Human study analysis — 2 September 2026 extract, recruitment window closed

Aggregate outputs backing Section `subsec:HumanResults` (Experiments) and
`subsec:EvalRQ3` (Evaluation) of the thesis. This extract supersedes
`clickworker_analysis_20260816`, which was an interim cut taken while the
recruitment window was still open.

Recruitment ran 2026-08-03 to 2026-08-24 (hard stop, Preregistration §4). The
backup is dated 2026-09-02 and its dispatcher ledger holds 38 randomized
participants, but five were assigned after the window closed (2026-08-26 to
2026-08-29) and are excluded by `--cutoff 2026-08-24`. The analysed population
is therefore the 33 participants randomized inside the window — the 31 of the
2026-08-16 cut plus two who arrived on 2026-08-19 and 2026-08-20.

Source data is the study backup `Backups/20260902T140453Z/study-data.tar.gz`
(V2 shop, V3 shop, dispatcher ledger). Participant-level output
(`session_level.csv`) is deliberately not stored here; regenerate it locally
from the backup when needed.

The worker identifiers appearing in the `audit_events` block of
`analysis_results.json` are pseudonymized as `audit-worker-1..3`; the platform
identifiers they replace are held with the study data and are not published.
A local regeneration therefore reproduces every reported quantity, but writes
the original identifiers into that block.

Achieved sample: 33 participants, 17 V2 / 16 V3, and 11 per persona
(FastBuyer, DetailedComparator, WindowShopper) — five or six per cell against a
planning target of 40 per persona. No participant was excluded; every in-window
assignment produced an observed session; no duplicates; no logged
decision-request errors.

## Regenerate

```bash
python Experiments/analyze_clickworker_study.py \
  --db v2=DemoSiteV2/data/demosite.db \
  --db v3=DemoSiteV3/data/demosite.db \
  --dispatcher-db StudyDispatcher/data/dispatcher.db \
  --baseline Experiments/study_policy_baselines/clickworker_release_20260803/policy_archetype_baseline.json \
  --personas fastbuyer,detailedcomparator,windowshopper \
  --cutoff 2026-08-24 \
  --out-dir Experiments/clickworker_analysis_20260902
```

`--personas` restricts every cell-averaged estimand to the three personas
Design Revision 1 recruits (see Preregistration §14.2).

`--cutoff` drops assignments made, and sessions first seen, after the close of
the preregistered recruitment window; a bare date keeps that whole day, so the
exclusive bound is 2026-08-25T00:00:00 UTC. The option was added on 2026-09-02
and defaults to off, leaving every earlier run reproducible byte-for-byte. The
bound is not sensitive to the timezone it is read in: the last in-window
assignment is 2026-08-20 03:58 UTC and the first out-of-window one is
2026-08-26 14:53 UTC, so any reading of "24 August" partitions the ledger
identically. No in-window participant has any session activity after the bound,
so filtering on assignment time and on session time selects the same 33 people.

Null-calibration floors at the achieved per-cell sample — the floors file
requires recomputing them at the achieved sample rather than reading the
planning table. The achieved cells are still five and six, so this file is
numerically identical to the 2026-08-16 one:

```bash
python Experiments/calculate_mace_floors.py \
  --baseline Experiments/study_policy_baselines/clickworker_release_20260803/policy_archetype_baseline.json \
  --personas FastBuyer,DetailedComparator,WindowShopper \
  --n-per-cell 5 6 \
  --out Experiments/clickworker_analysis_20260902/mace_floors_achieved_n5.json
```

## Outcome

Every headline conclusion of the 2026-08-16 cut is unchanged; two additional
participants move the point estimates without moving any decision.

- **Primary (transition ordering): NOT ANALYZABLE.** Two of three confirmatory
  contrasts estimable, neither rejecting (one-sided p = 0.059 and 0.500); the
  third is still below the pre-registered floor of ten (WindowShopper PDP→cart,
  n = 9, up from 8).
- **Secondary (calibration): NOT ESTABLISHED.** Conversion +0.352, transition
  count +7.409, funnel depth +1.376; every interval outside its margin and
  every MACE above the null ceiling, as before.
- **Exploratory (frozen-policy ITT):** V3−V2 session reward +2.024 (p = 0.483),
  conversion +0.211 (p = 0.311). Indistinguishable at this N.
- **Serving integrity:** 178/178 in-window V3 decisions served by the PPO policy
  (0% fallback), both arms frozen, V3 timing gate disabled, no decision errors
  in either arm.

## Comparison with the 2026-08-16 interim cut

| Quantity | 2026-08-16 (N=31) | 2026-09-02, window closed (N=33) |
|---|---:|---:|
| ITT session reward V3−V2 | +2.312 (p = 0.438) | +2.024 (p = 0.483) |
| ITT conversion V3−V2 | +0.167 (p = 0.429) | +0.211 (p = 0.311) |
| Quality-subset session reward (N=20 → 21) | +6.273 (p = 0.139) | +5.456 (p = 0.201) |
| Quality-subset conversion | +0.244 (p = 0.481) | +0.256 (p = 0.505) |
| Calibration gap, conversion | +0.374 | +0.352 |
| Calibration gap, transition count | +7.513 | +7.409 |
| Calibration gap, funnel depth | +1.447 | +1.376 |
| FastBuyer > WindowShopper, landing→PDP | p = 0.057 | p = 0.059 |
| DetailedComparator > WindowShopper, landing→PDP | p = 0.458 | p = 0.500 |
| WindowShopper PDP→cart conditioning set | n = 8 (below floor) | n = 9 (below floor) |

Both participants added by this extract were randomized to V2 (WindowShopper and
FastBuyer), which is why the ITT reward contrast moves slightly toward zero.
