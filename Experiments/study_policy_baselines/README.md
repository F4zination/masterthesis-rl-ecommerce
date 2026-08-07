# Frozen-policy simulator baselines

`evaluate_study_policy_baseline.py` evaluates the exact local V2 bandit
database and V3 PPO checkpoint in the ten aggregate Clickworker design cells
and all 160 policy × archetype × device × traffic strata. Aggregate allocation
is equal across the five archetypes, and the V2/V3 diagnostic pair for each
archetype uses the same deterministic random seed. Context strata use
independent deterministic seeds so their Monte Carlo errors can be propagated
in the preregistered bootstrap.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Experiments\evaluate_study_policy_baseline.py `
  --study-id domain_aligned_20260728_v2 `
  --sessions-per-cell 2000 `
  --context-sessions-per-stratum 500 `
  --seed 20260721
```

The evaluator refuses to write into a non-empty study directory. Use a new
`--study-id` for every new artifact set. Each JSON manifest records artifact
and configuration SHA-256 hashes, the Git revision, simulator dynamics, cell
seeds, runtime versions, action distributions, and coverage/fallback
diagnostics. The JSON contains the aggregate cells and context strata; the CSV
contains one flat row per aggregate policy/archetype cell. The primary human
fidelity analysis standardizes the simulator within each randomized cell to
the recorded joint device/traffic distribution. This avoids confounding
archetype calibration with the simulator's otherwise-uniform context sampler.

## Current domain-aligned reference

The current `domain_aligned_20260728_v2` diagnostic contains 2,000 aggregate
sessions per cell (20,000 aggregate sessions) plus 500 sessions in each of 160
context strata (80,000 additional sessions; 100,000 total). With equal
archetype weighting, the aggregate diagnostic is:

| Policy | Conversion | Mean session reward | Reward SD | Mean steps | Mean max funnel depth |
|---|---:|---:|---:|---:|---:|
| V2 frozen bandit | 7.49% | 1.477 | 3.297 | 2.979 | 2.258 |
| V3 frozen PPO | 7.69% | 1.484 | 3.343 | 2.963 | 2.259 |

Across aggregate and context-stratified cells, V2 had zero unseen contexts,
zero partially covered contexts, zero selected missing arms, and zero
fallbacks in 148,486 decisions. V3 had zero out-of-vocabulary decisions and
zero fallbacks in 148,730 decisions. The static artifact gate independently
validates all 848 V2 contexts and 5,088 context/action arms, plus all 46 PPO
state tokens and six actions.

Complete serving support is not the same as direct empirical support for every
structural combination. Of the 5,088 Bandit arms, 2,160 use device-pooled
empirical statistics and 2,928 use an explicitly labeled one-count
decision-point/action empirical-Bayes backoff. Four PPO token values were
absent from the 20,000-session seed dataset (`item_count_bucket=0/3`,
`page_depth_bucket=0`, and `steps_since_widget_bucket=3`); their input columns
were reserved and initialized to a neutral zero effect before online PPO
rollouts. The artifact can therefore encode them without treating arbitrary
random initialization as learned evidence.

The frozen artifact hashes for this reference are:

- V2 database:
  `bc052c13d7c7ae7278b1e4c19af5ee7f8794d708deff10e31ad0f1e3f34ea251`
- V3 checkpoint:
  `316bf2226390745e6e9c83a31790d35bdba32d24068a65434178336550c00361`
- V3 JSON export:
  `5f4b01b0b111de52b5e0f3833a838812730fc41367c64bc430841a54303b2324`
- Shared training JSONL:
  `ea65ad962359778bf3df72492b32cc7ae09ab9c85c55af6766c63175437eeb32`
- Baseline JSON:
  `abb184083060f90416d3278dec2a5fb2a73aca605a79b81b51969fa640e392bc`

## Legacy diagnosis and origin

The superseded `clickworker_pre_recruitment_20260721` reference exposed the
training/serving mismatch that motivated the domain-aligned rebuild.

V3 used no fallback action, but 5,198 of 29,351 aggregate decisions contained a
serving-time token absent from the checkpoint vocabulary. Across the aggregate
and context-stratified reference, 50,840 of 148,448 V3 decisions (34.2%)
contained at least one out-of-vocabulary token. The context-only rate was
45,642 of 119,097 (38.3%). Token occurrences were
`device_type=unknown` 29,762, `primed_credit_bucket=1` 24,211, and
`primed_credit_bucket=2` 6,219; a decision could contain more than one.

An exhaustive finite-domain check found six vocabulary omissions:
`device_type=unknown`, `item_count_bucket=3`, `steps_since_widget_bucket=3`,
and `primed_credit_bucket=1/2/3`. Across the aggregate and context-stratified
reference, 29,495 of 147,912 V2 decisions (19.9%) visited a completely unseen
context, 637 visited a partially covered context, and 29,507 selected an arm
with no trained row. These gaps originated in empirical-only vocabulary/table
construction, a simulator that never sampled `device_type=unknown`, and
different training/serving history and page/cart semantics. The fail-closed
release checks now reject those legacy artifacts.

These figures are simulator-oracle predictions for the two complete frozen
systems. They do not establish an online-learning effect or identify a
specific sequential mechanism.

`power_analysis.json` is generated by `calculate_clickworker_power.py` from
this exact baseline. Under the planning-only assumptions of zero true
human-simulator aggregate bias and human cell variances equal to the simulator,
the approximate endpointwise equivalence power at 30 participants/cell is
92.3% for conversion, 99.6% for transition count, and >99.9% for funnel depth;
at 40/cell it is approximately 97.8%, >99.9%, and >99.9%. These calculations
do not justify the equivalence margins, do not give joint power without an
endpoint-dependence assumption, and do not improve the weak power for the
predicted V3-V2 policy difference. These calculations use the aggregate
diagnostic SDs and are planning approximations for the context-standardized
primary analysis.
