# Observations (Offline Evaluation Update)

Date: 2026-05-16

## Context

As a result, OPE  compares the following 4 policies:
- behavior
- no_op
- greedy_empirical
- trained_policy

## Main Results (Point Estimates)
From the latest `ope_results.json`:

- behavior: IPS = 2.2773, SNIPS = 0.3796, DR = 0.3796
- no_op: IPS = 0.3428, SNIPS = 0.3458, DR = 0.2824
- greedy_empirical: IPS = 0.7773, SNIPS = 0.7805, DR = 0.7619
- trained_policy: IPS = 0.5957, SNIPS = 0.6019, DR = 0.5755

## DR Ranking (Primary Comparison)
1. greedy_empirical: 0.7619
2. trained_policy: 0.5755
3. behavior: 0.3796
4. no_op: 0.2824

Interpretation:
- The trained policy improves over the behavior baseline by about +0.196 DR.
- The trained policy is below greedy_empirical by about -0.186 DR.
- no_op remains the weakest policy.

## Confidence Interval Check (DR, 95%)
- behavior: [0.2777, 0.4941]
- no_op: [0.2399, 0.3168]
- greedy_empirical: [0.7093, 0.8085]
- trained_policy: [0.5271, 0.6213]

Observation:
- Intervals are clearly separated for the top 3 order in this run.
- This supports the ordering: greedy_empirical > trained_policy > behavior > no_op.

## Practical Takeaways
- The Makefile change worked: `trained_policy` is now part of OPE and visible in the HTML charts.
- The learned offline RL policy is materially better than historical behavior on this dataset.
- A simple state-wise empirical greedy policy still performs best in this offline comparison.
- For thesis discussion: frame trained_policy as a robust improvement over baseline, but not yet the top performer against strong in-sample heuristics.

## Suggested Follow-Up Analysis
- Compare trained_policy vs greedy_empirical under changed reward shaping and regularization.
- Evaluate sensitivity to `gamma` and `conservative_penalty`.
- Add policy-level plots with DR confidence intervals directly in HTML for clearer uncertainty communication.
- Validate whether ranking persists on alternative datasets (or held-out simulation seeds).
