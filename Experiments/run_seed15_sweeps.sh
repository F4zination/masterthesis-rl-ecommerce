#!/usr/bin/env bash
# 15-seed replication of the v3 sweep family, for the statistical-power item of
# the 2026-08-07 professional review (df = 4 is thin; 14 is not).
#
# Writes to *_v3_s15 directories and never touches the _v3 directories, which
# remain the citable confirmatory evidence until every number in the thesis has
# been migrated. See Experiments/SequentialitySweepRunbook.md.
#
# Seeds are 10..24: a *superset* of the preregistered 10..14. Two reasons.
# The pipeline is bit-reproducible from its seeds, so cells 10-14 must come out
# identical to the existing runs -- a free end-to-end validation that nothing in
# the environment has drifted. And the paired contrasts D(v) and D_FQI(v) need
# the confirmatory and myopic runs to share seeds, so both are extended together
# or neither is.
#
# Order is by what depends on it: confirmatory and myopic carry the central
# claim, the two ablations qualify it.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PY=.venv/Scripts/python
SEEDS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
ARGS="--dataset-sessions 6000 --eval-sessions 5000 --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30"
AXES3="fatigue_rate delayed_reward_strength transition_coupling_strength"
LOG=Experiments/seed15_run.log

echo "=== 15-seed sweep family started $(date -u +%FT%TZ) ===" | tee "$LOG"

run () {
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "--- $name : started $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  # shellcheck disable=SC2086
  if $PY Experiments/run_sequentiality_sweep.py "$@" >>"$LOG" 2>&1; then
    echo "--- $name : OK $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  else
    echo "--- $name : FAILED (exit $?) $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  fi
}

# 1. Confirmatory grid (95 cells at 5 seeds -> 285 at 15).
run confirmatory \
  --out-dir Experiments/sweep_results_confirmatory_v3_s15 \
  --seeds $SEEDS $ARGS

# 2. Myopic replication. gamma and gae-lambda to zero; the runner passes one
#    --gamma to both PPO and FQI, which is what makes D_FQI(v) available.
run myopic \
  --out-dir Experiments/sweep_results_myopic_v3_s15 \
  --seeds $SEEDS --axes $AXES3 --gamma 0 --gae-lambda 0 $ARGS

# 3. Fixed-dataset PPO ablation.
run offline-ppo \
  --out-dir Experiments/sweep_results_offline_ppo_v3_s15 \
  --seeds $SEEDS --axes $AXES3 --ppo-mode offline $ARGS

# 4. History-exposure ablation (fatigue axis only).
run expose-history \
  --out-dir Experiments/sweep_results_expose_history_v3_s15 \
  --seeds $SEEDS --axes fatigue_rate --expose-history-to-bandit $ARGS

echo "" | tee -a "$LOG"
echo "=== all runs finished $(date -u +%FT%TZ) ===" | tee -a "$LOG"
for d in confirmatory myopic offline_ppo expose_history; do
  f="Experiments/sweep_results_${d}_v3_s15/run_status.json"
  [ -f "$f" ] && echo "$d: $(cat "$f")" | tee -a "$LOG"
done
