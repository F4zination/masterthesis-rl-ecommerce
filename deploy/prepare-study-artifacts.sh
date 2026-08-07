#!/usr/bin/env bash
# Create the two frozen study treatments before building the production stack:
#   - V3 PPO checkpoint + JSON fallback/metadata
#   - V2 contextual-bandit database
#
# Prefer copying the exact pre-selected artifacts from the machine on which the
# experiment was trained. This helper is the deterministic fallback when those
# artifacts do not yet exist. It generates trajectory-history states and trains
# PPO with --sequential; V2 uses the same sessions but ignores history fields.
set -euo pipefail

repo_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_path"

dataset="${POLICY_DATASET:-OfflineTraining/data/deployment-sequential/offline_transitions.jsonl}"
dataset_summary="${POLICY_DATASET_SUMMARY:-$(dirname "$dataset")/dataset_summary.json}"
simulation_sessions="${POLICY_SESSIONS:-20000}"
simulation_seed="${SIMULATION_SEED:-42}"
ppo_seed="${PPO_SEED:-42}"
ppo_checkpoint="OfflineTraining/outputs/ppo_policy.pt"
ppo_policy="OfflineTraining/outputs/trained_policy.json"
bandit_db="DemoSiteV2/data/demosite.db"
dataset_expected_sessions="${POLICY_EXPECTED_SESSIONS:-}"
if [ -z "$dataset_expected_sessions" ] && [ -z "${POLICY_DATASET:-}" ]; then
  dataset_expected_sessions="$simulation_sessions"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required." >&2
  exit 1
fi
mkdir -p OfflineTraining/outputs OfflineTraining/data/deployment-sequential DemoSiteV2/data

v3_image_ready=0

ensure_v3_image() {
  if [ "$v3_image_ready" = "0" ]; then
    echo "### Building a temporary V3 image with policy validation dependencies ..."
    docker compose build v3
    v3_image_ready=1
  fi
}

validate_policy_artifacts() {
  ensure_v3_image
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$repo_path:/workspace" \
    -w /workspace \
    --entrypoint python \
    demosite-v3:latest \
    OfflineTraining/validate_policy_artifacts.py \
      --training-dataset "$dataset" \
      --require-clean-provenance \
      "$@"
}

validate_training_dataset() {
  ensure_v3_image
  dataset_validation_args=(
    --dataset "$dataset"
    --summary "$dataset_summary"
    --require-sequential
  )
  if [ -n "$dataset_expected_sessions" ]; then
    dataset_validation_args+=(
      --expected-sessions "$dataset_expected_sessions"
    )
  fi
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$repo_path:/workspace" \
    -w /workspace \
    --entrypoint python \
    demosite-v3:latest \
    OfflineTraining/validate_training_dataset.py \
      "${dataset_validation_args[@]}"
}

if [ ! -s "$dataset" ]; then
  if [ -n "${POLICY_DATASET:-}" ]; then
    echo "Error: POLICY_DATASET not found or empty: $dataset" >&2
    exit 1
  fi

  ensure_v3_image

  echo "### Generating sequential history-state data ($simulation_sessions sessions, seed=$simulation_seed) ..."
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$repo_path:/workspace" \
    -w /workspace/CustomerSimulation \
    --entrypoint python \
    demosite-v3:latest \
    run_simulation.py \
      --n-sessions "$simulation_sessions" \
      --seed "$simulation_seed" \
      --sequential \
      --output-dir ../OfflineTraining/data/deployment-sequential
  echo "### Validating the newly generated shared training dataset ..."
  validate_training_dataset
else
  echo "### Validating the existing shared training dataset before retaining it ..."
  validate_training_dataset
  echo "### Keeping the validated existing shared training dataset."
fi

if [ ! -s "$ppo_checkpoint" ] || [ ! -s "$ppo_policy" ]; then
  ensure_v3_image

  echo "### Training the frozen sequential PPO policy (seed=$ppo_seed) ..."
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$repo_path:/workspace" \
    -w /workspace \
    --entrypoint python \
    demosite-v3:latest \
    OfflineTraining/train_ppo_policy.py \
      --dataset "$dataset" \
      --mode online \
      --sequential \
      --timing-mode opportunity \
      --seed "$ppo_seed"

  echo "### Validating the newly trained V3 artifact pair ..."
  validate_policy_artifacts \
    --only ppo \
    --ppo-checkpoint "$ppo_checkpoint" \
    --ppo-policy "$ppo_policy"
else
  echo "### Validating the existing V3 artifact pair before retaining it ..."
  validate_policy_artifacts \
    --only ppo \
    --ppo-checkpoint "$ppo_checkpoint" \
    --ppo-policy "$ppo_policy"
  echo "### Keeping the validated existing V3 artifacts."
fi

if [ ! -s "$bandit_db" ]; then
  echo "### Building the V2 image and frozen contextual-bandit database ..."
  docker compose build v2
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$repo_path:/workspace" \
    -w /workspace \
    --entrypoint python \
    demosite-v2:latest \
    OfflineTraining/build_bandit_policy.py \
      --dataset "$dataset" \
      --out "$bandit_db" \
      --force

  echo "### Validating the newly built V2 policy database ..."
  validate_policy_artifacts \
    --only bandit \
    --bandit-db "$bandit_db"
else
  echo "### Validating the existing V2 policy database before retaining it ..."
  validate_policy_artifacts \
    --only bandit \
    --bandit-db "$bandit_db"
  echo "### Keeping the validated existing V2 database."
fi

echo "### Running the combined fail-closed artifact gate ..."
validate_policy_artifacts \
  --only all \
  --ppo-checkpoint "$ppo_checkpoint" \
  --ppo-policy "$ppo_policy" \
  --bandit-db "$bandit_db"

echo
echo "### Frozen study artifacts are ready:"
sha256sum "$ppo_checkpoint" "$ppo_policy" "$bandit_db"
