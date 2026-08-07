# MasterThesisProject — top-level Makefile
#
# Cross-platform: works on Windows (nmake / GNU make via Git Bash / WSL)
#                 and Linux / macOS.
#
# Prerequisites:
#   - Python venv at .venv/ (or activate it before running make)
#   - All per-project requirements already installed
#
# Usage:
#   make demo-v1          Start DemoSiteV1 (port 8001)
#   make demo-v2          Start DemoSiteV2 (port 8002)
#   make demo-v3          Start DemoSiteV3 (port 8003)
#   make simulate         Run customer simulation (default 5000 sessions)
#   make evaluate         Run offline policy evaluation
#   make simulate-eval    Run simulation then evaluate
#
# Override defaults:
#   make simulate N_SESSIONS=10000 SEED=99
#   make evaluate GAMMA=0.95 CONSERVATIVE_PENALTY=0.05
#   make evaluate CONSERVATIVE_PENALTY=0.05
# ---------------------------------------------------------------------------

# Detect OS and set the correct venv python path
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python
else
    PYTHON := .venv/bin/python
endif

SIM_DIR      := CustomerSimulation
OFFLINE_DIR  := OfflineTraining
SIM_OUTPUT   := $(SIM_DIR)/output/offline_transitions.jsonl
TRAINED_POLICY := $(OFFLINE_DIR)/outputs/trained_policy.json
PPO_CHECKPOINT := $(OFFLINE_DIR)/outputs/ppo_policy.pt
STUDY_POLICY_DATA_DIR ?= $(OFFLINE_DIR)/data/deployment-sequential
STUDY_POLICY_DATASET ?= $(STUDY_POLICY_DATA_DIR)/offline_transitions.jsonl
STUDY_POLICY_SUMMARY ?= $(STUDY_POLICY_DATA_DIR)/dataset_summary.json
POLICY_SESSIONS ?= 20000
PPO_SEED ?= 42
PPO_ITERATIONS ?= 40

# Docker image settings (override on the command line)
IMAGE_V2 := demosite-v2
IMAGE_V3 := demosite-v3
TAG      := latest
DOCKER_PORT_V2 := 8002
DOCKER_PORT_V3 := 8003
DATA_V2  := $(CURDIR)/DemoSiteV2/data
DATA_V3  := $(CURDIR)/DemoSiteV3/data
BANDIT_DB := DemoSiteV2/data/demosite.db
STUDY_RELEASE_ID ?= clickworker_release
STUDY_IMAGE_ARGS ?=
STUDY_BASELINE_ID ?= domain_aligned_20260728_v2
STUDY_SESSIONS_PER_CELL ?= 2000
STUDY_CONTEXT_SESSIONS_PER_STRATUM ?= 500
STUDY_SEED ?= 20260721
STUDY_BASELINE ?= Experiments/study_policy_baselines/$(STUDY_BASELINE_ID)/policy_archetype_baseline.json
STUDY_BASELINE_CSV ?= Experiments/study_policy_baselines/$(STUDY_BASELINE_ID)/policy_archetype_baseline.csv
STUDY_POWER_OUT ?= Experiments/study_policy_baselines/$(STUDY_BASELINE_ID)/power_analysis.json
STUDY_MACE_FLOORS ?= Experiments/study_policy_baselines/$(STUDY_BASELINE_ID)/mace_floors.json
STUDY_DISPATCHER_DB ?= StudyDispatcher/data/dispatcher.db
STUDY_ANALYSIS_OUT ?= Experiments/clickworker_analysis
STUDY_BOOTSTRAP ?= 2000
STUDY_PERMUTATIONS ?= 10000

# Sequentiality sweep, preregistration v3 (§D confirmatory + §F follow-ups).
# The seeds and output directories are fixed by preregistration_v3.md and are
# deliberately disjoint from every prior run. Override them only for a
# documented ablation, never for the confirmatory run itself.
SWEEP_V3_SEEDS ?= 10 11 12 13 14
SWEEP_V3_ARGS ?= --dataset-sessions 6000 --eval-sessions 5000 --ppo-iterations 40 --ppo-rollout-sessions 256 --fqi-iters 30
SWEEP_V3_CONFIRMATORY_DIR ?= Experiments/sweep_results_confirmatory_v3
SWEEP_V3_HISTORY_DIR ?= Experiments/sweep_results_expose_history_v3
SWEEP_V3_OFFLINE_DIR ?= Experiments/sweep_results_offline_ppo_v3
SWEEP_V3_SMOKE_DIR ?= Experiments/sweep_results_v3_smoke

# VPS deployment paths. The production host keeps all three repositories
# directly below /root, so these defaults resolve to:
#   /root/MasterThesisProject, /root/MaglioSite, /root/studenplan-planer
# Override either sibling path when a checkout lives elsewhere, for example:
#   make update-stundenplaner STUNDENPLANER_DIR=/srv/studenplan-planer
STUDY_REPO_PATH ?= $(CURDIR)
VPS_REPO_ROOT ?= $(abspath $(STUDY_REPO_PATH)/..)
MAGLIOSITE_DIR ?= $(VPS_REPO_ROOT)/MaglioSite
STUNDENPLANER_DIR ?= $(VPS_REPO_ROOT)/studenplan-planer
GATEWAY_OVERRIDE := $(STUDY_REPO_PATH)/deploy/shared-gateway/docker-compose.magliosite.yml
STUNDENPLANER_TEMPLATE := $(STUDY_REPO_PATH)/deploy/shared-gateway/stundenplaner.conf.template

STUDY_VPS_COMPOSE = docker compose \
	--project-directory "$(STUDY_REPO_PATH)" \
	-f "$(STUDY_REPO_PATH)/docker-compose.yml" \
	-f "$(STUDY_REPO_PATH)/docker-compose.vps.yml"
GATEWAY_COMPOSE = STUDY_REPO_PATH="$(STUDY_REPO_PATH)" docker compose \
	--project-directory "$(MAGLIOSITE_DIR)" \
	-f "$(MAGLIOSITE_DIR)/docker-compose.yml" \
	-f "$(GATEWAY_OVERRIDE)"
STUNDENPLANER_COMPOSE = docker compose \
	--project-directory "$(STUNDENPLANER_DIR)" \
	-f "$(STUNDENPLANER_DIR)/docker-compose.vps.yml"

# Simulation parameters (override on the command line)
N_SESSIONS   := 5000
SEED         := 42
GAMMA        := 0.99
CONSERVATIVE_PENALTY := 0.05

.PHONY: help demo-v1 demo-v2 demo-v3 migrate-v2 migrate-v3 migrate-all simulate evaluate simulate-eval evaluate-split evaluate-sim gen_report \
        docker-build docker-build-v2 docker-build-v3 docker-run-v2 docker-run-v3 bandit-policy ppo-policy \
        study-policy-data study-policies policy-artifacts-check \
        study-baseline study-power study-mace-floors study-analysis study-release-manifest images images-check \
        update-shops update-stundenplaner update-magliosite \
        vps-check-gateway vps-check-shop-artifacts vps-check-stundenplaner \
        sweep-v3 sweep-v3-smoke sweep-v3-confirmatory sweep-v3-expose-history \
        sweep-v3-offline-ppo sweep-v3-status

# Default target
help:
	@echo ""
	@echo "Available targets:"
	@echo "  demo-v1          Start DemoSiteV1  on http://127.0.0.1:8001"
	@echo "  demo-v2          Start DemoSiteV2  on http://127.0.0.1:8002"
	@echo "  demo-v3          Start DemoSiteV3  on http://127.0.0.1:8003"
	@echo "  migrate-v2       Run SharedSchema migrations for DemoSiteV2"
	@echo "  migrate-v3       Run SharedSchema migrations for DemoSiteV3"
	@echo "  migrate-all      Run SharedSchema migrations for both demo sites"
	@echo "  simulate         Run customer simulation (N_SESSIONS=$(N_SESSIONS), SEED=$(SEED))"
	@echo "  evaluate         Run offline policy extraction + training + OPE evaluation"
	@echo "                    (GAMMA=$(GAMMA), CONSERVATIVE_PENALTY=$(CONSERVATIVE_PENALTY))"
	@echo "  simulate-eval    Run simulation then evaluate"
	@echo "  evaluate-split   Train on seed=42 data, evaluate on independent seed=99 data"
	@echo "                    (removes in-sample evaluation bias)"
	@echo "  evaluate-sim     Evaluate trained policy via simulator oracle"
	@echo "                    (ground-truth comparison vs uniform-random and no-op baselines)"
	@echo "  gen_report       Generate OfflineTraining HTML report from current artifacts"
	@echo ""
	@echo "  images           Rebuild product image WebP variants (thumb + full)"
	@echo "  images-check     Verify image variants exist and are up to date"
	@echo ""
	@echo "  study-policy-data Generate the shared aligned sequential training dataset"
	@echo "  ppo-policy       Retrain the frozen V3 PPO policy from that dataset"
	@echo "  bandit-policy    Retrain the frozen V2 Bandit from that dataset"
	@echo "  study-policies   Regenerate data, retrain both policies, and validate them"
	@echo "  policy-artifacts-check  Fail-closed coverage/pairing validation"
	@echo "                    into $(BANDIT_DB) (seeded DB for docker-run-v2)"
	@echo "  docker-build     Build BOTH experiment images ($(IMAGE_V2) + $(IMAGE_V3))"
	@echo "  docker-build-v2  Build only the DemoSiteV2 (bandit) image"
	@echo "  docker-build-v3  Build only the DemoSiteV3 (PPO) image"
	@echo "  docker-run-v2    Build + run DemoSiteV2 (frozen) on http://127.0.0.1:$(DOCKER_PORT_V2)"
	@echo "  docker-run-v3    Build + run DemoSiteV3 (frozen) on http://127.0.0.1:$(DOCKER_PORT_V3)"
	@echo ""
	@echo "VPS updates (run after pulling the corresponding repository):"
	@echo "  update-shops         Rebuild V2/V3 and reload the shared nginx"
	@echo "  update-stundenplaner Rebuild Stundenplaner and reload the shared nginx"
	@echo "  update-magliosite    Rebuild MaglioSite with all shared gateway routes"
	@echo ""
	@echo "  study-baseline   Generate aggregate and context-stratified simulator references"
	@echo "  study-power      Reproduce A/B MDE and fidelity-equivalence planning values"
	@echo "  study-mace-floors Recompute the preregistration §7.1 MACE null-calibration floors"
	@echo "  study-analysis   Run participant-level fidelity and frozen-policy analysis"
	@echo "  study-release-manifest  Build a fail-closed study release manifest"
	@echo "                    (STUDY_RELEASE_ID=$(STUDY_RELEASE_ID); pass immutable IDs in STUDY_IMAGE_ARGS)"
	@echo ""
	@echo "Sequentiality sweep, preregistration v3 (seeds $(SWEEP_V3_SEEDS)):"
	@echo "  sweep-v3-smoke   Validate wiring in about a minute (throwaway output)"
	@echo "  sweep-v3-confirmatory   Sec. D confirmatory grid, 95 cells, ~50-70 min"
	@echo "  sweep-v3-expose-history Sec. F.1 history robustness, 25 cells, ~15 min"
	@echo "  sweep-v3-offline-ppo    Sec. F.2 offline-PPO ablation, 75 cells, ~65-85 min"
	@echo "  sweep-v3         All three in the preregistered order, ~2.5 h total"
	@echo "  sweep-v3-status  Print run_status.json for each v3 output directory"
	@echo ""
	@echo "Override defaults:  make simulate N_SESSIONS=10000 SEED=99"
	@echo "                    make evaluate GAMMA=0.95 CONSERVATIVE_PENALTY=0.05"
	@echo "                    make docker-build TAG=v1.0"
	@echo ""

# ---------------------------------------------------------------------------
# Demo Sites
# ---------------------------------------------------------------------------

demo-v1:
	cd DemoSiteV1 && $(CURDIR)/$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001


migrate-v2:
	cd DemoSiteV2 && $(CURDIR)/$(PYTHON) -m shared_schema.migrations run --db-path demosite_test.db

migrate-v3:
	cd DemoSiteV3 && $(CURDIR)/$(PYTHON) -m shared_schema.migrations run --db-path demosite_test.db

migrate-all: migrate-v2 migrate-v3

demo-v2: migrate-v2
	cd DemoSiteV2 && $(CURDIR)/$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8002

demo-v3: migrate-v3
	cd DemoSiteV3 && $(CURDIR)/$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8003

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

simulate:
	cd $(SIM_DIR) && $(CURDIR)/$(PYTHON) run_simulation.py \
		--n-sessions $(N_SESSIONS) \
		--seed $(SEED)

# ---------------------------------------------------------------------------
# Evaluation  (train → OPE → HTML report, using simulation JSONL directly)
# ---------------------------------------------------------------------------

evaluate:
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) train_offline_policy.py \
		--dataset ../$(SIM_OUTPUT) \
		--gamma $(GAMMA) \
		--conservative-penalty $(CONSERVATIVE_PENALTY)
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) ope_eval.py \
		--dataset ../$(SIM_OUTPUT) \
		--policy-file outputs/trained_policy.json
	@echo ""
	@echo "Done. Open OfflineTraining/outputs/report.html to view the dashboard."

# ---------------------------------------------------------------------------
# Docker images (Clickworker experiment: V2 vs V3)
#
# Both images are built from the repository root (build context = .) because
# requirements.txt installs the shared package via `-e ../SharedSchema`, and the
# DemoSiteV3 image additionally bakes in the frozen policy artifacts from
# OfflineTraining/outputs. The per-site Dockerfile is selected with `-f`.
# ---------------------------------------------------------------------------

# File rule: build the seeded DemoSiteV2 DB (catalog + bandit_arm_stats) from
# simulation transitions. As a real-file target it runs ONLY when the DB is
# missing, so docker-run-v2 can depend on it without ever clobbering a DB that
# already holds collected experiment data. Run `make simulate` first if the
# transitions file does not exist yet.
$(BANDIT_DB):
	$(PYTHON) $(OFFLINE_DIR)/build_bandit_policy.py \
		--dataset $(SIM_OUTPUT) \
		--out $(BANDIT_DB)

# Generate the single sequential dataset used by both study treatments. V2
# deliberately ignores the history fields while PPO consumes them.
study-policy-data:
	cd $(SIM_DIR) && $(CURDIR)/$(PYTHON) run_simulation.py \
		--n-sessions $(POLICY_SESSIONS) \
		--seed $(SEED) \
		--sequential \
		--output-dir ../$(STUDY_POLICY_DATA_DIR)
	$(PYTHON) $(OFFLINE_DIR)/validate_training_dataset.py \
		--dataset $(STUDY_POLICY_DATASET) \
		--summary $(STUDY_POLICY_SUMMARY) \
		--expected-sessions $(POLICY_SESSIONS) \
		--require-sequential

ppo-policy: study-policy-data
	$(PYTHON) $(OFFLINE_DIR)/train_ppo_policy.py \
		--dataset $(STUDY_POLICY_DATASET) \
		--mode online \
		--sequential \
		--timing-mode opportunity \
		--iterations $(PPO_ITERATIONS) \
		--seed $(PPO_SEED) \
		--out-dir $(OFFLINE_DIR)/outputs

# Explicitly rebuild the frozen V2 bandit from the same aligned sessions.
bandit-policy: study-policy-data
	$(PYTHON) $(OFFLINE_DIR)/build_bandit_policy.py \
		--dataset $(STUDY_POLICY_DATASET) \
		--out $(BANDIT_DB) \
		--force

policy-artifacts-check:
	$(PYTHON) $(OFFLINE_DIR)/validate_training_dataset.py \
		--dataset $(STUDY_POLICY_DATASET) \
		--summary $(STUDY_POLICY_SUMMARY) \
		--require-sequential
	$(PYTHON) $(OFFLINE_DIR)/validate_policy_artifacts.py \
		--training-dataset $(STUDY_POLICY_DATASET) \
		--ppo-checkpoint $(PPO_CHECKPOINT) \
		--ppo-policy $(TRAINED_POLICY) \
		--bandit-db $(BANDIT_DB)

study-policies: ppo-policy bandit-policy
	$(MAKE) policy-artifacts-check

docker-build-v2:
	docker build -f DemoSiteV2/Dockerfile -t $(IMAGE_V2):$(TAG) .

docker-build-v3:
	docker build -f DemoSiteV3/Dockerfile -t $(IMAGE_V3):$(TAG) .

# Product image derivatives. The shops serve only these WebP variants; the
# source PNGs are the design master and are excluded from the container
# images (see .dockerignore), so this must be run after changing any of them.
images:
	$(PYTHON) tools/build_image_variants.py

images-check:
	$(PYTHON) tools/build_image_variants.py --check

# Build both experiment images
docker-build: docker-build-v2 docker-build-v3

# Convenience run targets — apply the frozen-policy env vars from the
# ClickworkerExperiment spec so neither instance learns from participant data.
#
# Each instance bind-mounts a host data directory at /app/data (= DATABASE_PATH)
# so collected DecisionLog/event data survives container restarts. For V2 the
# FROZEN bandit must serve a trained policy rather than an empty (effectively
# untrained) one, so $(BANDIT_DB) is a prerequisite: it is auto-built from
# simulation data on the first run and reused (never overwritten) thereafter.
# Use `make bandit-policy` to force a rebuild from fresh simulation data.
docker-run-v2: docker-build-v2 $(BANDIT_DB)
	docker run --rm -p $(DOCKER_PORT_V2):8000 \
		-e FREEZE_POLICY=true \
		-e REQUIRE_BANDIT_POLICY=true \
		-v "$(DATA_V2):/app/data" \
		--name $(IMAGE_V2) $(IMAGE_V2):$(TAG)

# V3's policy comes from the baked-in ppo_policy.pt, so a fresh seeded DB is
# fine; the volume only persists collected experiment data.
docker-run-v3: docker-build-v3
	$(PYTHON) -c "from pathlib import Path; Path(r'$(DATA_V3)').mkdir(parents=True, exist_ok=True)"
	docker run --rm -p $(DOCKER_PORT_V3):8000 \
		-e FREEZE_POLICY=true \
		-e LEARNER_ENABLED=false \
		-e POLICY_MODE=ppo_only \
		-e REQUIRE_PPO_CHECKPOINT=true \
		-e TIMING_ENABLED=false \
		-v "$(DATA_V3):/app/data" \
		--name $(IMAGE_V3) $(IMAGE_V3):$(TAG)

# ---------------------------------------------------------------------------
# VPS application updates
#
# These targets deploy code that has already been pulled into its corresponding
# repository. They intentionally never run `git pull`, retrain a policy, replace
# a database, request certificates, or remove a Docker volume.
#
# MaglioSite nginx is the shared public ingress for MaglioSite, V2/V3, the
# dispatcher, and Stundenplaner. Recreated upstream containers can receive new
# Docker IP addresses, so every update validates and reloads nginx afterward.
# ---------------------------------------------------------------------------

vps-check-gateway:
	@test -f "$(MAGLIOSITE_DIR)/docker-compose.yml" || { \
		echo "Error: MaglioSite compose file not found at $(MAGLIOSITE_DIR)/docker-compose.yml"; \
		exit 1; \
	}
	@test -f "$(GATEWAY_OVERRIDE)" || { \
		echo "Error: shared gateway override not found at $(GATEWAY_OVERRIDE)"; \
		exit 1; \
	}
	@test -s "$(STUNDENPLANER_TEMPLATE)" || { \
		echo "Error: Stundenplaner nginx template is missing or empty: $(STUNDENPLANER_TEMPLATE)"; \
		exit 1; \
	}
	@grep -q 'stundenplaner.conf.template' "$(GATEWAY_OVERRIDE)" || { \
		echo "Error: refusing to manage nginx because the gateway override omits Stundenplaner"; \
		exit 1; \
	}
	@docker network inspect study-gateway >/dev/null 2>&1 || { \
		echo "Error: external Docker network 'study-gateway' does not exist"; \
		exit 1; \
	}
	@$(GATEWAY_COMPOSE) config --quiet

vps-check-shop-artifacts:
	@test -f "$(STUDY_REPO_PATH)/.env" || { \
		echo "Error: study environment file not found at $(STUDY_REPO_PATH)/.env"; \
		exit 1; \
	}
	@test -s "$(STUDY_REPO_PATH)/DemoSiteV2/data/demosite.db" || { \
		echo "Error: frozen V2 database is missing or empty"; \
		exit 1; \
	}
	@test -s "$(STUDY_REPO_PATH)/OfflineTraining/outputs/ppo_policy.pt" || { \
		echo "Error: frozen V3 PPO checkpoint is missing or empty"; \
		exit 1; \
	}
	@test -s "$(STUDY_REPO_PATH)/OfflineTraining/outputs/trained_policy.json" || { \
		echo "Error: frozen V3 JSON policy is missing or empty"; \
		exit 1; \
	}
	@docker image inspect demosite-v3:latest >/dev/null 2>&1 || { \
		echo "Error: demosite-v3:latest is required to validate the frozen policies"; \
		exit 1; \
	}
	docker run --rm \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(STUDY_REPO_PATH):/workspace:ro" \
		-w /workspace \
		--entrypoint python \
		demosite-v3:latest \
		OfflineTraining/validate_policy_artifacts.py \
		--require-clean-provenance \
		--ppo-checkpoint OfflineTraining/outputs/ppo_policy.pt \
		--ppo-policy OfflineTraining/outputs/trained_policy.json \
		--bandit-db DemoSiteV2/data/demosite.db

vps-check-stundenplaner:
	@test -f "$(STUNDENPLANER_DIR)/docker-compose.vps.yml" || { \
		echo "Error: Stundenplaner compose file not found at $(STUNDENPLANER_DIR)/docker-compose.vps.yml"; \
		exit 1; \
	}
	@test -f "$(STUNDENPLANER_DIR)/.env" || { \
		echo "Error: Stundenplaner environment file not found at $(STUNDENPLANER_DIR)/.env"; \
		exit 1; \
	}
	@$(STUNDENPLANER_COMPOSE) config --quiet

update-shops: vps-check-gateway vps-check-shop-artifacts
	@echo "### Rebuilding DemoSiteV2 and DemoSiteV3 ..."
	$(STUDY_VPS_COMPOSE) up -d --build --no-deps v2 v3
	@echo "### Testing and reloading the shared nginx ..."
	$(GATEWAY_COMPOSE) exec -T nginx nginx -t
	$(GATEWAY_COMPOSE) exec -T nginx nginx -s reload
	$(STUDY_VPS_COMPOSE) ps v2 v3

update-stundenplaner: vps-check-gateway vps-check-stundenplaner
	@echo "### Rebuilding Stundenplaner ..."
	$(STUNDENPLANER_COMPOSE) up -d --build stundenplaner
	@echo "### Testing and reloading the shared nginx ..."
	$(GATEWAY_COMPOSE) exec -T nginx nginx -t
	$(GATEWAY_COMPOSE) exec -T nginx nginx -s reload
	$(STUNDENPLANER_COMPOSE) ps stundenplaner

update-magliosite: vps-check-gateway vps-check-stundenplaner
	@$(STUNDENPLANER_COMPOSE) ps --status running --services | \
		grep -qx 'stundenplaner' || { \
			echo "Error: Stundenplaner must be running before shared nginx is recreated"; \
			exit 1; \
		}
	@echo "### Rebuilding MaglioSite with the complete shared gateway override ..."
	$(GATEWAY_COMPOSE) up -d --build web nginx
	@echo "### Validating the recreated shared nginx ..."
	$(GATEWAY_COMPOSE) exec -T nginx nginx -t
	$(GATEWAY_COMPOSE) exec -T nginx nginx -s reload
	$(GATEWAY_COMPOSE) ps web nginx

# Audit-only release builder. It may produce a DRAFT_NOT_READY manifest, but it
# never overwrites an existing non-empty release directory and never claims
# recruitment readiness while any fail-closed check remains unresolved.
study-release-manifest:
	$(PYTHON) Experiments/build_study_release_manifest.py \
		--release-id $(STUDY_RELEASE_ID) \
		--baseline-json $(STUDY_BASELINE) \
		--baseline-csv $(STUDY_BASELINE_CSV) \
		--power-analysis $(STUDY_POWER_OUT) \
		--dispatcher-db $(STUDY_DISPATCHER_DB) \
		$(STUDY_IMAGE_ARGS)

study-baseline:
	$(PYTHON) Experiments/evaluate_study_policy_baseline.py \
		--study-id $(STUDY_BASELINE_ID) \
		--sessions-per-cell $(STUDY_SESSIONS_PER_CELL) \
		--context-sessions-per-stratum $(STUDY_CONTEXT_SESSIONS_PER_STRATUM) \
		--seed $(STUDY_SEED)

study-power:
	$(PYTHON) Experiments/calculate_clickworker_power.py \
		--baseline $(STUDY_BASELINE) \
		--out $(STUDY_POWER_OUT)

# Preregistration §7.1 requires the MACE null-calibration floors to be recomputed
# against the locked reference before recruitment, and again at the achieved
# sample size before unblinding.
study-mace-floors:
	$(PYTHON) Experiments/calculate_mace_floors.py \
		--baseline $(STUDY_BASELINE) \
		--out $(STUDY_MACE_FLOORS)

study-analysis:
	$(PYTHON) Experiments/analyze_clickworker_study.py \
		--db v2=DemoSiteV2/data/demosite.db \
		--db v3=DemoSiteV3/data/demosite.db \
		--dispatcher-db $(STUDY_DISPATCHER_DB) \
		--baseline $(STUDY_BASELINE) \
		--bootstrap $(STUDY_BOOTSTRAP) \
		--permutations $(STUDY_PERMUTATIONS) \
		--out-dir $(STUDY_ANALYSIS_OUT)

# ---------------------------------------------------------------------------
# Sequentiality sweep — preregistration v3
# ---------------------------------------------------------------------------
# Commit 33378f3 ("fix: align policy training and serving domains") changed the
# simulator, so the completed runs in sweep_results_confirmatory/,
# sweep_results_expose_history/ and sweep_results_offline_ppo/ describe an
# environment that no longer exists. preregistration_v3.md §G retains them as
# disclosed history — they may not be pooled with or substituted for the runs
# below — and §D/§F fix the re-run reproduced here verbatim.
#
# The runner fingerprints the script, archetype file, grid and effective
# settings into run_manifest.json, so re-invoking an identical command resumes
# from the durable cell cache and any mismatch fails closed instead of silently
# mixing environments. Every run is safe to interrupt after a completed cell.

# Wiring check, about a minute, into a throwaway directory. Worth running first:
# 33378f3 added a ValueError in _make_next_state for cart/checkout transitions
# that arrive with an empty cart.
sweep-v3-smoke:
	$(PYTHON) Experiments/run_sequentiality_sweep.py --smoke --out-dir $(SWEEP_V3_SMOKE_DIR)

sweep-v3-confirmatory:
	$(PYTHON) Experiments/run_sequentiality_sweep.py \
		--out-dir $(SWEEP_V3_CONFIRMATORY_DIR) \
		--seeds $(SWEEP_V3_SEEDS) \
		$(SWEEP_V3_ARGS)

sweep-v3-expose-history:
	$(PYTHON) Experiments/run_sequentiality_sweep.py \
		--out-dir $(SWEEP_V3_HISTORY_DIR) \
		--seeds $(SWEEP_V3_SEEDS) \
		--axes fatigue_rate --expose-history-to-bandit \
		$(SWEEP_V3_ARGS)

sweep-v3-offline-ppo:
	$(PYTHON) Experiments/run_sequentiality_sweep.py \
		--out-dir $(SWEEP_V3_OFFLINE_DIR) \
		--seeds $(SWEEP_V3_SEEDS) \
		--axes fatigue_rate delayed_reward_strength transition_coupling_strength \
		--ppo-mode offline \
		$(SWEEP_V3_ARGS)

# All three in the preregistered order — §F follow-ups run after the §D
# confirmatory grid. Expressed as prerequisites rather than a recursive make
# call so the recipe survives a CURDIR containing spaces or parentheses.
sweep-v3: sweep-v3-confirmatory sweep-v3-expose-history sweep-v3-offline-ppo

sweep-v3-status:
	@$(PYTHON) -c "import json,os,sys;[sys.stdout.write(d+': '+(json.dumps(json.load(open(d+'/run_status.json'))) if os.path.exists(d+'/run_status.json') else 'not started')+'\n') for d in sys.argv[1:]]" $(SWEEP_V3_CONFIRMATORY_DIR) $(SWEEP_V3_HISTORY_DIR) $(SWEEP_V3_OFFLINE_DIR)

# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

simulate-eval: simulate evaluate

# evaluate-split: train on seed=42 data, score on independent seed=99 data.
# Removes the in-sample bias where greedy_empirical is fit and scored on the
# same transitions.  Uses 80/20 session split (4000 train / 1000 eval).
evaluate-split:
	cd $(SIM_DIR) && $(CURDIR)/$(PYTHON) run_simulation.py \
		--n-sessions 4000 \
		--seed 42 \
		--output-dir ../$(OFFLINE_DIR)/data/train/
	cd $(SIM_DIR) && $(CURDIR)/$(PYTHON) run_simulation.py \
		--n-sessions 1000 \
		--seed 99 \
		--output-dir ../$(OFFLINE_DIR)/data/eval/
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) train_offline_policy.py \
		--dataset data/train/offline_transitions.jsonl \
		--gamma $(GAMMA) \
		--conservative-penalty $(CONSERVATIVE_PENALTY) \
		--out-dir outputs/split/
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) ope_eval.py \
		--dataset data/eval/offline_transitions.jsonl \
		--train-dataset data/train/offline_transitions.jsonl \
		--policy-file outputs/split/trained_policy.json \
		--temperature 1.0 \
		--out-dir outputs/split/
	@echo ""
	@echo "Done. Open OfflineTraining/outputs/split/report.html for the split-eval dashboard."

# evaluate-sim: ground-truth policy comparison via CustomerSimulation oracle.
# Runs N_SIM sessions each for trained policy, uniform-random baseline, and
# no-op baseline under the same archetype distribution.
N_SIM := 2000
evaluate-sim:
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) eval_policy_sim.py \
		--policy-file outputs/trained_policy.json \
		--n-sessions $(N_SIM) \
		--seed 7 \
		--out-dir outputs/
	@echo ""
	@echo "Done. Simulator comparison written to OfflineTraining/outputs/sim_eval_results.json"

# gen_report: regenerate HTML report from whatever artifacts currently exist
# in OfflineTraining/outputs (policy/dataset/OPE/simulator).
gen_report:
	cd $(OFFLINE_DIR) && $(CURDIR)/$(PYTHON) generate_html_report.py \
		--output-dir outputs/ \
		--policy-file outputs/trained_policy.json \
		--dataset-file outputs/dataset_summary.json \
		--ope-file outputs/ope_results.json \
		--sim-file outputs/sim_eval_results.json
	@echo ""
	@echo "Done. Open OfflineTraining/outputs/report.html to view the dashboard."
