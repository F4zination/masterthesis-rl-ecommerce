# Master Thesis: Offline Reinforcement Learning for E-Commerce Personalization

Complete end-to-end pipeline for training and deploying offline RL policies in e-commerce environments. Includes customer simulation, offline policy learning (FQI), evaluation (OPE), and deployment options.


## 📁 Project Structure

```
MasterThesisProject/
├── Concept/                      # Design notes, thesis direction, pipeline concepts
├── Experiments/                  # Structured experiment specs and execution protocols
│
├── SharedSchema/                 # Unified constants & features
│   └── shared_schema/
│       ├── constants.py          # Rewards, costs, epsilons
│       ├── features.py           # State schema, feature extractors
│       └── __init__.py           # Public exports
│
├── DemoSiteV1/                   # Phase 1 baseline (no bandit logic, superseded)
│   └── app/
│
├── DemoSiteV2/                   # Online epsilon-greedy bandit
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── services/decision.py # Contextual bandit logic
│   │   └── ...
│   └── demosite.db
│
├── DemoSiteV3/                   # PPO-first + offline fallback
│   ├── app/
│   │   ├── main.py              # FastAPI app with health checks
│   │   ├── services/decision.py # PPO → offline tabular → bandit chain
│   │   ├── services/ppo.py      # PPO checkpoint loader & inference
│   │   └── ...
│   └── demosite.db
│
├── CustomerSimulation/           # Configurable customer journey simulator
│   ├── config/
│   │   └── archetypes.yaml      # 5 archetypes (Explorer, FastBuyer, etc.)
│   ├── simulation/
│   │   ├── archetype.py         # Archetype dataclass + loader
│   │   ├── state.py             # Customer state tracking
│   │   ├── behavior_policy.py   # Uniform random policy (max diversity)
│   │   ├── simulator.py         # Session simulation engine
│   │   └── writer.py            # JSONL/CSV output writer
│   ├── run_simulation.py         # CLI entry point
│   └── output/                   # Generated datasets
│       ├── offline_transitions.jsonl
│       ├── offline_transitions_flat.csv
│       └── dataset_summary.json
│
├── OfflineTraining/              # PPO + FQI training, OPE evaluation
│   ├── extract_offline_dataset.py
│   ├── train_ppo_policy.py       # PPO actor-critic (primary)
│   ├── train_offline_policy.py   # Tabular FQI (fallback / alternative)
│   ├── ope_eval.py
│   ├── generate_html_report.py
│   ├── data/                     # Extracted transitions
│   └── outputs/                  # Trained policies + reports
│       ├── report.html           # Interactive dashboard
│       ├── ppo_policy.pt         # PPO checkpoint
│       ├── trained_policy.json
│       └── v000NNN/              # Versioned runs
```

## 🚀 Quick Start

### Workflow 1: Simulate → Train → Deploy

Generate synthetic data, train a PPO policy, and deploy to DemoSiteV3.

**With Make (recommended):**
```bash
make simulate-eval          # generate data + full training pipeline
make demo-v3                # start DemoSiteV3 with the trained policy
```

**Manually:**
```bash
# 1. Generate 10,000 simulated customer journeys
cd CustomerSimulation
python run_simulation.py --n-sessions 10000 --seed 42 --output-dir ../OfflineTraining/data/simulated/

# 2. Train PPO policy from simulated data
cd ../OfflineTraining
python train_ppo_policy.py \
  --dataset data/simulated/offline_transitions.jsonl \
  --timing-mode opportunity

# 3. (Optional) Also train a tabular FQI fallback
python train_offline_policy.py --gamma 0.99 --iters 30 --conservative-penalty 0.2

# 4. Evaluate policy offline
python ope_eval.py

# 5. View interactive dashboard
# Open: outputs/report.html in web browser

# 6. Start DemoSiteV3 — PPO and tabular policies load automatically from outputs/
cd ../DemoSiteV3
python -m uvicorn app.main:app --reload --port 8000
# Startup logs confirm: PPO checkpoint loaded, offline policy loaded
```

### Workflow 2: Train from Real Data

Extract transitions from DemoSiteV2 and train a policy:

```bash
# 1. Ensure DemoSiteV2 DB is on the latest schema
cd DemoSiteV2
python -m shared_schema.migrations run --db-path ./demosite_test.db

# 2. Extract transitions and train
cd ../OfflineTraining
python extract_offline_dataset.py --db-path ../DemoSiteV2/demosite_test.db
python train_offline_policy.py
python ope_eval.py
# Open outputs/report.html
```

### Workflow 3: Compare Multiple Configurations

Train policies with different hyperparameters and evaluate:

```bash
cd OfflineTraining

# Conservative policy
python train_offline_policy.py --conservative-penalty 0.3 --output-dir outputs/conservative/

# Balanced policy
python train_offline_policy.py --conservative-penalty 0.1 --output-dir outputs/balanced/

# Aggressive policy
python train_offline_policy.py --conservative-penalty 0.0 --output-dir outputs/aggressive/

# Evaluate all
python ope_eval.py --policy-file outputs/conservative/trained_policy.json
python ope_eval.py --policy-file outputs/balanced/trained_policy.json
python ope_eval.py --policy-file outputs/aggressive/trained_policy.json
```

## 🔧 Component Documentation

### [Concept](Concept/) — Design Notes & Thesis Direction

Markdown documents covering the theoretical pipeline, bandit deployment concepts, MDP formulation, and evolving thesis direction. Not executable — reference material for understanding design decisions.

### [Experiments](Experiments/) — Structured Evaluation Protocols

Experiment definitions for quantifiable validation runs. Start with:
- [Clickworker Bandit vs RL Experiment](Experiments/Clickworker_Bandit_vs_RL_Experiment.md)



### [SharedSchema](SharedSchema/) — Unified Configuration

Central repository for all constants, feature schemas, pure helper functions, and versioned database migrations. Used by all downstream components to ensure consistency.

**Exports:**
- `EPSILON`, `PRIOR_COUNT`, `PRIOR_MEAN` — Bandit hyperparameters
- `ACTION_COST` dict — Cost per action
- `EVENT_REWARD` dict — Reward per event (purchase, add_to_cart, etc.)
- `CONTEXT_SCHEMA_VERSION` — State schema version
- `POINT_FEATURES` dict — Features per decision point
- `_normalize_context()`, `_eligible_actions()`, `_event_reward()` — Pure helpers

**Usage:**
```python
from shared_schema.constants import EVENT_REWARD, ACTION_COST
from shared_schema.features import POINT_FEATURES, _eligible_actions
```

**Migration CLI:**
```bash
python -m shared_schema.migrations status --db-path DemoSiteV2/demosite_test.db
python -m shared_schema.migrations run --db-path DemoSiteV2/demosite_test.db
```

See [SharedSchema README](SharedSchema/README.md).

### [DemoSiteV1](DemoSiteV1/) — Phase 1 Baseline

The original Phase 1 shop with no personalization logic. All decision-point hooks return a no-op stub. Superseded by DemoSiteV2 (online bandit) and DemoSiteV3 (offline-first). Kept as a reference for the un-personalized control baseline.

**Usage:**
```bash
make demo-v1            # http://127.0.0.1:8001
# or manually:
cd DemoSiteV1 && uvicorn app.main:app --port 8001
```

See [DemoSiteV1 README](DemoSiteV1/README.md).

### [DemoSiteV2](DemoSiteV2/) — Online Epsilon-Greedy Baseline

FastAPI e-commerce demo implementing contextual epsilon-greedy bandit for action selection. Logs all decisions and events to SQLite for offline analysis.

**Key Features:**
- 5 decision points (landing, pdp, cart, checkout, scroll_engagement)
- 6 actions per point (no-op, trending_carousel, discount_banner, etc.)
- Contextual features: device_type, traffic_source, page_depth, etc.
- SQLAlchemy ORM + SQLite database
- Real-time reward accumulation

**Usage:**
```bash
cd DemoSiteV2
python -m uvicorn app.main:app --port 8000
```

See [DemoSiteV2 README](DemoSiteV2/README.md).

### [DemoSiteV3](DemoSiteV3/) — PPO-First + Offline Fallback

Enhanced version of DemoSiteV2 that runs a PPO actor-critic policy first, falls back to a trained offline tabular policy, and finally falls back to a contextual bandit. Health checks and timing gates prevent stale or invalid policy deployment.

**Key Features:**
- PPO actor-critic serving from `OfflineTraining/outputs/ppo_policy.pt`
- Offline tabular policy fallback from `trained_policy.json`
- Timing gate: enforces min. time between decisions per session
- Graceful fallback chain: PPO → offline tabular → epsilon-greedy bandit
- Startup health logging with explicit fallback status
- Background learner that retrains PPO on live data on a configurable interval
- Analytics dashboard at `/analytics` with policy source breakdown, PPO checkpoint health panel, and timing diagnostics

**Usage:**
```bash
cd DemoSiteV3
python -m uvicorn app.main:app --port 8000
# PPO checkpoint and offline policy load automatically on startup
```

See [DemoSiteV3 README](DemoSiteV3/README.md).

### [CustomerSimulation](CustomerSimulation/) — Synthetic Data Generation

Generates realistic customer journeys with 5 configurable user archetypes (Explorer, FastBuyer, DetailedComparator, DiscountHunter, WindowShopper). Archetype probabilities and behaviors are fully tunable via `config/archetypes.yaml`.

**Features:**
- Markov-chain stage transitions per archetype
- Independent event sampling (exit_intent, add_to_cart, purchase, etc.)
- Action-specific widget effects (click/dismiss probabilities)
- Per-action additive lifts to base event rates
- Automatic state normalization matching DemoSite schema

**Configuration:**
```bash
# Edit config/archetypes.yaml to tune:
# - Archetype mixture weights
# - Stage transition probabilities
# - Event base rates
# - Widget interaction rates
# - Action-event interaction lifts
```

**Usage:**
```bash
cd CustomerSimulation
python run_simulation.py \
  --n-sessions 5000 \
  --seed 42 \
  --output-dir output/
# Generates: offline_transitions.jsonl, .csv, dataset_summary.json
```

See [CustomerSimulation README](CustomerSimulation/README.md).

### [OfflineTraining](OfflineTraining/) — Policy Learning & Evaluation

Offline RL pipeline supporting both **PPO actor-critic** and **tabular FQI** training, with OPE evaluation and algorithm-aware interactive HTML reports.

**Stages:**

1. **extract_offline_dataset.py** — Extract transitions from DemoSiteV2 DB or simulated data
   - Reconstructs state before/after each decision
   - Computes rewards: event_bonuses - action_costs
   - Attribution window: accumulate rewards within N seconds
   - Output: JSONL, CSV, summary stats

2. **train_ppo_policy.py** — Train PPO actor-critic policy (primary, requires PyTorch)
   - Neural MLP (default: 128→64 hidden layers)
   - GAE advantage estimation + clipped PPO loss + entropy bonus
   - Works directly on simulation JSONL — no separate extraction step required
   - Produces `ppo_policy.pt` checkpoint + `trained_policy.json` metadata

3. **train_offline_policy.py** — Train tabular FQI policy (alternative)
   - Conservative penalty reduces out-of-distribution actions
   - Fitted Q Iteration with iterative refinement
   - Produces `trained_policy.json` with Q-table for tabular lookup

4. **ope_eval.py** — Offline Policy Evaluation
   - Estimates performance using IPS, SNIPS, DR, DM estimators
   - Bootstrap confidence intervals (default 500 samples)
   - Compares against behavior policy, no-op, greedy baselines

5. **generate_html_report.py** — Interactive Dashboards
   - Auto-detects algorithm (PPO or FQI) from `trained_policy.json`
   - PPO: shows architecture summary, training hyperparams, vocab stats
   - FQI: shows Q-value distribution and state coverage chart
   - Single static HTML file with embedded Chart.js

**Usage:**
```bash
cd OfflineTraining
# PPO path (primary)
python train_ppo_policy.py --dataset ../CustomerSimulation/output/offline_transitions.jsonl
# FQI path (alternative / tabular fallback)
python extract_offline_dataset.py --input ../CustomerSimulation/output/offline_transitions.jsonl
python train_offline_policy.py --gamma 0.99 --iters 30
# Evaluate and view report
python ope_eval.py
# Open outputs/report.html
```

See [OfflineTraining README](OfflineTraining/README.md).


## 📊 End-to-End Example

Complete workflow from simulation to deployment (PPO path):

```bash
# 1. Generate simulated training data
cd CustomerSimulation
python run_simulation.py --n-sessions 10000 --seed 42

# 2. Train PPO policy from simulated data (no separate extraction step needed)
cd ../OfflineTraining
python train_ppo_policy.py \
  --dataset ../CustomerSimulation/output/offline_transitions.jsonl \
  --timing-mode opportunity

# 3. (Optional) Evaluate policy offline
python ope_eval.py --bootstrap 1000 --seed 42
# → outputs/report.html  (interactive dashboard, PPO-aware sections)
# → outputs/ope_results.json

# 4. Start DemoSiteV3 — policies load automatically from OfflineTraining/outputs/
cd ../DemoSiteV3
python -m uvicorn app.main:app --port 8000
# Startup logs confirm: PPO checkpoint loaded, fallback chain active

# 5. Monitor policy health and source breakdown at /analytics
# 6. Collect live data; background learner retrains automatically every 6h (LEARNER_ENABLED=true)
```

## 🔗 Key Coupling Points

### Shared Schema

All components reference `SharedSchema` for consistency:

```python
# DemoSiteV2 & V3 decision services
from shared_schema.constants import EPSILON, ACTION_COST, EVENT_REWARD
from shared_schema.features import POINT_FEATURES, _eligible_actions

# OfflineTraining extraction
from shared_schema.constants import CONTEXT_SCHEMA_VERSION, EVENT_REWARD

# CustomerSimulation state normalization
from shared_schema.features import POINT_FEATURES, _bucket
```

**Benefit:** Changing a reward value or feature definition in SharedSchema automatically propagates to all systems—no manual syncing needed.

### State Schema

All systems use the same normalized state structure:

```json
{
  "decision_point": "landing",
  "schema_version": 2,
  "device_type": "mobile",
  "traffic_source": "search",
  "page_depth_bucket": "low"
}
```

**Benefit:** Policies trained on CustomerSimulation data can be deployed directly to DemoSiteV3 without conversion.

### Transition Format

DemoSiteV2, CustomerSimulation, and OfflineTraining all produce identical transition format:

```json
{
  "trajectory_id": "...",
  "state": {...},
  "action": "discount_banner",
  "propensity": 0.166666,
  "reward": 0.85,
  "next_state": {...},
  "done": false
}
```

**Benefit:** Same training pipeline works for real and simulated data.

## 📈 Typical Workflows

### Development & Testing
Use CustomerSimulation to test new policies without real data:
1. Adjust archetype probabilities in `config/archetypes.yaml`
2. Run simulation
3. Train policy
4. Evaluate with OPE
5. Review HTML report

### Production Training
Use real data from DemoSiteV2:
1. Extract transitions from DemoSiteV2 database
2. Train policy with conservative penalty (0.1–0.3)
3. Evaluate with OPE (DR estimator recommended)
4. If improvement > threshold, deploy to DemoSiteV3

### Continuous Improvement Loop
```
DemoSiteV3 (collect data) → Extract → Train → Evaluate → Deploy
↑                                                           ↓
└───────────────────────────────────────────────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip + venv

### Quick Setup
```bash
# Clone/extract repository
cd MasterThesisProject

# Create virtual environment
python -m venv .venv

# Activate — Linux/macOS
source .venv/bin/activate
# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e SharedSchema/
pip install fastapi uvicorn sqlalchemy pyyaml
pip install torch  # required for PPO training and serving

# Verify installation
python -c "from shared_schema import POINT_FEATURES; print('OK')"
```

### Makefile Shortcuts

A top-level `Makefile` provides convenient commands for all common tasks.
Run `make` (or `make help`) to see all targets.

| Command | Description |
|---|---|
| `make demo-v1` | Start DemoSiteV1 on http://127.0.0.1:8001 |
| `make demo-v2` | Start DemoSiteV2 on http://127.0.0.1:8002 |
| `make demo-v3` | Start DemoSiteV3 on http://127.0.0.1:8003 |
| `make simulate` | Generate simulated customer data (default: 5000 sessions) |
| `make evaluate` | Extract → train → OPE-evaluate → HTML report |
| `make simulate-eval` | Run simulation then evaluate in one step |

Override simulation defaults: `make simulate N_SESSIONS=10000 SEED=99`

> **Windows users:** use GNU make via Git Bash, WSL, or install it with [Chocolatey](https://chocolatey.org/) (`choco install make`).

### Docker (Optional)
```bash
# Build and run DemoSiteV3
cd DemoSiteV3
docker build -t demosite-v3 .
docker run -p 8000:8000 demosite-v3
```

## 📚 References & Documentation

- [SharedSchema README](SharedSchema/README.md) — Constants, features, and migrations
- [DemoSiteV1 README](DemoSiteV1/README.md) — Phase 1 baseline (no bandit)
- [DemoSiteV2 README](DemoSiteV2/README.md) — Online epsilon-greedy bandit
- [DemoSiteV3 README](DemoSiteV3/README.md) — PPO-first serving + background learner
- [CustomerSimulation README](CustomerSimulation/README.md) — Synthetic data generation
- [OfflineTraining README](OfflineTraining/README.md) — Policy learning & evaluation

## 📝 License

Master's Thesis Project — All rights reserved.

## 🎓 Citation

For thesis references, cite this project as:

```bibtex
@thesis{rehnert26_e-commerce_personalization,
  author = {Rehnert, Finn},
  title = {Offline Reinforcement Learning for E-Commerce Personalization},
  school = {University of Applied Sciences Ulm},
  year = {2026},
  url = {URL}
}
```
