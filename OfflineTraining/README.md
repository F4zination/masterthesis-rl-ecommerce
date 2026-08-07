# OfflineTraining

Train and evaluate offline reinforcement learning policies. Extract transitions from real or simulated data, train PPO actor-critic or conservative FQI policies, evaluate with OPE (IPS/SNIPS/DR/DM), and generate interactive dashboards.

**Pipeline:** Data Extraction → Policy Training (PPO or FQI) → Offline Evaluation → HTML Reports

## 📊 Overview

Five-stage workflow for offline policy learning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Historical Data (DemoSiteV2 or CustomerSimulation)                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ extract_offline_dataset.py   │
                  │ (1) Extract Transitions       │
                  │ JSONL, CSV, statistics       │
                  └────────────┬────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
  ┌────────────▼────────────┐   ┌──────────────▼──────────────┐
  │ train_ppo_policy.py      │   │ train_offline_policy.py      │
  │ (2a) Train PPO Policy    │   │ (2b) Train FQI Policy        │
  │ Neural actor-critic      │   │ Conservative tabular FQI     │
  │ ppo_policy.pt + JSON     │   │ trained_policy.json          │
  └────────────┬────────────┘   └──────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ ope_eval.py                  │
                  │ (3) Evaluate Policy          │
                  │ IPS, SNIPS, DR, DM + lift    │
                  └────────────┬────────────┘
                               │            \
                  ┌────────────▼────────────┐  ┌────────────────────────────┐
                  │ generate_html_report.py      │  │ eval_policy_sim.py          │
                  │ (4) Interactive Dashboard    │  │ (4b) Simulator Oracle       │
                  │ Algorithm-adaptive charts    │  │ Ground-truth comparison     │
                  └──────────────────────────┘  └────────────────────────────┘
```

## 🚀 Quick Start

### PPO Workflow (Primary): Simulate → Train → Evaluate

```bash
# Train PPO directly from simulated data (no separate extraction step)
python train_ppo_policy.py \
  --dataset ../CustomerSimulation/output/offline_transitions.jsonl \
  --timing-mode opportunity

# Evaluate policy performance
python ope_eval.py

# View interactive dashboard
# Open: outputs/report.html in web browser
```

### FQI Workflow (Tabular Fallback): Extract → Train → Evaluate

```bash
# Extract transitions from simulated data
python extract_offline_dataset.py --input ../CustomerSimulation/output/offline_transitions.jsonl

# Train tabular FQI policy
python train_offline_policy.py --gamma 0.99

# Evaluate policy performance
python ope_eval.py

# View interactive dashboard
# Open: outputs/report.html in web browser
```

### From Real Data (DemoSiteV2)

```bash
python extract_offline_dataset.py --db-path ../DemoSiteV2/demosite.db
python train_offline_policy.py
python ope_eval.py
```

### Compare Hyperparameters

```bash
# Conservative policy
python train_offline_policy.py --conservative-penalty 0.3 --output-dir outputs/conservative/

# Balanced policy
python train_offline_policy.py --conservative-penalty 0.1 --output-dir outputs/balanced/

# Aggressive policy
python train_offline_policy.py --conservative-penalty 0.0 --output-dir outputs/aggressive/

# Evaluate all
for dir in outputs/conservative outputs/balanced outputs/aggressive; do
  python ope_eval.py --policy-file $dir/trained_policy.json
done
```

## 🔧 Scripts

### `extract_offline_dataset.py` — Data Extraction

Reconstructs customer transitions from raw data. Mirrors reward logic from DemoSite decision services.

**Input:**
- SQLite database (DemoSiteV2: `demosite.db`)
- JSONL transitions (CustomerSimulation: `offline_transitions.jsonl`)
- CSV files (flat format)

**Output:**
```
data/
├── offline_transitions.jsonl      # One transition per line
├── offline_transitions_flat.csv   # Flattened with JSON columns
└── dataset_summary.json           # Metadata and statistics
```

**Key Operations:**
- Reconstructs `state` before and `next_state` after decision
- Normalizes features: device_type, traffic_source, *_bucket fields
- Computes rewards: `event_bonuses - action_costs`
- Attribution window: accumulate rewards within N seconds
- Produces OPE-compatible propensities

**Usage:**
```bash
python extract_offline_dataset.py \
  --input ../CustomerSimulation/output/offline_transitions.jsonl \
  --output-dir data/ \
  --attribution-window-seconds 1800
```

### `train_offline_policy.py` — Policy Training

Trains a tabular FQI policy from extracted transitions. Conservative by default (penalizes out-of-distribution actions to reduce compounding errors).

**Input:** `data/offline_transitions.jsonl` (required)

**Output:**
```
outputs/
├── trained_policy.json        # Deployable policy
├── v000001/                   # Versioned artifacts
│   ├── trained_policy.json
│   ├── dataset_summary.json
│   └── learner_run.json
└── active_version.txt         # Latest version
```

**Algorithm:**
- Fitted Q Iteration with conservative penalty
- Reads transitions, groups by (state, action, next_state)
- Iterative refinement: `Q ← mean(rewards + γ·max_a' Q(s', a'))`
- Penalty term: reduces Q-values for low-support (s, a) pairs
- Auto-generates HTML report with training stats

**Usage:**
```bash
python train_offline_policy.py \
  --input data/offline_transitions.jsonl \
  --gamma 0.99 \
  --iters 30 \
  --conservative-penalty 0.2 \
  --seed 42
```

**Hyperparameter Guidance:**
| Use Case | γ | iters | penalty |
|----------|---|-------|---------|
| Short-term (conversions) | 0.90 | 15 | 0.3 |
| Long-term (engagement) | 0.99 | 40 | 0.1 |
| Risk-averse | 0.95 | 25 | 0.5 |
| Exploratory | 0.90 | 20 | 0.0 |

### `train_ppo_policy.py` — PPO Actor-Critic Training

Trains a neural PPO actor-critic policy from extracted transitions. Generalizes to unseen states via function approximation (MLP). Requires **PyTorch**.

Works directly on simulation JSONL or extracted transitions — no separate `extract_offline_dataset.py` step is required.

**Input:** JSONL transitions (simulated or extracted)

**Output:**
```
outputs/
├── ppo_policy.pt              # PyTorch checkpoint (actor + critic weights)
├── trained_policy.json        # Metadata + default actions (for serving)
├── v000001/                   # Versioned artifacts
│   ├── ppo_policy.pt
│   ├── trained_policy.json
│   └── learner_run.json
└── active_version.txt         # Latest version pointer
```

Newly trained checkpoints and JSON exports embed a `provenance` block with the
dataset/config/trainer SHA-256 hashes, Git revision and dirty state, seed,
dynamics, horizon, training mode, and effective PPO run parameters. Newly
built V2 databases likewise store their dataset and builder provenance in the
`policy_build_metadata` table. Older artifacts without these fields should be
treated as legacy and accompanied by an external release manifest.

**Algorithm:**
- Bag-of-tokens binary state encoding from a token vocabulary built over the training data
- Actor-critic MLP (default: 128→64 hidden layers) for discrete action logits + scalar value
- GAE (Generalized Advantage Estimation) for advantage computation
- Clipped PPO surrogate loss + entropy bonus + value loss
- Mini-batch updates over multiple epochs per dataset pass

**Usage:**
```bash
python train_ppo_policy.py \
  --dataset ../CustomerSimulation/output/offline_transitions.jsonl \
  --timing-mode opportunity \
  --gamma 0.95 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --epochs 6 \
  --hidden-sizes 128,64
```

**Hyperparameter Guidance:**
| Concern | Suggestion |
|---------|------------|
| Too much policy change per step | Lower `--clip-eps` (e.g., 0.1) |
| Insufficient exploration | Raise `--entropy-coef` (e.g., 0.05) |
| Value estimates unstable | Lower `--value-coef` (e.g., 0.25) |
| Slow convergence | Raise `--lr` (e.g., 1e-3) or more `--epochs` |
| Gradient instability | Lower `--max-grad-norm` (e.g., 0.3) |

### `ope_eval.py` — Offline Policy Evaluation

Estimates policy performance using four OPE estimators without deploying. Compares trained policy vs. baselines (behavior, no-op, greedy). Supports a separate train/eval split to remove in-sample bias.

**Input:**
- Transition data: `data/offline_transitions.jsonl` (eval set)
- Policy (optional): `outputs/trained_policy.json`
- Train data (optional): `data/train/offline_transitions.jsonl` (if split evaluation)

**Output:**
```
outputs/
├── report.html            # Interactive dashboard
├── ope_results.json       # Numeric results with CIs
└── ope_report.md          # Markdown summary
```

**Estimators:**
- **IPS (Importance Sampling):** `Q_IPS = mean(w·r)` where `w = policy(a|s) / behavior(a|s)`  
  High variance; watch for extreme weights
- **SNIPS (Normalized IPS):** Stabilized version; lower variance
- **DR (Doubly Robust):** Combines model + behavior adjustment  
  Recommended; converges under mild conditions
- **DM (Direct Method):** `Q_DM = E_s[sum_a pi(a|s) Q_hat(s,a)]` — model-only, no importance weights.  
  Lower variance but higher bias; use alongside DR as a sanity check

**Normalized Lift:**  
After evaluating all policies, computes  
`(trained_DR - noop_DR) / (behavior_DR - noop_DR)`  
Values >1.0 mean the trained policy outperforms the behavior policy relative to the no-op floor. Printed to console and saved in `ope_results.json` as `lift_vs_noop_normalized`.

**Usage:**
```bash
# Standard evaluation (all data used for both fit and score)
python ope_eval.py \
  --dataset data/offline_transitions.jsonl \
  --policy-file outputs/trained_policy.json \
  --bootstrap 1000 \
  --seed 42

# Unbiased split evaluation (fit reward model + greedy on train, score on eval)
python ope_eval.py \
  --dataset data/eval/offline_transitions.jsonl \
  --train-dataset data/train/offline_transitions.jsonl \
  --policy-file outputs/trained_policy.json \
  --temperature 1.0
```

### `generate_html_report.py` — Interactive Dashboards

Auto-generates HTML dashboards from training/evaluation results. Called automatically by train/eval scripts. **Algorithm-adaptive**: auto-detects whether `trained_policy.json` was produced by PPO or FQI and renders the appropriate sections.

**Visualizations:**
- 📊 Dataset overview: n_transitions, n_sessions, conversion rate
- 👥 Archetype distribution (if available)
- 📈 Reward distribution & statistics
- 🎓 Training hyperparameters & configuration
  - PPO: γ, GAE λ, clip ε, entropy coef, value coef, LR, epochs, minibatch size
  - FQI: γ, FQI iterations, conservative penalty
- 🎯 Policy statistics
  - PPO: state vocabulary size, action vocabulary size, training sample count
  - FQI: Q-value distribution doughnut chart & state coverage
- 🔬 OPE comparison: IPS vs SNIPS vs DR vs DM
- 📉 Policy performance vs baselines + normalized lift card
- 🏷️ Train/Eval Split badge when `--train-dataset` was used

**Usage:**
```bash
# Auto-generated when training/evaluating:
python train_offline_policy.py      # → outputs/report.html
python ope_eval.py                  # → outputs/report.html (updated)

# Manual generation:
python generate_html_report.py \
  --policy-file outputs/trained_policy.json \
  --dataset-file data/dataset_summary.json \
  --ope-file outputs/ope_results.json
```

**Features:**
- ✅ Responsive design (mobile-friendly)
- ✅ Embedded Chart.js (no external server)
- ✅ Single static HTML file
- ✅ Bootstrap-style confidence intervals
- ✅ Color-coded metrics & sections

### `eval_policy_sim.py` — Simulator Oracle Evaluation

Ground-truth policy comparison via the CustomerSimulation engine. Runs the trained policy and baselines forward through the same archetype-driven environment and measures actual on-policy returns — no importance-weighting required.

Use this when OPE results are noisy or when you need confirmation outside of importance-sampled estimates.

**Input:**
- `outputs/trained_policy.json`
- `CustomerSimulation/config/archetypes.yaml`

**Output:**
```
outputs/
└── sim_eval_results.json    # Per-policy reward statistics
```

**Policies compared:**
- `trained_policy` — FQI policy loaded from JSON (greedy or softmax)
- `uniform_random` — Uniform random baseline (same as data-collection policy)
- `no_op` — Always selects no-op (minimum intervention baseline)

**Design note:** `StateAwareSimulator` subclasses `SessionSimulator` and calls `policy.set_state()` before each step so the trained policy can look up its Q-table without modifying the shared simulator.

**Usage:**
```bash
python eval_policy_sim.py \
  --policy-file outputs/trained_policy.json \
  --n-sessions 2000 \
  --seed 7

# Softmax sampling instead of greedy:
python eval_policy_sim.py \
  --policy-file outputs/trained_policy.json \
  --temperature 1.0 \
  --n-sessions 2000
```

## 📋 Command Reference

### extract_offline_dataset.py

```bash
python extract_offline_dataset.py \
  --input ../CustomerSimulation/output/offline_transitions.jsonl \
  --output-dir data/ \
  --attribution-window-seconds 1800
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | auto-detect | Path to JSONL/CSV transitions |
| `--db-path` | `../DemoSiteV2/demosite.db` | SQLite database path |
| `--output-dir` | `data/` | Output directory |
| `--attribution-window-seconds` | `1800` | Reward accumulation window |

### train_offline_policy.py

```bash
python train_offline_policy.py \
  --input data/offline_transitions.jsonl \
  --output-dir outputs/ \
  --gamma 0.99 \
  --iters 30 \
  --conservative-penalty 0.2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data/offline_transitions.jsonl` | Transition data |
| `--output-dir` | `outputs/` | Output directory |
| `--gamma` | `0.95` | Discount factor |
| `--iters` | `20` | FQI iterations |
| `--conservative-penalty` | `0.15` | OOD penalty (0–1) |
| `--seed` | None | Random seed |

### train_ppo_policy.py

```bash
python train_ppo_policy.py \
  --dataset ../CustomerSimulation/output/offline_transitions.jsonl \
  --out-dir outputs/ \
  --timing-mode opportunity \
  --gamma 0.95 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --entropy-coef 0.01 \
  --value-coef 0.5 \
  --lr 3e-4 \
  --epochs 6 \
  --minibatch-size 64 \
  --hidden-sizes 128,64
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/offline_transitions.jsonl` | Transition JSONL (simulated or extracted) |
| `--out-dir` | `outputs/` | Output directory |
| `--timing-mode` | `legacy` | `legacy` or `opportunity` (metadata label) |
| `--gamma` | `0.95` | Discount factor |
| `--gae-lambda` | `0.95` | GAE lambda for advantage estimation |
| `--clip-eps` | `0.2` | PPO clipping epsilon |
| `--entropy-coef` | `0.01` | Entropy bonus coefficient |
| `--value-coef` | `0.5` | Value loss coefficient |
| `--lr` | `3e-4` | Adam learning rate |
| `--epochs` | `6` | PPO epochs per dataset pass |
| `--minibatch-size` | `64` | Mini-batch size |
| `--max-grad-norm` | `0.5` | Gradient clipping norm |
| `--hidden-sizes` | `128,64` | Comma-separated MLP hidden layer sizes |

### ope_eval.py

```bash
python ope_eval.py \
  --dataset data/offline_transitions.jsonl \
  --policy-file outputs/trained_policy.json \
  --bootstrap 1000 \
  --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/offline_transitions.jsonl` | Eval transition data |
| `--train-dataset` | None | Separate training JSONL; when set, builds qhat and greedy_empirical on train data only (removes in-sample bias) |
| `--policy-file` | None | Policy to evaluate |
| `--temperature` | `0.0` | Softmax temperature for trained policy (0 = greedy; >0 = softmax, gives non-zero IPS weights) |
| `--out-dir` | `outputs/` | Output directory |
| `--bootstrap` | `200` | Bootstrap samples for CIs |
| `--seed` | `7` | Random seed |

### eval_policy_sim.py

```bash
python eval_policy_sim.py \
  --policy-file outputs/trained_policy.json \
  --n-sessions 2000 \
  --seed 7
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy-file` | required | Path to trained_policy.json |
| `--archetypes` | `../CustomerSimulation/config/archetypes.yaml` | Archetype config |
| `--n-sessions` | `2000` | Sessions per policy |
| `--temperature` | `0.0` | Softmax temperature (0 = greedy) |
| `--t-max` | `20` | Max steps per session |
| `--seed` | `7` | Random seed |
| `--out-dir` | `outputs/` | Output directory |

## 📂 Output Formats

### Policy JSON (FQI)

```json
{
  "algorithm": "tabular_fqi_conservative",
  "gamma": 0.99,
  "iters": 30,
  "conservative_penalty": 0.2,
  "n_states": 1500,
  "n_policy_states": 1420,
  "policy": {
    "<state_json>": "best_action",
    ...
  },
  "default_action_by_decision_point": {
    "landing": "no-op",
    "pdp": "trust_badge",
    ...
  },
  "q_table": {
    "<state>|||<action>": 1.234,
    ...
  }
}
```

### Policy JSON (PPO)

```json
{
  "algorithm": "ppo_clip_actor_critic",
  "timing_mode": "opportunity",
  "gamma": 0.95,
  "gae_lambda": 0.95,
  "clip_eps": 0.2,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "lr": 0.0003,
  "epochs": 6,
  "minibatch_size": 64,
  "n_rows": 1594,
  "n_states": 201,
  "state_vocab_size": 27,
  "action_vocab_size": 6,
  "checkpoint_path": "outputs/ppo_policy.pt",
  "policy": {},
  "default_action_by_decision_point": {
    "landing": "no-op",
    "pdp": "no-op",
    ...
  }
}
```

The PPO serving loader reads `checkpoint_path` to find the `.pt` file. The `policy` dict is empty for PPO — action selection uses the neural network, not a Q-table lookup.

### OPE Results JSON

```json
{
  "behavior_policy": {
    "ips": {"mean": 0.42, "std": 0.15, "ci95": [0.13, 0.71]},
    "snips": {"mean": 0.45, "std": 0.12, "ci95": [0.21, 0.69]},
    "dr": {"mean": 0.47, "std": 0.10, "ci95": [0.27, 0.67]}
  },
  "trained_policy": {
    "ips": {"mean": 0.58, "std": 0.18, "ci95": [0.23, 0.93]},
    ...
  }
}
```

### Dataset Summary JSON

```json
{
  "n_transitions": 15234,
  "n_sessions": 1250,
  "reward_mean": 0.52,
  "reward_min": -1.45,
  "reward_max": 11.54,
  "conversion_rate": 0.18,
  "avg_session_length": 4.98,
  "archetype_distribution": {
    "Explorer": 298,
    "FastBuyer": 201,
    ...
  },
  "generated_at": "2026-05-16T10:30:00"
}
```

## 🔗 Integration with DemoSiteV3

Trained policies are consumed automatically by [DemoSiteV3](../DemoSiteV3/) — no manual file copy needed. The app reads from `OfflineTraining/outputs/` via env-var paths at startup.

```bash
# 1. Train PPO policy
python train_ppo_policy.py --dataset ../CustomerSimulation/output/offline_transitions.jsonl

# 2. Start DemoSiteV3 — PPO checkpoint and tabular policy load automatically
cd ../DemoSiteV3
python -m uvicorn app.main:app --port 8000

# 3. Monitor policy health via analytics dashboard or API
curl http://localhost:8000/api/analytics/learner_status
```

DemoSiteV3 automatically:
- Loads `ppo_policy.pt` checkpoint on startup (PPO first)
- Loads `trained_policy.json` as tabular fallback
- Falls back to contextual bandit if both are invalid
- Enforces timing gates to prevent stale decisions
- Logs policy health and fallback chain at startup

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: offline_transitions.jsonl` | Run `extract_offline_dataset.py` first, or specify `--input` |
| Low OPE estimates (e.g., mean=0.1) | Policy may not improve over behavior; try lower penalty, more iterations, more data |
| High variance (wide CIs) | Increase `--bootstrap` samples (500 → 2000); try DR estimator instead of IPS |
| Policy not loading in DemoSiteV3 | Check `schema_version` matches CONTEXT_SCHEMA_VERSION (2); verify file is readable |
| Incorrect feature normalization | Ensure buckets match SharedSchema thresholds (page_depth=[1,3,6], etc.) |

## 🏗️ Architecture

### Data Pipeline

```
CustomerSimulation ──────┐
                         ├──→ extract_offline_dataset.py
DemoSiteV2/V3 ───────────┘
                         │
                    (JSONL/CSV)
                         │
                  ┌──────▼──────┐
                  │ train_offline │
                  │   _policy.py  │
                  └──────┬────────┘
                         │
                    (Policy JSON)
                         │
                  ┌──────▼──────┐
                  │  ope_eval.py │
                  └──────┬────────┘
                        │
                  ┌──────────┴──────────┐
                  ▼                     ▼
                ┌──────────────┐   ┌──────────────────┐
                │  ope_eval.py  │   │ eval_policy_sim.py│
                │ IPS/SNIPS/    │   │ Simulator oracle  │
                │ DR/DM + lift  │   │ ground-truth      │
                └──────┬────────┘   └──────────────────┘
                  │
                ┌──────▼──────┐
                │generate_html │
                │  _report.py  │
                └─────────────┘
              ```

### SharedSchema Coupling

Uses unified constants/features from [SharedSchema](../SharedSchema/):

```python
from shared_schema.constants import EVENT_REWARD, ACTION_COST, CONTEXT_SCHEMA_VERSION
from shared_schema.features import POINT_FEATURES, _bucket
```

**Benefit:** Policy states match DemoSite decision logic exactly; same training works for real and simulated data.

## 📚 References

- [Root README](../README.md) — Project overview
- [CustomerSimulation](../CustomerSimulation/) — Data generation
- [DemoSiteV2](../DemoSiteV2/) — Online bandit baseline
- [DemoSiteV3](../DemoSiteV3/) — Offline-first deployment
- [SharedSchema](../SharedSchema/) — Unified constants & features

### 3. `ope_eval.py` — Offline Policy Evaluation

Estimates policy performance using four off-policy evaluation (OPE) estimators. Compares trained policy against baselines without deploying.

**Input:** Transition data + policy to evaluate  
**Output:**
- `outputs/ope_results.json` — Numeric results with confidence intervals
- `outputs/ope_report.md` — Human-readable summary
- `outputs/report.html` — Interactive dashboard with visualizations

**Estimators:**
- **IPS (Importance Sampling):** `Q_IPS = E[w·r]` where `w = policy(a|s) / behavior(a|s)`  
  ⚠️ High variance; sensitive to propensity mismatches
- **SNIPS (Normalized IPS):** Stabilized version dividing by expected weights  
  ✓ Lower variance than IPS
- **DR (Doubly Robust):** Combines model and behavior adjustment  
  ✓ Optimal: converges under mild conditions
- **DM (Direct Method):** `Q_DM = E_s[sum_a pi(a|s) Q_hat(s,a)]` — model-only, no importance weights  
  ✓ Low variance but higher bias; use alongside DR as sanity check

**Normalized Lift:**  
Computes `(trained_DR - noop_DR) / (behavior_DR - noop_DR)`.  
Values >1.0 mean the trained policy outperforms the behavior policy relative to the no-op floor.

**Baselines:**
- **Behavior Policy:** Empirical action distribution (baseline for OPE)
- **No-Op:** Always choose "no-op" action
- **Greedy:** Best action in hindsight (empirical greedy)
- **Trained Policy:** Your FQI policy

**See Also:** Results help decide whether to deploy policy; compare confidence intervals.

### 4. `eval_policy_sim.py` — Simulator Oracle Evaluation

Ground-truth policy comparison via the CustomerSimulation engine. Bypasses OPE approximations entirely by rolling out policies directly through the simulator.

**Policies compared:** `trained_policy`, `uniform_random`, `no_op`  
**Output:** `outputs/sim_eval_results.json` — mean reward, std, approximate purchase rate, action distribution

**When to use:** When OPE estimates are noisy or when you want confirmation that offline improvements transfer to the simulated environment.

### 5. `generate_html_report.py` — Interactive Dashboards

Auto-generates beautiful HTML dashboards visualizing training and evaluation results. Called automatically by training and evaluation scripts.

**Input:** Policy JSON, dataset summary, OPE results  
**Output:** `outputs/report.html` — Interactive dashboard with charts

**Visualizations:**
- 📊 Dataset overview: transitions, sessions, conversion rate
- 👥 Archetype distribution (pie chart)
- 📈 Reward distribution histograms
- 🎓 Training hyperparameters and configuration
- 🎯 Policy statistics: Q-value distribution, state coverage
- 🔬 OPE comparison: IPS vs SNIPS vs DR
  - 🔬 OPE comparison: IPS vs SNIPS vs DR vs DM
  - 📉 Policy performance vs baselines + normalized lift card
  - 🏷️ Train/Eval Split badge when separate train data was used

**Usage:**
```bash
# Auto-generated when running train/eval scripts
# Or manually generate:
python generate_html_report.py \
  --output-dir outputs/ \
  --policy-file outputs/trained_policy.json \
  --dataset-file data/dataset_summary.json \
  --ope-file outputs/ope_results.json
```

**Features:**
- Responsive design (mobile-friendly)
- Embedded Chart.js for client-side rendering
- Bootstrap-style confidence intervals
- Color-coded sections and metrics
- No external dependencies (single static HTML file)

## Quick Start

### Workflow 1: Train from Simulated Data

```bash
# Generate 10k customer simulations
cd CustomerSimulation
python run_simulation.py --n-sessions 10000 --seed 42 --output-dir ../OfflineTraining/data/simulated/

# Extract transitions
cd ../OfflineTraining
python extract_offline_dataset.py --input data/simulated/offline_transitions.jsonl

# Train policy
python train_offline_policy.py --gamma 0.99

# Evaluate policy
python ope_eval.py

# View interactive dashboard
# Open: outputs/report.html in your web browser
```

### Workflow 2: Train from DemoSiteV2 Data

```bash
# Extract from DemoSiteV2 database
python extract_offline_dataset.py --db-path ../DemoSiteV2/demosite.db

# Train and evaluate
python train_offline_policy.py
python ope_eval.py
```

### Workflow 3: Compare Multiple Policies

```bash
# Train baseline policy
python train_offline_policy.py --conservative-penalty 0.0 --output-dir outputs/baseline/

# Train aggressive policy
python train_offline_policy.py --conservative-penalty 0.5 --output-dir outputs/aggressive/

# Evaluate both
python ope_eval.py --policy-file outputs/baseline/trained_policy.json
python ope_eval.py --policy-file outputs/aggressive/trained_policy.json
```

## Command-Line Reference

### extract_offline_dataset.py

Extract transitions from historical data.

```bash
python extract_offline_dataset.py \
  --input ../CustomerSimulation/output/offline_transitions.jsonl \
  --output-dir data/ \
  --attribution-window-seconds 1800
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (see below) | Path to JSONL or CSV transitions. Auto-detects DemoSiteV2 DB if omitted. |
| `--db-path` | `../DemoSiteV2/demosite.db` | SQLite database (used if `--input` not provided) |
| `--output-dir` | `data/` | Directory for output JSONL, CSV, summary |
| `--attribution-window-seconds` | `1800` | Seconds after decision to accumulate rewards |

**Input Options:**
- **JSONL/CSV from CustomerSimulation:** `--input ../CustomerSimulation/output/offline_transitions.jsonl`
- **SQLite from DemoSiteV2:** `--db-path ../DemoSiteV2/demosite.db` (default)
- **CSV files:** `--input data/offline_transitions_flat.csv`

**Output Files:**
- `offline_transitions.jsonl` — One transition per line; includes metadata
- `offline_transitions_flat.csv` — State/action/reward as columns
- `dataset_summary.json` — n_transitions, n_sessions, reward stats, archetype distribution

### train_offline_policy.py

Train a tabular FQI policy.

```bash
python train_offline_policy.py \
  --input data/offline_transitions.jsonl \
  --output-dir outputs/ \
  --gamma 0.99 \
  --iters 30 \
  --conservative-penalty 0.2 \
  --epsilon-train 0.1 \
  --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data/offline_transitions.jsonl` | Path to transition JSONL/CSV |
| `--output-dir` | `outputs/` | Directory for policy and run metadata |
| `--gamma` | `0.9` | Discount factor (0.9–0.99; higher = long-term focus) |
| `--iters` | `20` | FQI iterations (10–50; more = better convergence) |
| `--conservative-penalty` | `0.1` | Penalty for out-of-distribution actions (0–1) |
| `--epsilon-train` | `0.1` | Epsilon-greedy during training (0–0.3) |
| `--seed` | `None` | Random seed for reproducibility |

**Output:**
- `trained_policy.json` — Deployable policy
- `v000NNN/` — Versioned directory with policy, summary, config
- `active_version.txt` — Latest version number

**Typical Hyperparameters:**
| Use Case | γ | iters | conservative_penalty |
|----------|---|-------|----------------------|
| Short-term conversions | 0.9 | 15 | 0.3 |
| Long-term engagement | 0.99 | 40 | 0.1 |
| Risk-averse | 0.95 | 25 | 0.5 |
| Exploratory | 0.9 | 20 | 0.0 |

### ope_eval.py

Evaluate policy performance offline using multiple estimators.

```bash
python ope_eval.py \
  --input data/offline_transitions.jsonl \
  --policy-file outputs/trained_policy.json \
  --output-dir outputs/ \
  --bootstrap 1000 \
  --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data/offline_transitions.jsonl` | Transition data |
| `--policy-file` | `outputs/trained_policy.json` | Policy to evaluate (or None for behavior only) |
| `--output-dir` | `outputs/` | Directory for OPE results and report |
| `--bootstrap` | `500` | Bootstrap samples for confidence intervals |
| `--seed` | `None` | Random seed |

**Output:**
- `ope_results.json` — Numeric results: mean, std, 95% CI for each estimator
- `ope_report.md` — Markdown summary with interpretation

**Example Output:**
```json
{
  "behavior_policy": {
    "ips": {"mean": 0.42, "std": 0.15, "ci_lower": 0.13, "ci_upper": 0.71},
    "snips": {"mean": 0.45, "std": 0.12, "ci_lower": 0.21, "ci_upper": 0.69},
    "dr": {"mean": 0.47, "std": 0.10, "ci_lower": 0.27, "ci_upper": 0.67}
  },
  "trained_policy": {
    "ips": {"mean": 0.58, "std": 0.18, "ci_lower": 0.23, "ci_upper": 0.93},
    ...
  }
}
```

## Output Formats

### Policy JSON

```json
{
  "schema_version": 2,
  "timestamp": "2026-05-16T10:30:00",
  "hyperparameters": {
    "gamma": 0.99,
    "conservative_penalty": 0.1,
    "iterations": 30
  },
  "decision_points": {
    "landing": {
      "mobile,direct,0": {"no-op": 1.2, "trending_carousel": 1.5, ...},
      "mobile,search,0": {"no-op": 0.8, "discount_banner": 1.3, ...},
      ...
    },
    "pdp": {...},
    ...
  },
  "defaults": {
    "landing": "no-op",
    "pdp": "trust_badge",
    ...
  }
}
```

**Structure:**
- `schema_version`: Matches `CONTEXT_SCHEMA_VERSION` from SharedSchema
- `decision_points`: Q-values per (state, action) pair
- `defaults`: Fallback action if state not in policy

### Run Metadata

```json
{
  "version": "v000001",
  "timestamp": "2026-05-16T10:30:00",
  "hyperparameters": {...},
  "dataset_stats": {
    "n_transitions": 15234,
    "n_sessions": 1250,
    "reward_mean": 0.52,
    "reward_std": 1.2
  },
  "training_results": {
    "convergence_iterations": 18,
    "final_coverage": 0.87
  }
}
```

## Integration with DemoSiteV3

Trained policies are consumed automatically by DemoSiteV3 — no manual file copy needed:

1. **Train PPO policy:**
   ```bash
   python train_ppo_policy.py --dataset ../CustomerSimulation/output/offline_transitions.jsonl
   ```

2. **Start DemoSiteV3 — policies load automatically from `OfflineTraining/outputs/`:**
   ```bash
   cd ../DemoSiteV3
   python -m uvicorn app.main:app --port 8000
   ```

3. **DemoSiteV3 loads automatically:**
   - Reads `ppo_policy.pt` and `trained_policy.json` from `outputs/` at startup
   - Serving chain: PPO → tabular → bandit fallback
   - See `DemoSiteV3/app/services/ppo.py` and `decision.py` for details

4. **Monitor policy health:**
   - GET `/api/analytics/learner_status` returns PPO checkpoint health + learner status
   - Logs policy load/fallback events at startup

## Troubleshooting

**Issue: `FileNotFoundError: No such file or directory: 'data/offline_transitions.jsonl'`**
- Run `extract_offline_dataset.py` first
- Or specify custom path: `python train_offline_policy.py --input custom_path.jsonl`

**Issue: Low OPE estimates (e.g., 0.1 mean)**
- Policy may not be better than behavior policy
- Check: `python ope_eval.py` to compare estimators
- Try: Lower conservative penalty, more training iterations, more data

**Issue: High OPE variance (wide confidence intervals)**
- Too few bootstrap samples: increase `--bootstrap` (default 500 → try 2000)
- Too little data: more transitions reduce IPS variance
- Try DR estimator (more stable than IPS)

**Issue: Policy not loading in DemoSiteV3**
- Verify `schema_version` in policy JSON matches `CONTEXT_SCHEMA_VERSION` (2)
- Check DemoSiteV3 logs: `docker logs demosite_v3` or app console
- Ensure `OfflineTraining/outputs/ppo_policy.pt` and `trained_policy.json` are readable

**Issue: Stale data/incorrect states**
- Verify `--attribution-window-seconds` matches your scenario (default 1800 = 30 min)
- Check dataset: `python -c "import json; print(json.load(open('data/dataset_summary.json')), indent=2)"`
- Ensure feature buckets match SharedSchema: device_type, traffic_source, page_depth_bucket, etc.

## Architecture Notes

### Coupling with SharedSchema

All policies use state/action schema defined in `SharedSchema`:

```python
# Policy evaluation uses these constants
from shared_schema.constants import EVENT_REWARD, ACTION_COST, CONTEXT_SCHEMA_VERSION
from shared_schema.features import POINT_FEATURES
```

**Benefits:**
- Policy states are normalized the same way as DemoSiteV2/V3 decisions
- Changing a feature or reward ripples through entire system
- Offline training reflects online bandit behavior exactly

### Dataset Pipeline

```
CustomerSimulation ──────┐
                         ├──→ extract_offline_dataset.py ──→ train_offline_policy.py ──→ DemoSiteV3
DemoSiteV2 (or V3) ──────┘
```

Both sources produce `offline_transitions.jsonl` with identical schema:
```json
{
  "trajectory_id": "sim_abc123...",
  "state": {"decision_point": "landing", "device_type": "mobile", ...},
  "action": "discount_banner",
  "propensity": 0.166666,
  "reward": 0.85,
  "next_state": {...},
  "done": false
}
```

Extraction is source-agnostic; same training pipeline works for both simulated and real data.

## References

- [CustomerSimulation README](../CustomerSimulation/README.md) — Data generation
- [DemoSiteV2 README](../DemoSiteV2/README.md) — Online bandit baseline
- [DemoSiteV3 README](../DemoSiteV3/README.md) — Offline-first deployment
- [SharedSchema](../SharedSchema/) — Unified constants and features
