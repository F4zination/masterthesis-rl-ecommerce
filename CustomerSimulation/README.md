# CustomerSimulation

Generates simulated customer journey data for offline reinforcement learning training. Produces transitions compatible with `OfflineTraining/train_offline_policy.py` for immediate policy training.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate 5000 simulated sessions
python run_simulation.py --n-sessions 5000 --seed 42

# Output appears in output/ directory
cat output/dataset_summary.json
```

## How It Uses SharedSchema

CustomerSimulation leverages **SharedSchema** for single-source-of-truth architecture:

### 1. Unified Constants

All scalar hyperparameters come from `shared_schema.constants`:

```python
from shared_schema.constants import (
    EPSILON,                    # 0.2 (exploration rate)
    PRIOR_COUNT,               # 5.0 (bandit prior)
    PRIOR_MEAN,                # 0.0
    CONTEXT_SCHEMA_VERSION,    # 2 (for state dicts)
    ACTION_COST,               # {"discount_banner": 0.1, ...}
    EVENT_REWARD,              # {"purchase": 8.0, "add_to_cart": 1.5, ...}
)
```

**Why this matters:** If you adjust reward values or action costs for policy training, update `SharedSchema/shared_schema/constants.py` once, and both:
- DemoSiteV2 (online bandit)
- DemoSiteV3 (offline-first)
- CustomerSimulation (offline data gen)

...automatically use the new values. No copy-paste inconsistencies.

### 2. Feature Schema

`shared_schema.features` defines which features are available per decision point:

```python
from shared_schema.features import (
    POINT_FEATURES,           # {"landing": ["device_type", "traffic_source", ...], ...}
    _eligible_actions,        # (decision_point) -> list of valid actions
    _event_reward,            # (event_name, metadata) -> float
)
```

**Key benefit:** When you add a new feature (e.g., `user_segment`):
1. Update `POINT_FEATURES` in `shared_schema/features.py`
2. Update the corresponding `_FEATURE_EXTRACTORS` bucket thresholds
3. Simulation automatically generates it; DemoSites can consume it immediately

No changes needed in simulation code or decision services.

### 3. State Normalization

`SimState.to_state_dict()` uses `POINT_FEATURES` to auto-select relevant features:

```python
from simulation.state import SimState

state = SimState(
    decision_point="pdp",
    device_type="mobile",
    price=45.5,
    page_depth=2,
)

# Produces:
# {
#   "decision_point": "pdp",
#   "schema_version": 2,
#   "device_type": "mobile",
#   "price_bucket": "medium",
#   "page_depth_bucket": "low",
# }
# Only features in POINT_FEATURES["pdp"] are included.
```

**Advantage:** If you change POINT_FEATURES, generated state dicts automatically match. Train/eval pipelines stay in sync.

## Configuration

All user behaviors and probabilities live in `config/archetypes.yaml`:

```yaml
mixture_prior:
  Explorer: 0.30           # 30% of sessions are explorers
  FastBuyer: 0.20
  DiscountHunter: 0.15
  # ...

archetypes:
  Explorer:
    stage_transitions:
      landing:
        pdp: 0.70          # From landing, 70% go to product page
        done: 0.20
    base_events:
      landing:
        exit_intent: 0.12  # 12% chance to exit
    widget_events:
      discount_banner:
        widget_click: 0.05
    action_event_lifts:
      discount_banner:
        add_to_cart: 0.02  # +2% add-to-cart rate when discount shown
```

### Adding a New Archetype

1. Add name and weight to `mixture_prior`:
   ```yaml
   mixture_prior:
     VIP_Customer: 0.10
   ```

2. Define the archetype:
   ```yaml
   archetypes:
     VIP_Customer:
       stage_transitions: { ... }
       base_events: { ... }
       widget_events: { ... }
       action_event_lifts: { ... }
       order_total_range: [100, 500]
   ```

3. Re-run simulation:
   ```bash
   python run_simulation.py --n-sessions 1000
   ```

Archetype distribution appears in summary:
```json
{
  "archetype_distribution": {
    "Explorer": 287,
    "VIP_Customer": 98,
    ...
  }
}
```

### Tuning Parameters

| Parameter | Where | Effect |
|-----------|-------|--------|
| `epsilon` | `config/archetypes.yaml` or `--epsilon` flag | Exploration rate (uniform random sampling) |
| `t_max` | `config/archetypes.yaml` | Max steps per session |
| `stage_transitions` | `config/archetypes.yaml` | Journey flow probabilities |
| `base_events` | `config/archetypes.yaml` | Event rates (e.g., click, purchase) |
| `action_event_lifts` | `config/archetypes.yaml` | How widgets affect behavior |
| `ACTION_COST` | `SharedSchema/shared_schema/constants.py` | Cost per action (0.1 for discount_banner, etc.) |
| `EVENT_REWARD` | `SharedSchema/shared_schema/constants.py` | Reward per event (8.0 for purchase, etc.) |

## Usage

### Generate Simulations

```bash
# Basic: 1000 sessions, archetype config, output to output/
python run_simulation.py --n-sessions 1000

# With seed for reproducibility
python run_simulation.py --n-sessions 1000 --seed 42

# Different output directory
python run_simulation.py --n-sessions 1000 --output-dir ../OfflineTraining/outputs/simulated/

# Override behavior policy epsilon
python run_simulation.py --n-sessions 1000 --epsilon 0.3

# Custom config file
python run_simulation.py --n-sessions 1000 --config my_archetypes.yaml

# All options together
python run_simulation.py \
  --n-sessions 5000 \
  --seed 123 \
  --config config/archetypes.yaml \
  --output-dir ../OfflineTraining/outputs/simulated/ \
  --epsilon 0.25
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-sessions` | 1000 | Number of customer sessions to simulate |
| `--config` | `config/archetypes.yaml` | Path to archetype YAML config |
| `--output-dir` | `output/` | Directory for JSONL, CSV, summary |
| `--epsilon` | (from config) | Override behavior policy epsilon |
| `--seed` | None | Random seed for reproducibility |

## Output Files

Generated in `--output-dir`:

### `offline_transitions.jsonl`
One JSON object per line, one per transition:
```json
{
  "trajectory_id": "sim_abc123...",
  "t": 0,
  "decision_id": 0,
  "timestamp": "2026-05-16T10:20:30.123456",
  "session_id": "sim_abc123...",
  "user_type": "Explorer",
  "decision_point": "landing",
  "state": {"decision_point": "landing", "schema_version": 2, "device_type": "mobile", ...},
  "action": "discount_banner",
  "propensity": 0.166666,
  "eligible_actions": ["no-op", "trending_carousel", ...],
  "reward": 0.85,
  "reward_without_cost": 0.95,
  "action_cost": 0.1,
  "next_state": {...},
  "done": false
}
```

Format is **drop-in compatible** with `OfflineTraining/extract_offline_dataset.py` output.

### `offline_transitions_flat.csv`
Flattened CSV with JSON columns for state/action/etc.:
```csv
trajectory_id,t,decision_id,timestamp,session_id,user_type,decision_point,state_json,action,propensity,eligible_actions_json,reward,reward_without_cost,action_cost,next_state_json,done
sim_abc123...,0,0,2026-05-16T10:20:30.123456,sim_abc123...,Explorer,landing,"{...}",discount_banner,0.166666,"[...]",0.85,0.95,0.1,"{...}",False
```

### `dataset_summary.json`
Metadata and statistics:
```json
{
  "dataset_contract_version": 1,
  "jsonl_sha256": "0123...cdef",
  "n_transitions": 4982,
  "n_sessions": 1000,
  "reward_mean": 0.487,
  "reward_min": -1.45,
  "reward_max": 11.54,
  "conversion_rate": 0.18,
  "avg_session_length": 4.98,
  "archetype_distribution": {
    "Explorer": 298,
    "FastBuyer": 201,
    ...
  },
  "per_archetype": {
    "Explorer": {
      "n_sessions": 298,
      "n_transitions": 981,
      "conversion_rate": 0.03,
      "mean_reward_per_session": 0.42,
      "std_reward_per_session": 2.10,
      "reward_mean_per_transition": 0.13,
      "avg_session_length": 3.29
    }
  },
  "generated_at": "2026-05-16T10:20:30.123456"
}
```

The writer stages each output beside its destination and publishes it with an
atomic replace. `jsonl_sha256` binds the summary counts to the exact JSONL
bytes, allowing the study-training gate to reject interrupted or stale output
before either policy is trained.

`user_type` is additive and does not form part of the policy state. Existing
offline-training readers therefore remain compatible, while simulator-fidelity
analyses can group transitions and sessions by their generating archetype.

## Integration with OfflineTraining

Use simulated data for offline RL policy training:

```bash
# Generate simulated dataset
cd CustomerSimulation
python run_simulation.py --n-sessions 10000 --output-dir ../OfflineTraining/outputs/simulated/

# Train policy on simulated data
cd ../OfflineTraining
python train_offline_policy.py \
  --input outputs/simulated/offline_transitions.jsonl \
  --output outputs/simulated_policy.json
```

The output format matches `extract_offline_dataset.py` exactly, so no preprocessing needed.

## Architecture: SharedSchema Coupling

```
┌─────────────────────────────────────────────┐
│       SharedSchema (single source)          │
├─────────────────────────────────────────────┤
│  constants.py: EPSILON, ACTION_COST, etc.   │
│  features.py: POINT_FEATURES, helpers       │
└─────────────────┬──────────────────────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
  ┌─────────┐ ┌───────┐ ┌──────────────┐
  │DemoSiteV2 │DemoSiteV3 │CustomerSimulation│
  │ (online)  │(offline)  │  (offline gen)  │
  └─────────┘ └───────┘ └──────────────┘
```

**Benefits:**

- **Single edit point:** Change `POINT_FEATURES` → all systems adapt automatically
- **Schema versioning:** `CONTEXT_SCHEMA_VERSION` in constants ensures compatibility
- **Deterministic behavior:** Same reward/cost values across training and deployment
- **Extensibility:** Add new features/archetypes without touching decision logic

## Architecture: Simulation Engine

```
Archetype (YAML)
  ├─ stage_transitions (Markov chain)
  ├─ base_events (Bernoulli probabilities)
  ├─ widget_events (click/dismiss)
  └─ action_event_lifts (policy effects)

BehaviorPolicy
  └─ select_action(eligible) → (action, propensity)
       Uniform random: p = 1/n for each eligible action
       (Maximizes diversity for OPE validity)

SimState
  └─ to_state_dict() → normalized feature dict
       Uses POINT_FEATURES from shared_schema
       Auto-computes buckets via _bucket()

SessionSimulator
  └─ simulate_session(archetype, policy, t_max)
       For each step:
         1. Sample eligible actions
         2. Choose action via behavior policy
         3. Sample events from archetype + action lifts
         4. Compute reward = event_rewards - action_cost
         5. Transition to next state/stage
         6. Append transition to session

DatasetWriter
  └─ write(all_sessions) → (jsonl, csv, summary)
       JSONL: one transition per line
       CSV: flattened with JSON columns
       Summary: aggregated stats
```

## Development

### Running Tests

```bash
# Verify shared_schema is importable
python -c "from shared_schema import POINT_FEATURES; print(POINT_FEATURES)"

# Test a single archetype
python -c "
from simulation.archetype import load_archetypes
archetypes, mixture_prior, config = load_archetypes('config/archetypes.yaml')
print(f'Loaded {len(archetypes)} archetypes')
print(f'Mixture: {mixture_prior}')
"

# Smoke test with 10 sessions
python run_simulation.py --n-sessions 10 --seed 42 --output-dir test_output
```

### Modifying Shared Schema

If you add a new feature or constant:

1. Edit `SharedSchema/shared_schema/constants.py` or `features.py`
2. Update imports in `DemoSiteV2/app/services/decision.py`
3. Update imports in `DemoSiteV3/app/services/decision.py`
4. Reinstall shared_schema:
   ```bash
   pip install -e ../SharedSchema
   ```
5. Re-run simulations to use new values

## Troubleshooting

**Import Error: `ModuleNotFoundError: No module named 'shared_schema'`**
- Ensure SharedSchema is installed: `pip install -e ../SharedSchema`
- Check that your venv is activated

**YAML parsing error in `config/archetypes.yaml`**
- YAML is whitespace-sensitive; use 2 spaces per indent, not tabs
- Run `python -c "import yaml; yaml.safe_load(open('config/archetypes.yaml'))"` to validate

**Simulations too slow**
- Reduce `--n-sessions` for testing
- Reduce `t_max` in config/archetypes.yaml for shorter sessions

**Output files missing**
- Check `--output-dir` is writable
- Ensure directory path exists (created automatically but parent must exist)

## References

- [OfflineTraining README](../OfflineTraining/README.md) — Policy training pipeline
- [DemoSiteV2 README](../DemoSiteV2/README.md) — Online bandit deployment
- [DemoSiteV3 README](../DemoSiteV3/README.md) — Offline-first deployment
- `config/archetypes.yaml` — Full configuration documentation
