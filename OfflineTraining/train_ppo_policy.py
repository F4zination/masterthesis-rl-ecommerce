#!/usr/bin/env python3
"""Train a PPO policy from simulated transition data.

The script consumes the shared JSONL transition format used by both
CustomerSimulation and DemoSiteV3 extraction, then trains a small actor-critic
model with PPO-style clipped updates. It exports both a PyTorch checkpoint and
a compatibility JSON policy artifact for the current DemoSiteV3 runtime.

Two training modes (V3_sequential_environment_plan.md §6):

* ``--mode online`` (default, recommended): true on-policy PPO.  Each iteration
  rolls out the *current* policy through ``SessionSimulator`` to collect fresh
  trajectories, records ``old_log_prob`` as the current policy's log-prob of the
  taken action at collection time, and computes advantages with the standard
  GAE(λ) recursion using the critic's V(s_t)/V(s_{t+1}).  This removes the three
  confounds of the legacy trainer (advantages never GAE'd, ratio referenced the
  uniform behaviour policy, purely offline on a fixed dataset), so a V3 win can
  be attributed to sequential structure rather than an under-tuned optimizer.

* ``--mode offline`` (labelled ablation): trains on the fixed dataset, but with
  ``old_log_probs`` taken from a frozen copy of the policy *before* each update
  iteration and advantages recomputed per epoch as ``returns - V(s)``.  This is
  the "minimum acceptable" correction; kept only for comparison.

The dataset is still required in both modes: it supplies the offline ablation
transitions, identifies which input columns need neutral unseen-category
initialisation, and seeds the greedy ``trained_policy.json`` export. The model
vocabulary itself is built from the complete finite production schema before
any online rollout, so serving categories cannot disappear merely because they
were absent from one dataset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - dependency error is runtime-specific
    raise SystemExit(
        "PyTorch is required for PPO training. Install DemoSiteV3 requirements first."
    ) from exc

# Make the simulator + shared schema importable for on-policy rollouts.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "CustomerSimulation"))
sys.path.insert(0, str(_repo_root / "SharedSchema"))

from simulation.archetype import load_archetypes  # noqa: E402
from simulation.simulator import SessionSimulator, merge_dynamics  # noqa: E402
from simulation.state import SimState  # noqa: E402
from shared_schema.constants import (  # noqa: E402
    ACTION_COST,
    ALL_ACTIONS,
    DECISION_POINTS,
    DEVICE_TYPES,
    FEATURE_VALUE_DOMAINS,
    TRAFFIC_SOURCES,
)
from shared_schema.features import (  # noqa: E402
    HISTORY_FEATURES,
    POINT_FEATURES,
    _eligible_actions,
)
from shared_schema.policy_contract import (  # noqa: E402
    required_ppo_state_tokens,
    validate_ppo_vocabulary,
)


# ---------------------------------------------------------------------------
# Production state/action domain
# ---------------------------------------------------------------------------

# Increment this when the explicit finite serving-time token contract changes.
# Keeping this definition here makes PPO input dimensionality independent of
# which categories happened to occur in one training dataset or rollout.
STATE_VOCAB_DOMAIN_VERSION = 1
UNOBSERVED_TOKEN_INITIALIZATION = "zero_input_columns"
PRODUCTION_DECISION_POINTS: tuple[str, ...] = tuple(DECISION_POINTS)
PRODUCTION_STATE_DOMAINS: tuple[tuple[str, tuple[Any, ...]], ...] = tuple(
    (feature, tuple(values))
    for feature, values in FEATURE_VALUE_DOMAINS.items()
)


# ---------------------------------------------------------------------------
# Data / encoding helpers
# ---------------------------------------------------------------------------

def state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=True)


def load_transitions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    safe_repo = str(_repo_root.resolve()).replace("\\", "/")
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={safe_repo}", "rev-parse", "HEAD"],
            cwd=_repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", f"safe.directory={safe_repo}", "status", "--porcelain"],
                cwd=_repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return (128, 64)
    return tuple(max(8, int(part)) for part in parts)


def state_tokens(state: dict[str, Any]) -> list[str]:
    tokens = []
    for key, value in sorted(state.items()):
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        tokens.append(f"{key}={value}")
    return tokens


def required_production_state_tokens() -> tuple[str, ...]:
    """Return the canonical, finite token domain emitted by V3 serving.

    The consistency checks deliberately fail closed if SharedSchema starts
    emitting another feature or decision point without this trainer being
    updated. That prevents a schema change from silently producing another
    partially out-of-vocabulary checkpoint.
    """
    declared_features = {feature for feature, _ in PRODUCTION_STATE_DOMAINS}
    emitted_features = {
        "schema_version",
        "decision_point",
        *HISTORY_FEATURES,
        *(feature for features in POINT_FEATURES.values() for feature in features),
    }
    if declared_features != emitted_features:
        missing = sorted(emitted_features - declared_features)
        stale = sorted(declared_features - emitted_features)
        raise RuntimeError(
            "PPO production state domain is out of sync with SharedSchema "
            f"(missing={missing}, stale={stale})"
        )

    shared_points = tuple(POINT_FEATURES)
    if set(shared_points) != set(PRODUCTION_DECISION_POINTS):
        raise RuntimeError(
            "PPO production decision-point domain is out of sync with "
            f"SharedSchema (trainer={list(PRODUCTION_DECISION_POINTS)}, "
            f"shared={list(shared_points)})"
        )

    return required_ppo_state_tokens()


def observed_state_tokens(rows: list[dict[str, Any]]) -> set[str]:
    """Return scalar state and decision-point tokens observed in ``rows``."""
    observed: set[str] = set()
    for row in rows:
        state = row.get("state") or {}
        if isinstance(state, dict):
            observed.update(state_tokens(state))
        point = str(row.get("decision_point") or "")
        if point:
            observed.add(f"decision_point={point}")
    return observed


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Build a deterministic vocabulary before any online rollout.

    Every finite production token is present even when absent from ``rows``.
    Extra scalar tokens in historical/experimental datasets remain supported,
    but are appended in sorted order so row ordering cannot alter input
    indices.
    """
    required = required_production_state_tokens()
    required_set = set(required)
    extras = sorted(observed_state_tokens(rows) - required_set)
    return {
        token: index
        for index, token in enumerate((*required, *extras))
    }


def build_action_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    observed_actions = {
        str(row.get("action") or "")
        for row in rows
        if str(row.get("action") or "")
    }
    actions = [*ALL_ACTIONS, *sorted(observed_actions - set(ALL_ACTIONS))]
    return {action: index for index, action in enumerate(actions)}


def vocabulary_coverage(
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    *,
    observed_tokens: set[str] | None = None,
) -> dict[str, Any]:
    """Return an auditable serving-domain coverage report."""
    required_tokens = set(required_production_state_tokens())
    required_actions = set(ALL_ACTIONS)
    observed = observed_tokens or set()
    try:
        state_indices = [int(index) for index in vocab.values()]
        action_indices = [int(index) for index in action_vocab.values()]
    except (TypeError, ValueError):
        state_indices = []
        action_indices = []
    return {
        "domain_version": STATE_VOCAB_DOMAIN_VERSION,
        "required_state_token_count": len(required_tokens),
        "state_vocab_size": len(vocab),
        "missing_state_tokens": sorted(required_tokens - set(vocab)),
        "unobserved_required_state_tokens": sorted(required_tokens - observed),
        "required_action_count": len(required_actions),
        "action_vocab_size": len(action_vocab),
        "missing_actions": sorted(required_actions - set(action_vocab)),
        "state_indices_dense_and_unique": (
            len(set(state_indices)) == len(state_indices)
            and set(state_indices) == set(range(len(vocab)))
        ),
        "action_indices_dense_and_unique": (
            len(set(action_indices)) == len(action_indices)
            and set(action_indices) == set(range(len(action_vocab)))
        ),
        "unobserved_token_initialization": UNOBSERVED_TOKEN_INITIALIZATION,
    }


def validate_vocabulary_coverage(
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    *,
    observed_tokens: set[str] | None = None,
) -> dict[str, Any]:
    """Validate that an artifact can encode every finite serving-time token."""
    # The shared contract is the authoritative validator used by training,
    # serving startup, and release audits.
    validate_ppo_vocabulary(vocab, action_vocab)
    coverage = vocabulary_coverage(
        vocab,
        action_vocab,
        observed_tokens=observed_tokens,
    )
    return coverage


def encode_state(state: dict[str, Any], decision_point: str, vocab: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for token in state_tokens(state):
        index = vocab.get(token)
        if index is not None:
            vector[index] = 1.0
    point_token = f"decision_point={decision_point}"
    index = vocab.get(point_token)
    if index is not None:
        vector[index] = 1.0
    return vector


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or row.get("session_id") or "")
        grouped[trajectory_id].append(row)
    for trajectory_rows in grouped.values():
        trajectory_rows.sort(key=lambda item: int(item.get("t", 0)))
    return grouped


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation for a single episode.

    Episodes collected here always terminate on the last step (``done`` is set
    when the funnel reaches ``done`` or the horizon ``t_max`` is hit), so the
    bootstrap value past the terminal step is 0.
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = values[t + 1] if (t + 1 < n) else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


@dataclass
class PPOBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def initialize_unobserved_token_columns(
    model: ActorCritic,
    vocab: dict[str, int],
    observed_tokens: set[str],
) -> list[str]:
    """Give tokens absent from the seed dataset a neutral input effect.

    PyTorch's default random input-column initialization would otherwise make
    never-seen serving categories (notably ``device_type=unknown``) behave
    according to arbitrary seed noise. Zero columns mean "no learned evidence
    from this category" initially. Tokens encountered in online rollouts can
    still acquire gradients and learn normally; tokens never encountered stay
    at the explicit neutral fallback.
    """
    unobserved = sorted(set(vocab) - observed_tokens)
    indices = [vocab[token] for token in unobserved]
    if not indices:
        return unobserved

    first_linear = next(
        (module for module in model.backbone.modules() if isinstance(module, nn.Linear)),
        None,
    )
    with torch.no_grad():
        if first_linear is not None:
            first_linear.weight[:, indices] = 0.0
        else:
            # With no hidden layers, the heads consume the state vector
            # directly and therefore own the input columns.
            model.policy_head.weight[:, indices] = 0.0
            model.value_head.weight[:, indices] = 0.0
    return unobserved


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)


# ---------------------------------------------------------------------------
# On-policy rollout collection (online PPO)
# ---------------------------------------------------------------------------

def collect_online_rollouts(
    model: ActorCritic,
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    archetypes: dict[str, Any],
    mixture_prior: dict[str, float],
    dynamics: dict[str, Any],
    sequential: bool,
    t_max: int,
    n_sessions: int,
    gamma: float,
    gae_lambda: float,
    rng: random.Random,
) -> PPOBatch:
    """Roll the *current* policy through the simulator and build a PPO batch.

    ``old_log_prob`` is the current policy's log-prob of the sampled action at
    collection time (the frozen reference for this iteration's clip), and
    advantages use GAE(λ) over the critic's collection-time values.
    """
    sim = SessionSimulator(dynamics=dynamics, sequential=sequential)
    inv_action = {idx: a for a, idx in action_vocab.items()}
    arch_names = list(mixture_prior.keys())
    arch_weights = [mixture_prior[a] for a in arch_names]

    all_states: list[torch.Tensor] = []
    all_actions: list[int] = []
    all_old_logp: list[float] = []
    all_adv: list[float] = []
    all_ret: list[float] = []

    model.eval()
    for _ in range(n_sessions):
        archetype = archetypes[rng.choices(arch_names, weights=arch_weights, k=1)[0]]
        state = SimState(
            decision_point="landing",
            device_type=rng.choice(DEVICE_TYPES),
            traffic_source=rng.choice(TRAFFIC_SOURCES),
            price=rng.uniform(5.0, 200.0),
        )

        ep_states: list[torch.Tensor] = []
        ep_actions: list[int] = []
        ep_logp: list[float] = []
        ep_rewards: list[float] = []
        ep_values: list[float] = []
        ep_dones: list[bool] = []

        for t in range(t_max):
            dp = state.decision_point
            eligible = _eligible_actions(dp)
            sd = state.to_state_dict(sequential=sequential)
            x = encode_state(sd, dp, vocab)

            with torch.no_grad():
                logits, value = model(x.unsqueeze(0))
            logits = logits.squeeze(0)

            # Mask to eligible actions (kept general; POINT_ACTIONS == ALL_ACTIONS).
            mask = torch.full_like(logits, float("-inf"))
            for a in eligible:
                idx = action_vocab.get(a)
                if idx is not None:
                    mask[idx] = 0.0
            dist = torch.distributions.Categorical(logits=logits + mask)
            a_idx = dist.sample()
            logp = dist.log_prob(a_idx)
            action = inv_action.get(int(a_idx.item()), "no-op")

            events, order_total = sim._sample_events(state, action, archetype, rng)
            next_stage = sim._sample_next_stage(dp, archetype, events, rng, action)
            events = sim._ensure_cart_reachability_events(state, next_stage, events)
            reward = sim._compute_reward(events, order_total) - ACTION_COST.get(action, 0.0)
            done = next_stage == "done" or t == t_max - 1
            next_state = sim._make_next_state(state, next_stage, events, archetype, rng, action)

            ep_states.append(x)
            ep_actions.append(int(a_idx.item()))
            ep_logp.append(float(logp.item()))
            ep_rewards.append(float(reward))
            ep_values.append(float(value.item()))
            ep_dones.append(bool(done))

            if done:
                break
            state = next_state

        adv, ret = compute_gae(ep_rewards, ep_values, ep_dones, gamma, gae_lambda)
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_old_logp.extend(ep_logp)
        all_adv.extend(adv)
        all_ret.extend(ret)

    if not all_states:
        raise ValueError("Rollout produced no transitions")

    return PPOBatch(
        states=torch.stack(all_states),
        actions=torch.tensor(all_actions, dtype=torch.long),
        old_log_probs=torch.tensor(all_old_logp, dtype=torch.float32),
        returns=torch.tensor(all_ret, dtype=torch.float32),
        advantages=_normalize(torch.tensor(all_adv, dtype=torch.float32)),
    )


# ---------------------------------------------------------------------------
# Offline batch (ablation)
# ---------------------------------------------------------------------------

def make_offline_batch(
    rows: list[dict[str, Any]],
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    gamma: float,
) -> PPOBatch:
    grouped = group_rows(rows)
    encoded_states: list[torch.Tensor] = []
    action_indices: list[int] = []
    returns: list[float] = []

    for trajectory_rows in grouped.values():
        rewards = [float(row.get("reward", 0.0)) for row in trajectory_rows]
        trajectory_returns = discounted_returns(rewards, gamma)
        for row, target_return in zip(trajectory_rows, trajectory_returns):
            state = row.get("state") or {}
            if not isinstance(state, dict):
                state = {}
            decision_point = str(row.get("decision_point") or "")
            encoded_states.append(encode_state(state, decision_point, vocab))
            action = str(row.get("action") or "no-op")
            action_indices.append(action_vocab.get(action, action_vocab["no-op"]))
            returns.append(target_return)

    if not encoded_states:
        raise ValueError("No transitions available for PPO training")

    return_tensor = torch.tensor(returns, dtype=torch.float32)
    return PPOBatch(
        states=torch.stack(encoded_states),
        actions=torch.tensor(action_indices, dtype=torch.long),
        old_log_probs=torch.zeros(len(action_indices), dtype=torch.float32),
        returns=return_tensor,
        advantages=return_tensor.clone(),
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    recompute_advantages: bool,
    rng: random.Random,
) -> None:
    """Run ``epochs`` of clipped minibatch updates over ``batch``.

    When ``recompute_advantages`` is set (offline ablation), advantages are
    re-derived as ``returns - V(s)`` at the start of each epoch using the
    current critic; otherwise the GAE advantages computed at collection time
    are reused (standard online PPO).

    Minibatch shuffling uses the caller's seeded ``rng`` so a training run is
    fully reproducible from its seed (the global ``random`` module is never
    seeded here).
    """
    model.train()
    dataset_size = batch.states.shape[0]
    indices = list(range(dataset_size))

    for _ in range(epochs):
        if recompute_advantages:
            with torch.no_grad():
                _, values = model(batch.states)
            batch.advantages = _normalize(batch.returns - values)

        rng.shuffle(indices)
        for start in range(0, dataset_size, minibatch_size):
            mb = indices[start : start + minibatch_size]
            mb_states = batch.states[mb]
            mb_actions = batch.actions[mb]
            mb_old_log_probs = batch.old_log_probs[mb]
            mb_returns = batch.returns[mb]
            mb_advantages = batch.advantages[mb]

            logits, values = model(mb_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, mb_returns)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def train_ppo(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    hidden_sizes: tuple[int, ...],
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    lr: float,
    iterations: int,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    rollout_sessions: int,
    t_max: int,
    dynamics: dict[str, Any],
    sequential: bool,
    archetypes: dict[str, Any],
    mixture_prior: dict[str, float],
    seed: int,
) -> tuple[ActorCritic, dict[str, int], dict[str, int], dict[str, Any]]:
    vocab = build_vocab(rows)
    action_vocab = build_action_vocab(rows)
    observed_tokens = observed_state_tokens(rows)
    coverage = validate_vocabulary_coverage(
        vocab,
        action_vocab,
        observed_tokens=observed_tokens,
    )

    torch.manual_seed(seed)
    model = ActorCritic(len(vocab), len(action_vocab), hidden_sizes)
    initialized_tokens = initialize_unobserved_token_columns(
        model,
        vocab,
        observed_tokens,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    if mode == "online":
        for iteration in range(iterations):
            batch = collect_online_rollouts(
                model, vocab, action_vocab, archetypes, mixture_prior,
                dynamics, sequential, t_max, rollout_sessions, gamma, gae_lambda, rng,
            )
            ppo_update(
                model, optimizer, batch,
                epochs=epochs, minibatch_size=minibatch_size, clip_eps=clip_eps,
                entropy_coef=entropy_coef, value_coef=value_coef,
                max_grad_norm=max_grad_norm, recompute_advantages=False,
                rng=rng,
            )
    elif mode == "offline":
        base = make_offline_batch(rows, vocab, action_vocab, gamma)
        for iteration in range(iterations):
            # Freeze the policy-before-update so the clip ratio references the
            # previous policy (not the uniform behaviour policy).
            old_model = copy.deepcopy(model)
            old_model.eval()
            with torch.no_grad():
                old_logits, _ = old_model(base.states)
                base.old_log_probs = torch.distributions.Categorical(
                    logits=old_logits
                ).log_prob(base.actions)
            ppo_update(
                model, optimizer, base,
                epochs=epochs, minibatch_size=minibatch_size, clip_eps=clip_eps,
                entropy_coef=entropy_coef, value_coef=value_coef,
                max_grad_norm=max_grad_norm, recompute_advantages=True,
                rng=rng,
            )
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'online' or 'offline')")

    summary = {
        "mode": mode,
        "iterations": iterations,
        "rollout_sessions": rollout_sessions if mode == "online" else None,
        "n_rows": len(rows),
        "n_states": len({state_key(row.get("state") or {}) for row in rows}),
        "state_vocab_size": len(vocab),
        "action_vocab_size": len(action_vocab),
        "gae_lambda": gae_lambda,
        "sequential": sequential,
        "vocabulary_coverage": coverage,
        "unobserved_initialized_token_count": len(initialized_tokens),
    }
    return model, vocab, action_vocab, summary


def greedy_policy_from_model(
    model: ActorCritic,
    rows: list[dict[str, Any]],
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    *,
    sequential: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    index_to_action = {index: action for action, index in action_vocab.items()}
    policy: dict[str, str] = {}
    defaults: dict[str, str] = {}

    domain_defaults = {
        feature: values[0]
        for feature, values in PRODUCTION_STATE_DOMAINS
        if values
    }

    def choose_action(
        state: dict[str, Any],
        decision_point: str,
        eligible_actions: list[str],
    ) -> str:
        action_logits, _ = model(
            encode_state(state, decision_point, vocab).unsqueeze(0)
        )
        probs = torch.softmax(action_logits.squeeze(0), dim=-1)
        allowed = set(eligible_actions)
        for action_index in torch.argsort(probs, descending=True).tolist():
            candidate = index_to_action.get(int(action_index))
            if candidate is not None and (not allowed or candidate in allowed):
                return candidate
        raise ValueError(
            f"PPO model cannot produce an eligible action for {decision_point!r}"
        )

    model.eval()
    with torch.no_grad():
        for row in rows:
            state = row.get("state") or {}
            if not isinstance(state, dict):
                state = {}
            decision_point = str(row.get("decision_point") or "")
            eligible_actions = row.get("eligible_actions") or []
            allowed_actions = (
                [str(action) for action in eligible_actions]
                if isinstance(eligible_actions, list) and eligible_actions
                else list(_eligible_actions(decision_point))
            )
            policy[state_key(state)] = choose_action(
                state,
                decision_point,
                allowed_actions,
            )

        # Point defaults are model-derived as well, rather than an unrelated
        # hard-coded no-op. They use a deterministic baseline production state
        # and cover every serving decision point even if one was absent from
        # the seed dataset.
        for decision_point in PRODUCTION_DECISION_POINTS:
            default_state: dict[str, Any] = {
                "schema_version": domain_defaults["schema_version"],
                "decision_point": decision_point,
            }
            for feature in POINT_FEATURES[decision_point]:
                default_state[feature] = domain_defaults[feature]
            if sequential:
                for feature in HISTORY_FEATURES:
                    default_state.setdefault(feature, domain_defaults[feature])
            defaults[decision_point] = choose_action(
                default_state,
                decision_point,
                list(_eligible_actions(decision_point)),
            )

    return policy, defaults


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and logical bytes in a model state dict."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        if tensor.numel():
            byte_values = tensor.reshape(-1).view(torch.uint8).tolist()
            digest.update(bytes(byte_values))
        digest.update(b"\0")
    return digest.hexdigest()


def model_state_sha256(model: ActorCritic) -> str:
    return _state_dict_sha256(dict(model.state_dict()))


def build_export_identity(
    model: ActorCritic,
    vocab: dict[str, int],
    action_vocab: dict[str, int],
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Build the shared identity embedded in both exported artifacts."""
    identity = {
        "model_state_sha256": model_state_sha256(model),
        "state_vocab_sha256": _canonical_json_sha256(vocab),
        "action_vocab_sha256": _canonical_json_sha256(action_vocab),
    }
    identity["export_id"] = _canonical_json_sha256(
        {
            **identity,
            "dataset_sha256": provenance.get("dataset_sha256"),
            "training_script_sha256": provenance.get("training_script_sha256"),
            "seed": provenance.get("seed"),
            "mode": provenance.get("mode"),
            "sequential": provenance.get("sequential"),
            "timing_mode": provenance.get("timing_mode"),
        }
    )
    return identity


def load_checkpoint_for_validation(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"PPO checkpoint is not a mapping: {path}")
    return payload


def validate_saved_artifact_pair(
    checkpoint_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    """Verify that the checkpoint and JSON export are one auditable pair."""
    checkpoint = load_checkpoint_for_validation(checkpoint_path)
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy_payload, dict):
        raise ValueError(f"PPO JSON policy is not a mapping: {policy_path}")

    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("PPO checkpoint has no model_state_dict")
    actual_model_sha256 = _state_dict_sha256(checkpoint_state)

    errors: list[str] = []
    checkpoint_provenance = checkpoint.get("provenance")
    policy_provenance = policy_payload.get("provenance")
    if checkpoint.get("algorithm") != policy_payload.get("algorithm"):
        errors.append("algorithm differs between checkpoint and JSON")
    if not isinstance(checkpoint_provenance, dict):
        errors.append("checkpoint provenance is missing")
    if not isinstance(policy_provenance, dict):
        errors.append("JSON provenance is missing")
    for key in (
        "export_id",
        "model_state_sha256",
        "state_vocab_sha256",
        "action_vocab_sha256",
    ):
        if checkpoint.get(key) != policy_payload.get(key):
            errors.append(f"{key} differs between checkpoint and JSON")
        if isinstance(checkpoint_provenance, dict) and (
            checkpoint_provenance.get(key) != checkpoint.get(key)
        ):
            errors.append(f"checkpoint provenance has inconsistent {key}")
        if isinstance(policy_provenance, dict) and (
            policy_provenance.get(key) != policy_payload.get(key)
        ):
            errors.append(f"JSON provenance has inconsistent {key}")
    if checkpoint.get("provenance") != policy_payload.get("provenance"):
        errors.append("training provenance differs between checkpoint and JSON")
    if checkpoint.get("model_state_sha256") != actual_model_sha256:
        errors.append("checkpoint model_state_sha256 does not match model tensors")
    if policy_payload.get("checkpoint_sha256") != checkpoint_sha256:
        errors.append("JSON checkpoint_sha256 does not match checkpoint bytes")
    artifact_integrity = policy_payload.get("artifact_integrity")
    if (
        not isinstance(artifact_integrity, dict)
        or artifact_integrity.get("checkpoint_sha256") != checkpoint_sha256
    ):
        errors.append("JSON artifact_integrity checkpoint hash is invalid")

    state_vocab = checkpoint.get("state_vocab")
    action_vocab = checkpoint.get("action_vocab")
    if not isinstance(state_vocab, dict) or not isinstance(action_vocab, dict):
        errors.append("checkpoint state/action vocabulary is missing")
    else:
        try:
            validate_vocabulary_coverage(state_vocab, action_vocab)
        except ValueError as exc:
            errors.append(str(exc))
        if checkpoint.get("state_vocab_sha256") != _canonical_json_sha256(state_vocab):
            errors.append("checkpoint state_vocab_sha256 is invalid")
        if checkpoint.get("action_vocab_sha256") != _canonical_json_sha256(action_vocab):
            errors.append("checkpoint action_vocab_sha256 is invalid")

    policy_export = policy_payload.get("policy_export")
    if not isinstance(policy_export, dict):
        errors.append("JSON policy_export provenance is missing")
    elif policy_export.get("source_model_state_sha256") != actual_model_sha256:
        errors.append("JSON policy was not attributed to the checkpoint model")
    else:
        policy = policy_payload.get("policy")
        defaults = policy_payload.get("default_action_by_decision_point")
        if not isinstance(policy, dict) or (
            policy_export.get("policy_sha256") != _canonical_json_sha256(policy)
        ):
            errors.append("JSON policy digest is invalid")
        if not isinstance(defaults, dict) or (
            policy_export.get("defaults_sha256") != _canonical_json_sha256(defaults)
        ):
            errors.append("JSON decision-point defaults digest is invalid")

    if errors:
        raise ValueError("Inconsistent PPO artifact pair: " + "; ".join(errors))
    return {
        "export_id": checkpoint["export_id"],
        "model_state_sha256": actual_model_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "state_vocab_size": len(state_vocab),
        "action_vocab_size": len(action_vocab),
    }


def write_policy_artifact_pair(
    checkpoint_path: Path,
    policy_path: Path,
    checkpoint_payload: dict[str, Any],
    policy_payload: dict[str, Any],
) -> dict[str, Any]:
    """Write both artifacts via temporary files, then validate their pairing.

    A filesystem cannot atomically replace two independent paths at once.
    Both complete files are therefore staged first and each final path is
    replaced atomically. The shared ``export_id`` plus the JSON's hash of the
    exact checkpoint bytes makes an interrupted/stale mixed pair detectable.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.parent.resolve() != policy_path.parent.resolve():
        raise ValueError("PPO checkpoint and JSON policy must share an output directory")

    for key in (
        "export_id",
        "model_state_sha256",
        "state_vocab_sha256",
        "action_vocab_sha256",
    ):
        if checkpoint_payload.get(key) != policy_payload.get(key):
            raise ValueError(f"Cannot export PPO pair with mismatched {key}")

    checkpoint_fd, checkpoint_tmp_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    policy_fd, policy_tmp_name = tempfile.mkstemp(
        prefix=f".{policy_path.name}.",
        suffix=".tmp",
        dir=policy_path.parent,
    )
    os.close(checkpoint_fd)
    os.close(policy_fd)
    checkpoint_tmp = Path(checkpoint_tmp_name)
    policy_tmp = Path(policy_tmp_name)

    try:
        torch.save(checkpoint_payload, checkpoint_tmp)
        checkpoint_sha256 = _sha256(checkpoint_tmp)

        final_policy_payload = copy.deepcopy(policy_payload)
        final_policy_payload["checkpoint_sha256"] = checkpoint_sha256
        final_policy_payload.setdefault("artifact_integrity", {}).update(
            {
                "pairing_scheme": "shared_export_id_and_checkpoint_sha256",
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
        with policy_tmp.open("w", encoding="utf-8") as handle:
            json.dump(final_policy_payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Each replace is atomic. If the process stops between them, the
        # retained shared IDs/hash expose the mixed generation immediately.
        os.replace(checkpoint_tmp, checkpoint_path)
        os.replace(policy_tmp, policy_path)
    finally:
        checkpoint_tmp.unlink(missing_ok=True)
        policy_tmp.unlink(missing_ok=True)

    return validate_saved_artifact_pair(checkpoint_path, policy_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy from transition data")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "offline_transitions.jsonl"),
        help="Path to extracted or simulated transitions JSONL "
             "(training seed + greedy compatibility export)",
    )
    parser.add_argument(
        "--mode", choices=["online", "offline"], default="online",
        help="online = true on-policy PPO with simulator rollouts (recommended); "
             "offline = labelled ablation on the fixed dataset",
    )
    parser.add_argument(
        "--archetypes",
        default=str(_repo_root / "CustomerSimulation" / "config" / "archetypes.yaml"),
        help="Archetypes YAML for online rollouts + dynamics block",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--iterations", type=int, default=40,
                        help="PPO outer iterations (rollout+update cycles in online mode)")
    parser.add_argument("--rollout-sessions", type=int, default=256,
                        help="Sessions collected per online iteration")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Update epochs per iteration")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sequential", action="store_true",
        help="Condition rollouts on trajectory-history features. The full "
             "history token domain is always reserved in the vocabulary.",
    )
    parser.add_argument("--fatigue-rate", type=float, default=None)
    parser.add_argument("--fatigue-window", type=int, default=None)
    parser.add_argument("--delayed-reward-strength", type=float, default=None)
    parser.add_argument("--delayed-reward-decay", type=float, default=None)
    parser.add_argument("--transition-coupling-strength", type=float, default=None)
    parser.add_argument(
        "--hidden-sizes",
        default="128,64",
        help="Comma-separated MLP hidden sizes",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for trained policy artifacts",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["legacy", "opportunity"],
        default="legacy",
        help="Metadata label for the trained artifact to indicate state schema mode",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_transitions(dataset_path)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    archetypes, mixture_prior, full_config = load_archetypes(args.archetypes)
    dynamics = dict(full_config.get("dynamics", {}) or {})
    for key in (
        "fatigue_rate", "fatigue_window", "delayed_reward_strength",
        "delayed_reward_decay", "transition_coupling_strength",
    ):
        val = getattr(args, key, None)
        if val is not None:
            dynamics[key] = val
    dynamics = merge_dynamics(dynamics)

    model, vocab, action_vocab, summary = train_ppo(
        rows=rows,
        mode=args.mode,
        hidden_sizes=hidden_sizes,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        lr=args.lr,
        iterations=args.iterations,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        rollout_sessions=args.rollout_sessions,
        t_max=args.t_max,
        dynamics=dynamics,
        sequential=args.sequential,
        archetypes=archetypes,
        mixture_prior=mixture_prior,
        seed=args.seed,
    )
    coverage = validate_vocabulary_coverage(
        vocab,
        action_vocab,
        observed_tokens=observed_state_tokens(rows),
    )
    policy, defaults = greedy_policy_from_model(
        model,
        rows,
        vocab,
        action_vocab,
        sequential=args.sequential,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "ppo_policy.pt"
    archetypes_path = Path(args.archetypes).resolve()
    dataset_path = dataset_path.resolve()
    git_commit, git_dirty = _git_state()
    provenance = {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "archetypes_path": str(archetypes_path),
        "archetypes_sha256": _sha256(archetypes_path),
        "training_script_path": str(Path(__file__).resolve()),
        "training_script_sha256": _sha256(Path(__file__).resolve()),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "seed": args.seed,
        "mode": args.mode,
        "sequential": args.sequential,
        "timing_mode": args.timing_mode,
        "dynamics": dynamics,
        "t_max": args.t_max,
        "iterations": args.iterations,
        "rollout_sessions": args.rollout_sessions,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "hidden_sizes": list(hidden_sizes),
    }
    identity = build_export_identity(model, vocab, action_vocab, provenance)
    provenance.update(
        {
            **identity,
            "state_vocab_domain_version": STATE_VOCAB_DOMAIN_VERSION,
            "unobserved_token_initialization": UNOBSERVED_TOKEN_INITIALIZATION,
        }
    )
    checkpoint_payload = {
        "algorithm": "ppo_clip_actor_critic",
        **identity,
        "model_state_dict": model.state_dict(),
        "state_vocab": vocab,
        "action_vocab": action_vocab,
        "hidden_sizes": hidden_sizes,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "lr": args.lr,
        "mode": args.mode,
        "sequential": args.sequential,
        "timing_mode": args.timing_mode,
        "provenance": provenance,
        "summary": summary,
        "vocabulary_coverage": coverage,
    }

    policy_export = {
        "source": "greedy_argmax_from_checkpoint_model",
        "source_model_state_sha256": identity["model_state_sha256"],
        "policy_sha256": _canonical_json_sha256(policy),
        "defaults_sha256": _canonical_json_sha256(defaults),
        "n_policy_states": len(policy),
        "n_decision_point_defaults": len(defaults),
    }
    payload = {
        "algorithm": "ppo_clip_actor_critic",
        **identity,
        "mode": args.mode,
        "sequential": args.sequential,
        "timing_mode": args.timing_mode,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "lr": args.lr,
        "iterations": args.iterations,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "n_rows": len(rows),
        "n_states": summary["n_states"],
        "state_vocab_size": summary["state_vocab_size"],
        "action_vocab_size": summary["action_vocab_size"],
        "checkpoint_path": str(checkpoint_path),
        "provenance": provenance,
        "vocabulary_coverage": coverage,
        "policy_export": policy_export,
        "policy": policy,
        "default_action_by_decision_point": defaults,
    }

    out_path = out_dir / "trained_policy.json"
    integrity = write_policy_artifact_pair(
        checkpoint_path,
        out_path,
        checkpoint_payload,
        payload,
    )

    print(
        json.dumps(
            {
                "out_path": str(out_path),
                "checkpoint_path": str(checkpoint_path),
                "n_policy_states": len(policy),
                "mode": args.mode,
                **integrity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
