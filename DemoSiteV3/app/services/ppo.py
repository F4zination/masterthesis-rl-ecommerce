from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from shared_schema.policy_contract import (
    PolicyContractError,
    validate_and_load_ppo_checkpoint_model,
    validate_ppo_study_metadata,
)

from ..config import PPO_CHECKPOINT_PATH

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime dependency issue
    torch = None
    nn = None


_PPO_CACHE_LOCK = Lock()
_PPO_CACHE_MTIME: float | None = None
_PPO_CACHE_PATH: str | None = None
_PPO_CACHE_PAYLOAD: dict[str, Any] | None = None
_PPO_CACHE_MODEL: "ActorCritic | None" = None


def _state_tokens(state: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in sorted(state.items()):
        if value is None or isinstance(value, (dict, list)):
            continue
        tokens.append(f"{key}={value}")
    return tokens


def _state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=True)


if torch is not None and nn is not None:

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

else:  # pragma: no cover - only used when Torch is unavailable locally
    ActorCritic = None  # type: ignore[assignment]


def _load_checkpoint(path: Path) -> tuple[dict[str, Any] | None, ActorCritic | None]:
    if torch is None or nn is None or not path.exists():
        return None, None

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, None

    global _PPO_CACHE_MTIME, _PPO_CACHE_PATH, _PPO_CACHE_PAYLOAD, _PPO_CACHE_MODEL
    with _PPO_CACHE_LOCK:
        if (
            _PPO_CACHE_PAYLOAD is not None
            and _PPO_CACHE_MODEL is not None
            and _PPO_CACHE_MTIME == mtime
            and _PPO_CACHE_PATH == str(path)
        ):
            return _PPO_CACHE_PAYLOAD, _PPO_CACHE_MODEL

        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older supported PyTorch
                payload = torch.load(path, map_location="cpu")
        except Exception:
            return None, None

        if not isinstance(payload, dict):
            return None, None

        try:
            model, _ = validate_and_load_ppo_checkpoint_model(
                payload,
                ActorCritic,
            )
        except PolicyContractError:
            return None, None

        _PPO_CACHE_MTIME = mtime
        _PPO_CACHE_PATH = str(path)
        _PPO_CACHE_PAYLOAD = payload
        _PPO_CACHE_MODEL = model
        return payload, model


def get_ppo_health_status() -> dict[str, Any]:
    path = Path(PPO_CHECKPOINT_PATH)
    payload, _ = _load_checkpoint(path)
    study_contract: dict[str, Any] | None = None
    study_contract_error = ""
    if payload is not None:
        try:
            study_contract = validate_ppo_study_metadata(payload)
        except PolicyContractError as exc:
            study_contract_error = str(exc)
    return {
        "checkpoint_path": str(path),
        "available": path.exists(),
        "valid": payload is not None,
        "study_contract_valid": study_contract is not None,
        "study_contract": study_contract or {},
        "study_contract_error": study_contract_error,
        "summary": (payload or {}).get("summary", {}),
    }


def _encode_state(state: dict[str, Any], decision_point: str, vocab: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for token in _state_tokens(state):
        index = vocab.get(token)
        if index is not None:
            vector[index] = 1.0
    point_token = f"decision_point={decision_point}"
    index = vocab.get(point_token)
    if index is not None:
        vector[index] = 1.0
    return vector


def run_ppo_policy(decision_point: str, normalized_context: dict[str, Any], actions: list[str]) -> tuple[str, float, str, dict[str, Any]] | None:
    payload, model = _load_checkpoint(Path(PPO_CHECKPOINT_PATH))
    if payload is None or model is None or torch is None:
        return None

    state_vocab = payload.get("state_vocab") or {}
    action_vocab = payload.get("action_vocab") or {}
    if not isinstance(state_vocab, dict) or not isinstance(action_vocab, dict):
        return None

    inverse_action_vocab = {int(index): action for action, index in action_vocab.items()}
    state_tensor = _encode_state(normalized_context, decision_point, state_vocab).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits.squeeze(0), dim=-1)

    ranked_indices = torch.argsort(probs, descending=True).tolist()
    allowed_actions = set(actions)
    chosen_action = None
    chosen_score = 0.0
    for index in ranked_indices:
        candidate = inverse_action_vocab.get(int(index))
        if candidate is None:
            continue
        if allowed_actions and candidate not in allowed_actions:
            continue
        chosen_action = candidate
        chosen_score = float(probs[int(index)].item())
        break

    if chosen_action is None:
        return None

    # Serving is deterministic argmax, so the logged propensity — the actual
    # probability the serving policy assigned to the served action — is 1.0.
    # The softmax score is kept as diagnostic metadata only; logging it as the
    # propensity would corrupt importance-weighted OPE downstream.
    return chosen_action, 1.0, str(payload.get("algorithm") or "ppo_clip_actor_critic"), {
        "checkpoint_path": str(Path(PPO_CHECKPOINT_PATH)),
        "state_key": _state_key(normalized_context),
        "policy_source": "ppo_policy",
        "policy_score": chosen_score,
        "state_vocab_size": len(state_vocab),
        "action_vocab_size": len(action_vocab),
    }
