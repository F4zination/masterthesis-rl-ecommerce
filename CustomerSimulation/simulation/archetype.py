from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Archetype:
    name: str
    stage_transitions: dict[str, dict[str, float]]
    base_events: dict[str, dict[str, float]]
    widget_events: dict[str, dict[str, float]]
    action_event_lifts: dict[str, dict[str, float]]
    order_total_range: list[float]
    # Mechanism 3 (action-dependent transitions): additive lift applied to the
    # next-stage sampling weights when a given action is served.  Shape:
    # {action: {next_stage: lift}}.  Inert unless the global
    # ``transition_coupling_strength`` knob is > 0.
    action_stage_lifts: dict[str, dict[str, float]] = field(default_factory=dict)


def load_archetypes(config_path: str) -> tuple[dict[str, Archetype], dict[str, float], dict[str, Any]]:
    """Load archetypes, mixture prior and full config from a YAML file.

    Returns:
        (archetypes_by_name, mixture_prior, full_config)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)

    archetypes: dict[str, Archetype] = {}
    for name, data in config.get("archetypes", {}).items():
        archetypes[name] = Archetype(
            name=name,
            stage_transitions=data.get("stage_transitions", {}),
            base_events=data.get("base_events", {}),
            widget_events=data.get("widget_events", {}),
            action_event_lifts=data.get("action_event_lifts", {}),
            order_total_range=data.get("order_total_range", [20.0, 100.0]),
            action_stage_lifts=data.get("action_stage_lifts", {}),
        )

    mixture_prior: dict[str, float] = config.get("mixture_prior", {name: 1.0 for name in archetypes})

    return archetypes, mixture_prior, config
