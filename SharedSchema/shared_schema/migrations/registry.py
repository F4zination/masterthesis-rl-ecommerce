"""Migration registry and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable

from .versions import v1_to_v2, v2_to_v3


@dataclass(frozen=True)
class MigrationStep:
    from_version: int
    to_version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


MIGRATIONS: tuple[MigrationStep, ...] = (
    MigrationStep(
        from_version=1,
        to_version=2,
        name="v1_to_v2_bandit_context_schema",
        apply=v1_to_v2.apply,
    ),
    MigrationStep(
        from_version=2,
        to_version=3,
        name="v2_to_v3_study_attribution",
        apply=v2_to_v3.apply,
    ),
)


def latest_version() -> int:
    if not MIGRATIONS:
        return 0
    return max(step.to_version for step in MIGRATIONS)


def next_step(current_version: int) -> MigrationStep | None:
    for step in MIGRATIONS:
        if step.from_version == current_version:
            return step
    return None


def validate_registry() -> None:
    """Ensure migration graph has no duplicate from_version edges."""
    seen: set[int] = set()
    for step in MIGRATIONS:
        if step.from_version in seen:
            raise ValueError(f"Duplicate migration origin version: {step.from_version}")
        seen.add(step.from_version)
