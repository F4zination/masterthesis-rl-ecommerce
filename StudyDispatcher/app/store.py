"""Persistent, balanced cell-assignment ledger backed by SQLite.

The dispatcher assigns each participant to exactly one of the 2x5 = 10
experiment cells (condition x persona) using two-stage blocked allocation:
first choose a least-filled persona with a random tie-break, then choose its
least-filled policy arm with a random tie-break. This keeps persona totals and
the V2/V3 split within each persona balanced as recruitment progresses.

Assignments are keyed by participant id (``pid``) so that a worker who opens
their link twice always lands in the same cell, and so the balance survives a
container restart. A process-level lock makes the read-counts-then-insert step
atomic; run the dispatcher with a single worker (see the Dockerfile CMD).
"""
from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

from . import config

_lock = threading.Lock()


class Assignment(NamedTuple):
    pid: str
    wid: str
    condition: str
    persona: str
    created_at: float
    consent_version: str
    consented_at: float | None


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Yield a database connection and always close it after use."""
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    """Create the assignments table if it does not yet exist."""
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                pid        TEXT PRIMARY KEY,
                wid        TEXT NOT NULL,
                condition  TEXT NOT NULL,
                persona    TEXT NOT NULL,
                created_at REAL NOT NULL,
                consent_version TEXT NOT NULL DEFAULT '',
                consented_at REAL
            )
            """
        )
        # Restart-safe migration for ledgers created before consent recording
        # was added. Existing platform assignments remain valid and have an
        # empty consent version because consent may have occurred externally.
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(assignments)").fetchall()
        }
        if "consent_version" not in columns:
            conn.execute(
                "ALTER TABLE assignments ADD COLUMN consent_version "
                "TEXT NOT NULL DEFAULT ''"
            )
        if "consented_at" not in columns:
            conn.execute("ALTER TABLE assignments ADD COLUMN consented_at REAL")


def all_cells() -> list[tuple[str, str]]:
    """Return every (condition, persona) cell in the 2x5 design."""
    return [(cond, persona) for cond in config.CONDITIONS for persona in config.PERSONAS]


def counts() -> dict[tuple[str, str], int]:
    """Return the current assignment count for every cell (zero-filled)."""
    tally = {cell: 0 for cell in all_cells()}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT condition, persona, COUNT(*) FROM assignments GROUP BY condition, persona"
        ).fetchall()
    for cond, persona, n in rows:
        if (cond, persona) in tally:
            tally[(cond, persona)] = n
    return tally


def get(pid: str) -> Assignment | None:
    """Return the existing assignment for ``pid`` or ``None``."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT pid, wid, condition, persona, created_at, consent_version, "
            "consented_at FROM assignments WHERE pid = ?",
            (pid,),
        ).fetchone()
    return Assignment(*row) if row else None


def is_full() -> bool:
    """Return ``True`` once every cell has reached the recruitment target."""
    return min(counts().values()) >= config.TARGET_PER_CELL


def assign(
    pid: str,
    wid: str | None = None,
    *,
    consent_version: str = "",
    consented_at: float | None = None,
) -> tuple[Assignment, bool]:
    """Return ``(assignment, created)`` for ``pid``, assigning a cell if new.

    Idempotent: a repeat ``pid`` returns its original assignment with
    ``created=False``. New participants enter a least-filled persona (random
    tie-break), then its least-filled condition (random tie-break).
    """
    with _lock:
        existing = get(pid)
        if existing is not None:
            return existing, False

        tally = counts()
        persona_totals = {
            persona: sum(tally[(condition, persona)] for condition in config.CONDITIONS)
            for persona in config.PERSONAS
        }
        minimum_persona_total = min(persona_totals.values())
        persona = random.choice(
            [
                name
                for name, total in persona_totals.items()
                if total == minimum_persona_total
            ]
        )
        minimum_condition_total = min(
            tally[(condition, persona)] for condition in config.CONDITIONS
        )
        condition = random.choice(
            [
                name
                for name in config.CONDITIONS
                if tally[(name, persona)] == minimum_condition_total
            ]
        )

        record = Assignment(
            pid=pid,
            wid=wid or pid,
            condition=condition,
            persona=persona,
            created_at=time.time(),
            consent_version=consent_version,
            consented_at=consented_at,
        )
        try:
            with _conn() as conn:
                conn.execute(
                    "INSERT INTO assignments (pid, wid, condition, persona, created_at, "
                    "consent_version, consented_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    record,
                )
        except sqlite3.IntegrityError:
            # Raced with another worker on the same pid; return the winner.
            again = get(pid)
            if again is not None:
                return again, False
            raise
        return record, True
