"""Migration step v2 -> v3 for Clickworker study attribution.

Adds ``worker_id`` and ``persona`` columns to the ``events`` and ``orders``
tables so that every collected session can be joined back to its experiment
cell (policy condition) and persona scenario. See
``Experiments/ExperimentImplementationPlan.md`` (Phase 1).
"""

from __future__ import annotations

import sqlite3


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_def: str,
) -> None:
    """Idempotently add a column, guarded by a ``PRAGMA table_info`` check."""
    if not _table_exists(conn, table_name):
        return
    if column_name in _column_names(conn, table_name):
        return
    conn.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
    )


def apply(conn: sqlite3.Connection) -> None:
    """Apply schema migration from v2 to v3.

    Changes (all idempotent):
    - Add events.worker_id / events.persona (default '').
    - Add orders.worker_id / orders.persona (default '').
    """
    _add_column_if_missing(conn, "events", "worker_id", "VARCHAR(64) NOT NULL DEFAULT ''")
    _add_column_if_missing(conn, "events", "persona", "VARCHAR(40) NOT NULL DEFAULT ''")
    _add_column_if_missing(conn, "orders", "worker_id", "VARCHAR(64) NOT NULL DEFAULT ''")
    _add_column_if_missing(conn, "orders", "persona", "VARCHAR(40) NOT NULL DEFAULT ''")
