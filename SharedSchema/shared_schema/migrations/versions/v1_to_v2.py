"""Migration step v1 -> v2 for bandit context schema."""

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


def apply(conn: sqlite3.Connection) -> None:
    """Apply schema migration from v1 to v2.

    Changes:
    - Add bandit_arm_stats.schema_version (default 2) if missing.
    - Mark legacy v1 keys with schema_version=1 based on key shape.
    """
    if not _table_exists(conn, "bandit_arm_stats"):
        return

    columns = _column_names(conn, "bandit_arm_stats")
    if "schema_version" not in columns:
        conn.execute(
            "ALTER TABLE bandit_arm_stats ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 2"
        )

    conn.execute(
        """
        UPDATE bandit_arm_stats
        SET schema_version = 1
        WHERE (LENGTH(context_key) - LENGTH(REPLACE(context_key, '|', ''))) = 6
        """
    )
