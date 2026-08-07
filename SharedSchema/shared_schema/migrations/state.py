"""Migration state detection and metadata persistence."""

from __future__ import annotations

import sqlite3

MIGRATIONS_TABLE = "schema_migrations"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_version INTEGER NOT NULL,
            to_version INTEGER NOT NULL,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def infer_schema_version(conn: sqlite3.Connection) -> int:
    """Infer schema version from DB structure when metadata is absent."""
    if not table_exists(conn, "bandit_arm_stats"):
        return 0

    cols = column_names(conn, "bandit_arm_stats")
    if "schema_version" in cols:
        return 2
    return 1


def current_version(conn: sqlite3.Connection) -> int:
    if table_exists(conn, MIGRATIONS_TABLE):
        row = conn.execute(
            f"SELECT to_version FROM {MIGRATIONS_TABLE} ORDER BY to_version DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            return int(row[0])

    return infer_schema_version(conn)


def record_step(conn: sqlite3.Connection, from_version: int, to_version: int, name: str) -> None:
    ensure_migrations_table(conn)
    conn.execute(
        f"""
        INSERT INTO {MIGRATIONS_TABLE} (from_version, to_version, name)
        VALUES (?, ?, ?)
        """,
        (from_version, to_version, name),
    )
