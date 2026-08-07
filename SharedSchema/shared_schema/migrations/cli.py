"""Command line interface for shared schema migrations."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

from .registry import latest_version, next_step, validate_registry
from .state import current_version, ensure_migrations_table, record_step


def resolve_db_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    return os.environ.get("DATABASE_PATH", os.path.join(os.getcwd(), "demosite_test.db"))


def migration_status(db_path: str) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        validate_registry()
        current = current_version(conn)
        latest = latest_version()

        pending: list[dict] = []
        probe = current
        while probe < latest:
            step = next_step(probe)
            if step is None:
                break
            pending.append(
                {
                    "from_version": step.from_version,
                    "to_version": step.to_version,
                    "name": step.name,
                }
            )
            probe = step.to_version

        return {
            "db_path": db_path,
            "current_version": current,
            "latest_version": latest,
            "pending": pending,
        }
    finally:
        conn.close()


def run_migrations(db_path: str, target_version: int | None = None) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    applied: list[dict] = []
    try:
        validate_registry()
        ensure_migrations_table(conn)

        current = current_version(conn)
        latest = latest_version()
        target = latest if target_version is None else target_version

        if target < current:
            raise RuntimeError(
                f"Target version {target} is behind current version {current}. Downgrades are not supported."
            )
        if target > latest:
            raise RuntimeError(
                f"Target version {target} is ahead of latest known version {latest}."
            )

        probe = current
        while probe < target:
            step = next_step(probe)
            if step is None:
                raise RuntimeError(
                    f"No migration path from version {probe} to version {target}."
                )
            step.apply(conn)
            record_step(conn, step.from_version, step.to_version, step.name)
            applied.append(
                {
                    "from_version": step.from_version,
                    "to_version": step.to_version,
                    "name": step.name,
                }
            )
            probe = step.to_version

        conn.commit()
        return {
            "db_path": db_path,
            "start_version": current,
            "target_version": target,
            "end_version": probe,
            "applied": applied,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shared schema migration runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run pending migrations")
    run_parser.add_argument("--db-path", default=None, help="Path to SQLite database file")
    run_parser.add_argument(
        "--target-version",
        "--version",
        type=int,
        default=None,
        help="Target schema version (defaults to latest)",
    )

    status_parser = sub.add_parser("status", help="Show migration status")
    status_parser.add_argument("--db-path", default=None, help="Path to SQLite database file")

    args = parser.parse_args(argv)
    db_path = resolve_db_path(getattr(args, "db_path", None))

    try:
        if args.command == "status":
            result = migration_status(db_path)
            print(f"[migrations] Database: {result['db_path']}")
            print(f"[migrations] Current version: {result['current_version']}")
            print(f"[migrations] Latest version: {result['latest_version']}")
            if result["pending"]:
                print("[migrations] Pending steps:")
                for step in result["pending"]:
                    print(
                        f"  - v{step['from_version']} -> v{step['to_version']}: {step['name']}"
                    )
            else:
                print("[migrations] No pending migrations.")
            return 0

        if args.command == "run":
            result = run_migrations(db_path, target_version=args.target_version)
            print(f"[migrations] Database: {result['db_path']}")
            print(
                f"[migrations] Version: {result['start_version']} -> {result['end_version']}"
            )
            if result["applied"]:
                print("[migrations] Applied steps:")
                for step in result["applied"]:
                    print(
                        f"  - v{step['from_version']} -> v{step['to_version']}: {step['name']}"
                    )
            else:
                print("[migrations] No steps applied (already up to date).")
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"[migrations] ERROR: {exc}", file=sys.stderr)
        return 1
