#!/usr/bin/env python3
"""Verify Clickworker completion codes against the collected study databases.

A participant finishes a session, the shop shows them a 6-digit completion code,
and they paste that code back into the crowd platform. This tool lets you check
those submissions against what the apps actually recorded.

It reads the SQLite databases directly (no app / SharedSchema imports needed),
so you can run it anywhere Python is available:

    python Experiments/verify_completion_codes.py \
        --db v2=DemoSiteV2/data/demosite.db \
        --db v3=DemoSiteV3/data/demosite.db

Each ``--db LABEL=PATH`` registers one condition (the label is reported as the
condition, e.g. ``v2``/``v3``). You can pass one or both.

Modes
-----
(default)           List every completed session: code, worker, persona, pass,
                    quality flags, condition.
--code CODE         Look up which worker/session a single code belongs to.
--worker WID        Show the expected code(s) for one participant.
--csv FILE          Bulk-verify submissions. The CSV needs a worker-id column
                    and a code column (header names are auto-detected, or pass
                    --wid-col / --code-col). Prints VALID / INVALID per row.

A submission is VALID when the worker has a completed session whose recorded
code equals the submitted code. Quality flags (duration >= 90 s, page views
>= 3, some dwell/scroll or widget-dismiss activity, attention passed) are
reported only for the pre-specified sensitivity analysis. They must never be
used to deny base compensation or remove a randomized participant from ITT.

Codes recorded before the code-persistence change are recomputed from the same
formulas the app uses, so older databases still verify.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime

MIN_DURATION_S = 90.0
MIN_PAGE_VIEWS = 3


# ── Code formulas (must match DemoSite*/app/routers/shop.py:_completion_code) ──
def code_from_order(order_id: int) -> str:
    return str((order_id * 7919 + 100003) % 900000 + 100000)


def code_from_session(session_id: str) -> str:
    digest = hashlib.sha256((session_id or "anon").encode("utf-8")).hexdigest()
    return str(int(digest, 16) % 900000 + 100000)


@dataclass
class Session:
    condition: str
    worker_id: str
    persona: str
    session_id: str
    code: str
    passed: bool
    page_views: int
    duration_s: float
    has_dwell_or_dismiss: bool
    code_source: str  # "recorded" or "recomputed"

    @property
    def quality_ok(self) -> bool:
        return (
            self.passed
            and self.duration_s >= MIN_DURATION_S
            and self.page_views >= MIN_PAGE_VIEWS
            and self.has_dwell_or_dismiss
        )

    def quality_reasons(self) -> list[str]:
        out = []
        if not self.passed:
            out.append("attention failed")
        if self.duration_s < MIN_DURATION_S:
            out.append(f"duration {self.duration_s:.0f}s<90s")
        if self.page_views < MIN_PAGE_VIEWS:
            out.append(f"page_views {self.page_views}<3")
        if not self.has_dwell_or_dismiss:
            out.append("no dwell/dismiss")
        return out


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace(" ", "T"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def load_sessions(label: str, db_path: str) -> list[Session]:
    """Read all completed (attention-check) sessions from one database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # session_id -> order_id (for recomputing codes on older rows)
    order_by_session = {
        r["session_id"]: r["id"]
        for r in conn.execute("SELECT id, session_id FROM orders")
    }

    sessions: list[Session] = []
    rows = conn.execute(
        "SELECT session_id, worker_id, persona, metadata_json "
        "FROM events WHERE event_type='attention_check'"
    ).fetchall()

    for r in rows:
        sid = r["session_id"]
        try:
            meta = json.loads(r["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            meta = {}

        passed = bool(meta.get("passed"))
        code = meta.get("completion_code")
        code_source = "recorded"
        if not code:
            code_source = "recomputed"
            order_id = meta.get("order_id") or order_by_session.get(sid)
            code = (
                code_from_order(int(order_id))
                if order_id is not None
                else code_from_session(sid)
            )

        # Quality signals from the session's event stream.
        ev = conn.execute(
            "SELECT event_type, timestamp FROM events WHERE session_id=?", (sid,)
        ).fetchall()
        page_views = sum(1 for e in ev if e["event_type"] == "page_view")
        has_dwell_or_dismiss = any(
            e["event_type"] == "widget_dismiss"
            or e["event_type"].startswith("dwell")
            or e["event_type"].startswith("scroll")
            for e in ev
        )
        times = [t for t in (_parse_ts(e["timestamp"]) for e in ev) if t]
        duration_s = (max(times) - min(times)).total_seconds() if len(times) > 1 else 0.0

        sessions.append(
            Session(
                condition=label,
                worker_id=r["worker_id"] or "",
                persona=r["persona"] or "",
                session_id=sid,
                code=str(code),
                passed=passed,
                page_views=page_views,
                duration_s=duration_s,
                has_dwell_or_dismiss=has_dwell_or_dismiss,
                code_source=code_source,
            )
        )
    conn.close()
    return sessions


def _print_table(sessions: list[Session]) -> None:
    if not sessions:
        print("(no completed sessions found)")
        return
    hdr = f"{'CODE':<8} {'WORKER':<20} {'COND':<5} {'PERSONA':<18} {'PASS':<5} {'VIEWS':<6} {'DUR_S':<7} {'QUALITY'}"
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(sessions, key=lambda x: (x.condition, x.worker_id)):
        q = (
            "ok"
            if s.quality_ok
            else "sensitivity flags: " + "; ".join(s.quality_reasons())
        )
        flag = "" if s.code_source == "recorded" else " *recomputed"
        print(
            f"{s.code:<8} {s.worker_id:<20} {s.condition:<5} {s.persona:<18} "
            f"{('yes' if s.passed else 'no'):<5} {s.page_views:<6} {s.duration_s:<7.0f} {q}{flag}"
        )


def _detect_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    lower = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def verify_submission(sub_wid: str, sub_code: str, by_worker, by_code) -> tuple[str, str]:
    """Return (status, detail) for one (worker, code) submission."""
    sub_code = sub_code.strip()
    sub_wid = sub_wid.strip()
    matches = [s for s in by_worker.get(sub_wid, []) if s.code == sub_code]
    if matches:
        s = matches[0]
        detail = f"{s.condition}/{s.persona}"
        if not s.quality_ok:
            detail += ": sensitivity flags only: " + "; ".join(s.quality_reasons())
        return "VALID", detail
    # Code exists but for a different worker?
    other = by_code.get(sub_code)
    if other:
        return "INVALID", f"code belongs to worker '{other.worker_id}', not '{sub_wid}'"
    if sub_wid not in by_worker:
        return "INVALID", f"no completed session for worker '{sub_wid}'"
    expected = ", ".join(sorted({s.code for s in by_worker[sub_wid]}))
    return "INVALID", f"code mismatch (expected {expected})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", action="append", default=[], metavar="LABEL=PATH",
                    help="condition database, e.g. v2=DemoSiteV2/data/demosite.db (repeatable)")
    ap.add_argument("--code", help="look up a single completion code")
    ap.add_argument("--worker", help="show expected code(s) for one worker id")
    ap.add_argument("--csv", help="bulk-verify a CSV of submissions")
    ap.add_argument("--wid-col", help="worker-id column name in the CSV")
    ap.add_argument("--code-col", help="code column name in the CSV")
    args = ap.parse_args(argv)

    if not args.db:
        ap.error("at least one --db LABEL=PATH is required")

    sessions: list[Session] = []
    for spec in args.db:
        if "=" not in spec:
            ap.error(f"--db must be LABEL=PATH, got '{spec}'")
        label, path = spec.split("=", 1)
        try:
            sessions.extend(load_sessions(label, path))
        except sqlite3.Error as e:
            print(f"error reading {path}: {e}", file=sys.stderr)
            return 2

    by_worker: dict[str, list[Session]] = {}
    by_code: dict[str, Session] = {}
    for s in sessions:
        by_worker.setdefault(s.worker_id, []).append(s)
        by_code[s.code] = s

    if args.code:
        s = by_code.get(args.code.strip())
        if not s:
            print(f"NOT FOUND — no completed session produced code {args.code}")
            return 1
        q = (
            "ok"
            if s.quality_ok
            else "sensitivity flags only: " + "; ".join(s.quality_reasons())
        )
        print(f"code {s.code}: worker={s.worker_id} condition={s.condition} "
              f"persona={s.persona} passed={s.passed} quality={q} source={s.code_source}")
        return 0

    if args.worker:
        rows = by_worker.get(args.worker.strip(), [])
        if not rows:
            print(f"no completed session for worker '{args.worker}'")
            return 1
        for s in rows:
            print(f"worker {s.worker_id}: code={s.code} condition={s.condition} "
                  f"persona={s.persona} passed={s.passed} "
                  f"sensitivity_quality_ok={s.quality_ok}")
        return 0

    if args.csv:
        with open(args.csv, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            wid_col = args.wid_col or _detect_col(
                reader.fieldnames or [], ["worker_id", "wid", "participant_id", "prolific_pid", "pid"])
            code_col = args.code_col or _detect_col(
                reader.fieldnames or [], ["code", "completion_code", "completion code", "answer"])
            if not wid_col or not code_col:
                print(f"could not detect columns in {reader.fieldnames}; "
                      f"pass --wid-col and --code-col", file=sys.stderr)
                return 2
            n_ok = n_flagged = n_bad = 0
            for row in reader:
                status, detail = verify_submission(row[wid_col], row[code_col], by_worker, by_code)
                if status == "VALID":
                    n_ok += 1
                    if "sensitivity flags only:" in detail:
                        n_flagged += 1
                else:
                    n_bad += 1
                print(f"{status:<11} {row[wid_col]:<20} {row[code_col]:<8} {detail}")
            print(
                f"\n{n_ok} valid ({n_flagged} with sensitivity-only flags), "
                f"{n_bad} invalid"
            )
            return 0 if n_bad == 0 else 1

    _print_table(sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
