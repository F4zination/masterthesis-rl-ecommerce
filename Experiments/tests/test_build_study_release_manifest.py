from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_study_release_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_study_release_manifest", SCRIPT)
assert SPEC and SPEC.loader
manifest_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_builder)


class ReleaseManifestTests(unittest.TestCase):
    def test_file_hash_and_image_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"frozen-policy")
            record = manifest_builder.file_record(path, Path(tmp))

        self.assertTrue(record["exists"])
        self.assertEqual(record["sha256"], hashlib.sha256(b"frozen-policy").hexdigest())
        self.assertTrue(manifest_builder.immutable_image_identifier("sha256:" + "a" * 64))
        self.assertTrue(
            manifest_builder.immutable_image_identifier("registry/study@sha256:" + "B" * 64)
        )
        self.assertFalse(manifest_builder.immutable_image_identifier("demosite-v3:latest"))

    def test_required_v3_tokens_cover_history_and_unknown_device(self) -> None:
        tokens = manifest_builder.required_v3_serving_tokens()
        self.assertIn("primed_credit_bucket=3", tokens)
        self.assertIn("device_type=unknown", tokens)
        self.assertIn("decision_point=scroll_engagement", tokens)
        self.assertIn("schema_version=2", tokens)

    def test_manifest_write_refuses_nonempty_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            first = manifest_builder.write_manifest_safely(
                {"status": "DRAFT_NOT_READY"}, output_root, "release_001"
            )
            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8"))["status"],
                "DRAFT_NOT_READY",
            )
            with self.assertRaises(manifest_builder.ReleaseManifestError):
                manifest_builder.write_manifest_safely({}, output_root, "release_001")
            with self.assertRaises(manifest_builder.ReleaseManifestError):
                manifest_builder.write_manifest_safely({}, output_root, "../escape")

    def test_catalog_digest_is_content_based_and_detects_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.db"
            second = root / "second.db"
            for path in (first, second):
                with closing(sqlite3.connect(path)) as conn:
                    conn.executescript(
                        """
                        CREATE TABLE categories (
                            id INTEGER PRIMARY KEY, name TEXT, slug TEXT, description TEXT
                        );
                        CREATE TABLE products (
                            id INTEGER PRIMARY KEY, name TEXT, slug TEXT, description TEXT,
                            price FLOAT, image_url TEXT, category_id INTEGER, stock INTEGER
                        );
                        INSERT INTO categories VALUES (1, 'Books', 'books', 'Reading');
                        INSERT INTO products VALUES
                            (1, 'Novel', 'novel', 'A book', 9.99, '/novel.png', 1, 3);
                        """
                    )
                    conn.commit()
            digest_one = manifest_builder.inspect_catalog(first)
            digest_two = manifest_builder.inspect_catalog(second)
            self.assertTrue(digest_one["valid"])
            self.assertEqual(
                digest_one["canonical_catalog_sha256"],
                digest_two["canonical_catalog_sha256"],
            )
            with closing(sqlite3.connect(second)) as conn:
                conn.execute("UPDATE products SET stock = 2 WHERE id = 1")
                conn.commit()
            changed = manifest_builder.inspect_catalog(second)
            self.assertNotEqual(
                digest_one["canonical_catalog_sha256"],
                changed["canonical_catalog_sha256"],
            )

    def test_preregistration_distinguishes_fields_from_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.md"
            path.write_text(
                "\n".join(
                    [
                        "**Status**: draft",
                        "**Study release commit**: `TO_BE_FILLED`",
                        "Fields marked `TO_BE_FILLED` must be completed.",
                        "- [ ] Fill all `TO_BE_FILLED` fields.",
                    ]
                ),
                encoding="utf-8",
            )
            result = manifest_builder.inspect_preregistration(path, "a" * 40)
        self.assertEqual(len(result["unfilled_fields"]), 1)
        self.assertEqual(result["unfilled_fields"][0]["line"], 2)
        self.assertFalse(result["declared_commit_matches_head"])

    def test_legacy_v2_database_has_no_embedded_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.db"
            with closing(sqlite3.connect(path)) as conn:
                conn.execute("CREATE TABLE bandit_arm_stats (id INTEGER PRIMARY KEY)")
                conn.commit()
            result = manifest_builder.inspect_v2_provenance(path)
        self.assertFalse(result["embedded"])
        self.assertIn("no embedded", result["error"])

    def test_release_requires_empty_confirmatory_data_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / "v2.db"
            v3 = root / "v3.db"
            dispatcher = root / "dispatcher.db"
            for path in (v2, v3):
                with closing(sqlite3.connect(path)) as conn:
                    conn.executescript(
                        "CREATE TABLE events (id INTEGER PRIMARY KEY);"
                        "CREATE TABLE orders (id INTEGER PRIMARY KEY);"
                        "CREATE TABLE decision_logs (id INTEGER PRIMARY KEY);"
                    )
                    conn.commit()
            with closing(sqlite3.connect(dispatcher)) as conn:
                conn.execute("CREATE TABLE assignments (pid TEXT PRIMARY KEY)")
                conn.commit()

            clean = manifest_builder.inspect_pre_recruitment_data(
                v2, v3, dispatcher
            )
            self.assertTrue(clean["clean_for_confirmatory_recruitment"])

            with closing(sqlite3.connect(v2)) as conn:
                conn.execute("INSERT INTO events DEFAULT VALUES")
                conn.commit()
            dirty = manifest_builder.inspect_pre_recruitment_data(
                v2, v3, dispatcher
            )
            self.assertFalse(dirty["clean_for_confirmatory_recruitment"])
            self.assertEqual(dirty["v2"]["counts"]["events"], 1)

    def test_baseline_surfaces_v3_oov_and_serving_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(
                json.dumps({
                    "git_head": "a" * 40,
                    "design": {"n_cells": 10, "sessions_per_cell": 10},
                    "inputs": {},
                    "cells": [
                        {
                            "policy": "v2_bandit",
                            "diagnostics": {
                                "decisions": 10,
                                "unseen_context_decisions": 4,
                                "partially_unseen_context_decisions": 2,
                                "selected_unseen_arm_decisions": 3,
                                "missing_feature_decisions": 0,
                            },
                        },
                        {
                            "policy": "v3_ppo",
                            "diagnostics": {
                                "decisions": 20,
                                "decisions_with_oov_tokens": 3,
                                "fallback_decisions": 0,
                                "serving_equivalent": True,
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )
            artifacts = {
                name: {"sha256": "hash"}
                for name in ("v2_db", "v3_checkpoint", "archetype_config")
            }
            result = manifest_builder.inspect_baseline(path, artifacts, "a" * 40)
        diagnostics = result["v3_serving_diagnostics"]
        self.assertEqual(diagnostics["decisions_with_oov_tokens"], 3)
        self.assertEqual(diagnostics["oov_decision_rate"], 0.15)
        self.assertTrue(diagnostics["all_cells_serving_equivalent"])
        self.assertFalse(result["context_standardization"]["complete"])
        self.assertEqual(
            result["v2_serving_diagnostics"]["unseen_context_decisions"], 4
        )
        self.assertEqual(result["v2_serving_diagnostics"]["unseen_context_rate"], 0.4)

    def test_baseline_requires_every_unique_independently_seeded_context_stratum(self) -> None:
        context_cells = [
            {
                "policy": policy,
                "archetype": archetype,
                "device_type": device,
                "traffic_source": traffic,
                "n_sessions": 10,
                "diagnostics": {
                    "decisions": 1,
                    "decisions_with_oov_tokens": 0,
                    "fallback_decisions": 0,
                    "serving_equivalent": True,
                },
            }
            for policy in manifest_builder.BASELINE_POLICIES
            for archetype in manifest_builder.BASELINE_ARCHETYPES
            for device in manifest_builder.BASELINE_DEVICES
            for traffic in manifest_builder.BASELINE_TRAFFIC
        ]
        payload = {
            "git_head": "a" * 40,
            "design": {
                "n_cells": 10,
                "sessions_per_cell": 10,
                "context_standardization": {
                    "enabled": True,
                    "sessions_per_stratum": 10,
                    "n_strata_cells": 160,
                    "independent_seeds_across_strata_cells": True,
                },
            },
            "inputs": {},
            "cells": [],
            "context_strata": context_cells,
        }
        artifacts = {
            name: {"sha256": "hash"}
            for name in ("v2_db", "v3_checkpoint", "archetype_config")
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            complete = manifest_builder.inspect_baseline(path, artifacts, "a" * 40)
            self.assertTrue(complete["context_standardization"]["complete"])

            payload["context_strata"][-1] = dict(payload["context_strata"][0])
            path.write_text(json.dumps(payload), encoding="utf-8")
            duplicate = manifest_builder.inspect_baseline(path, artifacts, "a" * 40)
            self.assertFalse(duplicate["context_standardization"]["complete"])
            self.assertEqual(
                duplicate["context_standardization"]["duplicate_strata_cells"], 1
            )


if __name__ == "__main__":
    unittest.main()
