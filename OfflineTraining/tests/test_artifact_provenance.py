from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from OfflineTraining import build_bandit_policy


class ArtifactProvenanceTests(unittest.TestCase):
    def test_bandit_builder_embeds_dataset_and_build_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "transitions.jsonl"
            dataset.write_text('{"reward": 1}\n', encoding="utf-8")
            database = root / "policy.db"
            sqlite3.connect(database).close()

            with patch.object(
                build_bandit_policy,
                "_git_state",
                return_value=("a" * 40, False),
            ):
                build_bandit_policy._write_policy_metadata(
                    database,
                    dataset_path=dataset,
                    n_rows=1,
                    n_skipped=0,
                    schema_version=2,
                    seed_products=True,
                    arm_support={
                        "coverage_validation_passed": True,
                        "required_serving_contexts": 848,
                    },
                )

            connection = sqlite3.connect(database)
            try:
                metadata = {
                    key: json.loads(value)
                    for key, value in connection.execute(
                        "SELECT key, value FROM policy_build_metadata"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(metadata["dataset_sha256"], build_bandit_policy._sha256(dataset))
            self.assertEqual(metadata["build_git_commit"], "a" * 40)
            self.assertEqual(metadata["n_transitions_used"], 1)
            self.assertTrue(metadata["product_catalog_seeded"])
            self.assertEqual(metadata["provenance_schema_version"], 2)
            self.assertEqual(metadata["arm_support"]["required_serving_contexts"], 848)


if __name__ == "__main__":
    unittest.main()
