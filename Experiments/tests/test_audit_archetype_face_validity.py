from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Experiments.audit_archetype_face_validity import audit, write_outputs


REPO_ROOT = Path(__file__).resolve().parents[2]


class ArchetypeFaceValidityAuditTests(unittest.TestCase):
    def test_audit_is_deterministic_and_covers_every_archetype(self) -> None:
        config = REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml"
        first = audit(config, sessions_per_archetype=5, base_seed=7)
        second = audit(config, sessions_per_archetype=5, base_seed=7)
        self.assertEqual(first["cells"], second["cells"])
        self.assertEqual(len(first["cells"]), 5)
        self.assertEqual(first["design"]["mechanism_strengths"]["fatigue_rate"], 0.0)

    def test_writer_refuses_overwrite(self) -> None:
        payload = {"cells": [{"archetype": "test", "conversion_rate": 0.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            write_outputs(payload, path)
            with self.assertRaises(FileExistsError):
                write_outputs(payload, path)


if __name__ == "__main__":
    unittest.main()
