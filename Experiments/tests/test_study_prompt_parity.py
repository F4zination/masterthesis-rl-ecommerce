from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _scenario_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("{% set _scenarios")
    end = text.index("} %}", start) + len("} %}")
    return text[start:end]


def _study_bar_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("{# ── Study session-end bar")
    end = text.index("{% endblock %}", start) + len("{% endblock %}")
    return text[start:end]


class StudyPromptParityTests(unittest.TestCase):
    def test_v2_and_v3_show_identical_persona_prompts(self) -> None:
        v2 = ROOT / "DemoSiteV2" / "app" / "templates" / "home.html"
        v3 = ROOT / "DemoSiteV3" / "app" / "templates" / "home.html"
        self.assertEqual(_scenario_block(v2), _scenario_block(v3))

    def test_v2_and_v3_offer_identical_session_end_access(self) -> None:
        v2 = ROOT / "DemoSiteV2" / "app" / "templates" / "base.html"
        v3 = ROOT / "DemoSiteV3" / "app" / "templates" / "base.html"
        block = _study_bar_block(v2)
        self.assertEqual(block, _study_bar_block(v3))
        self.assertNotIn("MIN_SECONDS", block)
        self.assertNotIn("disabled", block)

    def test_both_clients_record_decision_request_failures(self) -> None:
        for version in ("DemoSiteV2", "DemoSiteV3"):
            script = (
                ROOT / version / "app" / "static" / "js" / "decision.js"
            ).read_text(encoding="utf-8")
            self.assertIn("opportunity_error", script)
            self.assertIn("if (!res.ok)", script)
            self.assertIn("network_or_parse_error", script)


if __name__ == "__main__":
    unittest.main()
