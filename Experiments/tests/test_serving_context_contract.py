from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ServingContextContractTests(unittest.TestCase):
    def test_shop_clients_keep_context_history_per_session(self) -> None:
        for site in ("DemoSiteV2", "DemoSiteV3"):
            with self.subTest(site=site):
                decision_js = (
                    REPO_ROOT / site / "app" / "static" / "js" / "decision.js"
                ).read_text(encoding="utf-8")
                tracking_js = (
                    REPO_ROOT / site / "app" / "static" / "js" / "tracking.js"
                ).read_text(encoding="utf-8")

                self.assertIn(
                    "return `${name}:${window.SESSION_ID || 'anonymous'}`;",
                    decision_js,
                )
                self.assertIn("const initialReferrer = getInitialReferrer();", decision_js)
                self.assertIn("referrer: initialReferrer", decision_js)
                self.assertNotIn("referrer: document.referrer", decision_js)
                self.assertIn(
                    "page_depth:${window.SESSION_ID || 'anonymous'}",
                    tracking_js,
                )


if __name__ == "__main__":
    unittest.main()
