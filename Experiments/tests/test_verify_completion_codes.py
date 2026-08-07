from __future__ import annotations

import unittest

from Experiments.verify_completion_codes import Session, verify_submission


class CompletionCodeCompensationTests(unittest.TestCase):
    def test_matching_code_is_valid_even_when_quality_flags_fail(self):
        session = Session(
            condition="v2",
            worker_id="worker-1",
            persona="explorer",
            session_id="session-1",
            code="123456",
            passed=False,
            page_views=1,
            duration_s=5.0,
            has_dwell_or_dismiss=False,
            code_source="recorded",
        )

        status, detail = verify_submission(
            "worker-1",
            "123456",
            {"worker-1": [session]},
            {"123456": session},
        )

        self.assertEqual(status, "VALID")
        self.assertIn("sensitivity flags only", detail)


if __name__ == "__main__":
    unittest.main()
