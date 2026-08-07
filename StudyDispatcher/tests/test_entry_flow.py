from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from starlette.requests import Request

from StudyDispatcher.app import config, main, store


def _form_request(body: bytes) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/start",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/x-www-form-urlencoded"),
            ],
        },
        receive,
    )


def _get_request(path: str, host: str | None = None) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": headers,
        },
        receive,
    )


class EntryFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = config.DB_PATH
        self.original_v2_url = config.V2_BASE_URL
        self.original_v3_url = config.V3_BASE_URL
        self.original_conditions = config.CONDITIONS

        config.DB_PATH = str(Path(self.temp_dir.name) / "dispatcher.db")
        config.V2_BASE_URL = "https://v2.test"
        config.V3_BASE_URL = "https://v3.test"
        config.CONDITIONS = {
            "v2": config.V2_BASE_URL,
            "v3": config.V3_BASE_URL,
        }
        store.init()

    def tearDown(self):
        config.DB_PATH = self.original_db_path
        config.V2_BASE_URL = self.original_v2_url
        config.V3_BASE_URL = self.original_v3_url
        config.CONDITIONS = self.original_conditions
        self.temp_dir.cleanup()

    def test_entry_and_legal_pages_are_present(self):
        entry = main.index().body.decode("utf-8")
        self.assertIn('action="/start"', entry)
        self.assertIn('name="consent"', entry)
        self.assertIn("Start study*", entry)
        self.assertIn('href="/imprint"', entry)

        self.assertIn(
            "Provider information",
            main.imprint(_get_request("/imprint")).body.decode("utf-8"),
        )
        self.assertIn(
            "Article 6(1)(a)",
            main.privacy(_get_request("/privacy")).body.decode("utf-8"),
        )
        self.assertIn(
            "Voluntary consent",
            main.participant_information(
                _get_request("/participant-information")
            ).body.decode("utf-8"),
        )

    def test_persona_subset_comes_from_env_and_fails_closed(self):
        """STUDY_PERSONAS restricts recruitment without touching the sites.

        The August 2026 redesign assigns three of the five scenario cells; the
        unlisted scenarios stay deployed and digest-locked but are never
        assigned. A typo must raise rather than silently recruit the wrong
        cells, because the shops would reject the persona and every affected
        assignment would be lost.
        """
        import os

        original = os.environ.get("STUDY_PERSONAS")
        try:
            os.environ["STUDY_PERSONAS"] = "fastbuyer, DetailedComparator ,windowshopper"
            self.assertEqual(
                config._personas_from_env(),
                ["fastbuyer", "detailedcomparator", "windowshopper"],
            )
            os.environ["STUDY_PERSONAS"] = ""
            self.assertEqual(config._personas_from_env(), config.KNOWN_PERSONAS)
            os.environ["STUDY_PERSONAS"] = "fastbuyer,fastbuyer"
            with self.assertRaises(ValueError):
                config._personas_from_env()
            os.environ["STUDY_PERSONAS"] = "fastbuyer,windowshoper"
            with self.assertRaises(ValueError):
                config._personas_from_env()
        finally:
            if original is None:
                os.environ.pop("STUDY_PERSONAS", None)
            else:
                os.environ["STUDY_PERSONAS"] = original

    def test_legal_back_link_returns_to_the_shop_on_shop_hosts(self):
        """A legal page opened mid-session must lead back to the shop.

        The gateway proxies the legal routes on both condition domains, so a
        participant can reach them without leaving their session. Sending them
        back to the study entry form let them re-register under a new nickname,
        which minted a second assignment and unbalanced the randomized cells.
        """
        on_dispatcher = main.privacy(
            _get_request("/privacy", "start.test")
        ).body.decode("utf-8")
        self.assertIn("Back to study entry", on_dispatcher)

        for host in ("v2.test", "v3.test", "V2.TEST", "v2.test:8443"):
            page = main.privacy(_get_request("/privacy", host)).body.decode("utf-8")
            self.assertIn("Back to the shop", page)
            self.assertNotIn("Back to study entry", page)

    def test_start_records_consent_and_redirects(self):
        request = _form_request(b"nickname=Blue+Otter&consent=yes")
        response = asyncio.run(main.start(request))

        self.assertEqual(response.status_code, 303)
        query = parse_qs(urlparse(response.headers["location"]).query)
        participant_id = query["wid"][0]
        self.assertTrue(participant_id.startswith("blue-otter-"))
        self.assertIn(query["persona"][0], config.PERSONAS)

        assignment = store.get(participant_id)
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.consent_version, config.CONSENT_VERSION)
        self.assertIsNotNone(assignment.consented_at)

    def test_start_rejects_missing_consent(self):
        request = _form_request(b"nickname=Blue+Otter")
        response = asyncio.run(main.start(request))

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Consent required", response.body)
        self.assertEqual(sum(store.counts().values()), 0)


class StoreMigrationTests(unittest.TestCase):
    def test_existing_assignment_ledger_gets_consent_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = Path(temp_dir) / "old-dispatcher.db"
            conn = sqlite3.connect(old_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE assignments (
                        pid TEXT PRIMARY KEY,
                        wid TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        persona TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                    ("existing", "existing", "v2", "explorer", 1.0),
                )
                conn.commit()
            finally:
                conn.close()

            original_db_path = config.DB_PATH
            try:
                config.DB_PATH = str(old_path)
                store.init()
                assignment = store.get("existing")
            finally:
                config.DB_PATH = original_db_path

            self.assertIsNotNone(assignment)
            self.assertEqual(assignment.consent_version, "")
            self.assertIsNone(assignment.consented_at)


class BlockedAssignmentTests(unittest.TestCase):
    def test_personas_and_policy_arms_stay_balanced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_db_path = config.DB_PATH
            try:
                config.DB_PATH = str(Path(temp_dir) / "blocked.db")
                store.init()
                for index in range(73):
                    store.assign(f"worker-{index}")
                tally = store.counts()
            finally:
                config.DB_PATH = original_db_path

        persona_totals = [
            sum(tally[(condition, persona)] for condition in config.CONDITIONS)
            for persona in config.PERSONAS
        ]
        self.assertLessEqual(max(persona_totals) - min(persona_totals), 1)
        for persona in config.PERSONAS:
            arm_counts = [tally[(condition, persona)] for condition in config.CONDITIONS]
            self.assertLessEqual(max(arm_counts) - min(arm_counts), 1)


if __name__ == "__main__":
    unittest.main()
