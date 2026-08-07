from __future__ import annotations

import asyncio
import tempfile
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

from StudyDispatcher.app import admin, config, store


class AdminFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        names = (
            "DB_PATH",
            "ADMIN_USERNAME",
            "ADMIN_PASSWORD",
            "ADMIN_SESSION_SECRET",
            "ADMIN_COOKIE_NAME",
            "ADMIN_COOKIE_DOMAIN",
            "ADMIN_COOKIE_SECURE",
            "ADMIN_SESSION_TTL_SECONDS",
            "V2_BASE_URL",
            "V3_BASE_URL",
            "V2_ANALYTICS_INTERNAL_URL",
            "V3_ANALYTICS_INTERNAL_URL",
        )
        self.original = {name: getattr(config, name) for name in names}
        config.DB_PATH = str(Path(self.temp_dir.name) / "dispatcher.db")
        config.ADMIN_USERNAME = "researcher"
        config.ADMIN_PASSWORD = "correct horse battery staple"
        config.ADMIN_SESSION_SECRET = "session-secret-which-is-long-and-random"
        config.ADMIN_COOKIE_NAME = "study_admin_session"
        config.ADMIN_COOKIE_DOMAIN = ""
        config.ADMIN_COOKIE_SECURE = True
        config.ADMIN_SESSION_TTL_SECONDS = 3600
        config.V2_BASE_URL = "https://v2.test"
        config.V3_BASE_URL = "https://v3.test"
        config.V2_ANALYTICS_INTERNAL_URL = "http://v2:8000"
        config.V3_ANALYTICS_INTERNAL_URL = "http://v3:8000"
        store.init()

    def tearDown(self):
        for name, value in self.original.items():
            setattr(config, name, value)
        self.temp_dir.cleanup()

    @staticmethod
    def _request(
        method: str,
        path: str,
        *,
        body: bytes = b"",
        cookie: str = "",
    ) -> Request:
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        headers = [(b"host", b"start.test")]
        if body:
            headers.append((b"content-type", b"application/x-www-form-urlencoded"))
        if cookie:
            headers.append((b"cookie", cookie.encode("ascii")))
        return Request(
            {
                "type": "http",
                "method": method,
                "path": path,
                "query_string": b"",
                "scheme": "https",
                "headers": headers,
                "client": ("127.0.0.1", 12345),
                "server": ("start.test", 443),
            },
            receive,
        )

    def _login(self):
        body = (
            "username=researcher&password=correct+horse+battery+staple&next=%2Fadmin"
        ).encode("ascii")
        response = asyncio.run(
            admin.admin_login_submit(
                self._request("POST", "/admin/login", body=body)
            )
        )
        parsed = SimpleCookie()
        parsed.load(response.headers["set-cookie"])
        cookie = parsed[config.ADMIN_COOKIE_NAME]
        return response, f"{config.ADMIN_COOKIE_NAME}={cookie.value}"

    def test_login_issues_secure_cookie_and_unlocks_dashboard(self):
        locked = admin.admin_dashboard(self._request("GET", "/admin"))
        self.assertEqual(locked.status_code, 303)
        self.assertIn("/admin/login", locked.headers["location"])

        bad = asyncio.run(
            admin.admin_login_submit(
                self._request(
                    "POST",
                    "/admin/login",
                    body=b"username=researcher&password=wrong&next=%2Fadmin",
                )
            )
        )
        self.assertEqual(bad.status_code, 401)

        logged_in, cookie_header = self._login()
        self.assertEqual(logged_in.status_code, 303)
        cookie = logged_in.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("SameSite=lax", cookie)

        dashboard = admin.admin_dashboard(
            self._request("GET", "/admin", cookie=cookie_header)
        )
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Combined V2 contextual-bandit", dashboard.body)
        self.assertEqual(dashboard.headers["cache-control"], "no-store")

    def test_login_rejects_external_next_url(self):
        response = asyncio.run(
            admin.admin_login_submit(
                self._request(
                    "POST",
                    "/admin/login",
                    body=(
                        b"username=researcher&password=correct+horse+battery+staple"
                        b"&next=https%3A%2F%2Fattacker.example%2Fsteal"
                    ),
                )
            )
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin")
        self.assertEqual(admin._safe_next(r"/\attacker.example"), "/admin")

    def test_overview_combines_both_condition_summaries(self):
        _response, cookie_header = self._login()

        def fake_fetch(label, _internal, public, _token):
            multiplier = 1 if label == "v2" else 2
            return {
                "ok": True,
                "dashboard_url": f"{public}/analytics",
                "summary": {
                    "study_participants": 3 * multiplier,
                    "completed_sessions": 2 * multiplier,
                    "attention_passed": multiplier,
                    "total_decisions": 10 * multiplier,
                    "unique_sessions": 3 * multiplier,
                    "total_events": 20 * multiplier,
                    "widget_clicks": multiplier,
                    "widget_dismissals": multiplier,
                    "add_to_cart_count": multiplier,
                    "purchases": multiplier,
                    "total_revenue": 12.5 * multiplier,
                },
            }

        with patch.object(admin, "_fetch_condition", side_effect=fake_fetch):
            response = admin.admin_overview(
                self._request("GET", "/admin/api/overview", cookie=cookie_header)
            )

        self.assertEqual(response.status_code, 200)
        import json

        payload = json.loads(response.body)
        self.assertEqual(payload["combined"]["study_participants"], 9)
        self.assertEqual(payload["combined"]["completed_sessions"], 6)
        self.assertEqual(payload["combined"]["total_decisions"], 30)
        self.assertEqual(payload["combined"]["total_revenue"], 37.5)
        self.assertEqual(set(payload["conditions"]), {"v2", "v3"})

    def test_counts_accepts_admin_session(self):
        _login, cookie_header = self._login()
        response = admin.admin_counts(
            self._request("GET", "/admin/counts", cookie=cookie_header)
        )
        self.assertEqual(response.status_code, 200)
        import json

        self.assertEqual(json.loads(response.body)["total"], 0)

    def test_dispatcher_cookie_is_accepted_by_both_condition_apps(self):
        from DemoSiteV2.app import config as v2_config
        from DemoSiteV2.app import main as v2_main
        from DemoSiteV3.app import config as v3_config
        from DemoSiteV3.app import main as v3_main
        from shared_schema.admin_session import create_admin_session

        originals = []
        try:
            for site_config in (v2_config, v3_config):
                originals.append(
                    (
                        site_config,
                        site_config.ADMIN_USERNAME,
                        site_config.ADMIN_SESSION_SECRET,
                        site_config.ADMIN_COOKIE_NAME,
                    )
                )
                site_config.ADMIN_USERNAME = config.ADMIN_USERNAME
                site_config.ADMIN_SESSION_SECRET = config.ADMIN_SESSION_SECRET
                site_config.ADMIN_COOKIE_NAME = config.ADMIN_COOKIE_NAME

            token = create_admin_session(
                config.ADMIN_SESSION_SECRET,
                config.ADMIN_USERNAME,
                ttl_seconds=60,
            )
            request = self._request(
                "GET",
                "/api/analytics/summary",
                cookie=f"{config.ADMIN_COOKIE_NAME}={token}",
            )
            self.assertTrue(v2_main._analytics_authorized(request))
            self.assertTrue(v3_main._analytics_authorized(request))

            tampered = self._request(
                "GET",
                "/api/analytics/summary",
                cookie=f"{config.ADMIN_COOKIE_NAME}={token}x",
            )
            self.assertFalse(v2_main._analytics_authorized(tampered))
            self.assertFalse(v3_main._analytics_authorized(tampered))
        finally:
            for site_config, username, secret, cookie_name in originals:
                site_config.ADMIN_USERNAME = username
                site_config.ADMIN_SESSION_SECRET = secret
                site_config.ADMIN_COOKIE_NAME = cookie_name


if __name__ == "__main__":
    unittest.main()
