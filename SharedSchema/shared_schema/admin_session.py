"""Signed admin-session cookies shared by the study services.

The implementation intentionally uses only the Python standard library.  The
dispatcher issues a token after a successful login; V2 and V3 verify the same
token with a shared secret before serving their analytics routes.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def create_admin_session(
    secret: str,
    username: str,
    *,
    ttl_seconds: int = 28_800,
    now: int | None = None,
) -> str:
    """Create a compact HMAC-SHA256 signed admin-session token."""
    if not secret:
        raise ValueError("admin session secret must not be empty")
    if not username:
        raise ValueError("admin username must not be empty")

    issued_at = int(time.time() if now is None else now)
    payload: dict[str, Any] = {
        "exp": issued_at + max(1, int(ttl_seconds)),
        "iat": issued_at,
        "sub": username,
        "v": 1,
    }
    encoded = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{_b64encode(signature)}"


def verify_admin_session(
    token: str,
    secret: str,
    *,
    expected_username: str = "",
    now: int | None = None,
) -> bool:
    """Return whether ``token`` is authentic, unexpired, and for the admin."""
    if not token or not secret:
        return False
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _b64encode(
            hmac.new(
                secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return False
        payload = json.loads(_b64decode(encoded).decode("utf-8"))
        expires_at = int(payload["exp"])
        username = str(payload["sub"])
        version = int(payload["v"])
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False

    current_time = int(time.time() if now is None else now)
    if version != 1 or expires_at <= current_time:
        return False
    if expected_username and not hmac.compare_digest(username, expected_username):
        return False
    return True
