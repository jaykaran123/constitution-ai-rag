"""Session tokens, administrator authentication and rate limiting.

* Browser sessions use server-issued, HMAC-signed tokens (``<session_id>.<signature>``),
  replacing the previous design where any client-chosen string was accepted as a session.
* Administrative endpoints (LLM health, traces, graph inspection, debug output) require
  ``ADMIN_API_TOKEN`` via the ``X-Admin-Token`` header. If no token is configured they are
  only available from the local machine in development.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request

from app.config import get_settings
from app.database import get_db
from app.models import AccountSession
from sqlalchemy.orm import Session

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ACCOUNT_SESSION_ID = re.compile(r"^u_([a-f0-9]{32})_(\d{10})_([A-Za-z0-9_-]{8,16})$")
ACCOUNT_SESSION_TTL = 7 * 24 * 60 * 60


def _sign(session_id: str) -> str:
    key = get_settings().app_secret_key.encode("utf-8")
    return hmac.new(key, session_id.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def issue_session(previous: str | None = None) -> str:
    """Issue a signed token. In development, a legacy unsigned browser id may be upgraded so that
    existing local conversation history remains accessible."""
    settings = get_settings()
    session_id = None
    if previous and settings.app_env == "development" and SESSION_ID.match(previous):
        session_id = previous
    session_id = session_id or secrets.token_urlsafe(18).replace(".", "_")
    return f"{session_id}.{_sign(session_id)}"


def issue_account_session(user_id: str) -> str:
    """Issue a signed, expiring token whose stable owner ID is the account."""
    uid = uuid.UUID(user_id).hex
    expires = int(time.time()) + ACCOUNT_SESSION_TTL
    session_id = f"u_{uid}_{expires}_{secrets.token_urlsafe(9)}"
    return f"{session_id}.{_sign(session_id)}"


def verify_session(token: str | None) -> str:
    if not token or "." not in token:
        raise HTTPException(401, "A valid session token is required. Reload the page to start a session.")
    session_id, signature = token.rsplit(".", 1)
    if not SESSION_ID.match(session_id) or not hmac.compare_digest(signature, _sign(session_id)):
        raise HTTPException(401, "The session token is invalid. Reload the page to start a new session.")
    account_match = ACCOUNT_SESSION_ID.fullmatch(session_id)
    if account_match:
        user_id, expires, _nonce = account_match.groups()
        if int(expires) < int(time.time()):
            raise HTTPException(401, "Your sign-in expired. Please sign in again.")
        return f"user_{user_id}"
    if session_id.startswith("u_"):
        raise HTTPException(401, "The account session is invalid. Please sign in again.")
    return session_id


def session_from_request(request: Request, x_session_token: str | None = Header(default=None),
                         db: Session = Depends(get_db)) -> str:
    token = x_session_token or request.query_params.get("session_token")
    owner_id = verify_session(token)
    if owner_id.startswith("user_"):
        subject = token.partition(".")[0]
        session = db.get(AccountSession, subject)
        if not session or session.user_id.replace("-", "") != owner_id[5:] or session.revoked_at:
            raise HTTPException(401, "Your sign-in is no longer active. Please sign in again.")
    return owner_id


def is_admin(request: Request, token: str | None) -> bool:
    settings = get_settings()
    if settings.admin_api_token:
        return bool(token) and hmac.compare_digest(token, settings.admin_api_token)
    client = request.client.host if request.client else ""
    return settings.app_env == "development" and client in {"127.0.0.1", "::1", "localhost", "testclient"}


def require_admin(request: Request, x_admin_token: str | None = Header(default=None)) -> None:
    if not is_admin(request, x_admin_token):
        raise HTTPException(403, "Administrator access is required.")


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        settings = get_settings()
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > settings.rate_limit_window:
                hits.popleft()
            if len(hits) >= settings.rate_limit_requests:
                raise HTTPException(429, "Too many requests. Please wait a moment and try again.")
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()
