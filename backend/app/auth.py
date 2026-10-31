"""Password helpers and account/session identity conversion."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1


def normalize_email(email: str) -> str:
    value = email.strip().casefold()
    if len(value) > 254 or not EMAIL_RE.fullmatch(value):
        raise ValueError("Enter a valid email address.")
    return value


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                            p=_SCRYPT_P, dklen=64, maxmem=64 * 1024 * 1024)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt_hex, expected_hex = stored.split("$", 2)
        if algorithm != "scrypt":
            return False
        salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(expected_hex)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                                p=_SCRYPT_P, dklen=len(expected), maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def account_owner_id(user_id: str) -> str:
    return f"user_{uuid.UUID(user_id).hex}"


def user_id_from_owner(owner_id: str) -> str | None:
    if not owner_id.startswith("user_"):
        return None
    try:
        return str(uuid.UUID(hex=owner_id[5:]))
    except ValueError:
        return None
