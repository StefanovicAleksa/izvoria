"""
Password hashing with the standard library's scrypt: no native dependency on the gateway, and memory-hard.
Stored as `scrypt$n$r$p$salt$hash` (base64), so the cost can be raised later without breaking existing hashes.
"""

import base64
import hashlib
import hmac
import secrets

N, R, P = 2 ** 14, 8, 1
KEY_BYTES = 32
MIN_LENGTH = 10


class WeakPassword(ValueError):
    pass


def check_strength(password: str) -> None:
    if len(password or "") < MIN_LENGTH:
        raise WeakPassword(f"Passwords need at least {MIN_LENGTH} characters.")


def hash_password(password: str) -> str:
    check_strength(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=KEY_BYTES)
    return "$".join(["scrypt", str(N), str(R), str(P),
                     base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt((password or "").encode(), salt=base64.b64decode(salt),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Compared against when the email is unknown, so a login takes as long whether or not the account exists
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
