"""
How the gateway vouches for a user to the GPU box.

The box API is reachable from the internet (Vast maps its port), so it trusts nothing a browser could send. The
gateway, having checked the session cookie, adds the firm, user and role to every forwarded request and signs them
together with the method, path, query and a timestamp. The body is not signed, so uploads stream through unbuffered;
a signature is only good for MAX_SKEW_SECONDS, which bounds replay. Standard library only.
"""

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Mapping, Optional

from web_deploy.common.tenancy import InvalidSlug, validate_slug

MAX_SKEW_SECONDS = 60

H_FIRM = "x-onyx-firm"
H_USER = "x-onyx-user"
H_ROLE = "x-onyx-role"
H_TS = "x-onyx-ts"
H_SIG = "x-onyx-sig"

ROLES = frozenset({"member", "admin", "owner"})


class SignatureError(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    """Who is asking: a user of one firm. `user_id` is the gateway's stable id, never an email."""
    firm: str
    user_id: str
    role: str


def _message(method: str, path: str, query: str, principal: Principal, ts: str) -> bytes:
    return "\n".join([method.upper(), path, query, principal.firm, principal.user_id, principal.role, ts]).encode()


def sign(secret: str, method: str, path: str, query: str, principal: Principal,
         now: Optional[float] = None) -> dict:
    """The headers to add to a request forwarded on behalf of `principal`."""
    ts = str(int(now if now is not None else time.time()))
    digest = hmac.new(secret.encode(), _message(method, path, query, principal, ts), hashlib.sha256).hexdigest()
    return {H_FIRM: principal.firm, H_USER: principal.user_id, H_ROLE: principal.role, H_TS: ts, H_SIG: digest}


def verify(secret: str, method: str, path: str, query: str, headers: Mapping[str, str],
           now: Optional[float] = None) -> Principal:
    """The principal a request was signed for. Raises SignatureError for anything unsigned, altered or stale."""
    if not secret:
        raise SignatureError("The API has no signing secret configured.")
    lowered = {k.lower(): v for k, v in headers.items()}
    try:
        principal = Principal(firm=lowered[H_FIRM], user_id=lowered[H_USER], role=lowered[H_ROLE])
        ts, signature = lowered[H_TS], lowered[H_SIG]
    except KeyError:
        raise SignatureError("The request is not signed.")
    if not ts.isdigit() or abs((now if now is not None else time.time()) - int(ts)) > MAX_SKEW_SECONDS:
        raise SignatureError("The request signature has expired.")
    expected = hmac.new(secret.encode(), _message(method, path, query, principal, ts), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureError("The request signature does not match.")
    try:
        validate_slug(principal.firm)
    except InvalidSlug as exc:
        raise SignatureError(str(exc))
    if principal.role not in ROLES or not principal.user_id or len(principal.user_id) > 64:
        raise SignatureError("The request names an unknown role or user.")
    return principal


# ---------- box-internal service calls (the ingestion worker asking the API to give up the GPU) ----------

H_SERVICE_TS = "x-onyx-service-ts"
H_SERVICE_SIG = "x-onyx-service-sig"


def _service_message(method: str, path: str, ts: str) -> bytes:
    return "\n".join(["service", method.upper(), path, ts]).encode()


def sign_service(secret: str, method: str, path: str, now: Optional[float] = None) -> dict:
    ts = str(int(now if now is not None else time.time()))
    digest = hmac.new(secret.encode(), _service_message(method, path, ts), hashlib.sha256).hexdigest()
    return {H_SERVICE_TS: ts, H_SERVICE_SIG: digest}


def verify_service(secret: str, method: str, path: str, headers: Mapping[str, str],
                   now: Optional[float] = None) -> None:
    """Raises SignatureError unless the request was signed by a service holding the API secret."""
    lowered = {k.lower(): v for k, v in headers.items()}
    ts, signature = lowered.get(H_SERVICE_TS, ""), lowered.get(H_SERVICE_SIG, "")
    if not secret or not ts.isdigit() or not signature:
        raise SignatureError("The service request is not signed.")
    if abs((now if now is not None else time.time()) - int(ts)) > MAX_SKEW_SECONDS:
        raise SignatureError("The service request signature has expired.")
    expected = hmac.new(secret.encode(), _service_message(method, path, ts), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureError("The service request signature does not match.")
