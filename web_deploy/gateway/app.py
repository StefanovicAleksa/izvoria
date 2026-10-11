"""
The always-on gateway on the Hetzner VPS: who you are, which firm you belong to, and the door to the GPU box.

Everyone uses one address. Signing in takes two steps: the firm (its sign-in name and the password its people share)
gives the browser a firm pass (`onyx_firm`), and a person then signs in with their own email and password within
that firm (`onyx_session`). The owner's console is the firm "admin". A firm's administrators manage its accounts
(/firm/api); the owner manages firms (/admin/api). Two public pages need no sign-in: a firm asking for Izvoria, and a
help request (/public), both stored here and emailed to the owner (mailer.py).

Signing in wakes the GPU box. Once it is ready, /api/* is forwarded to it, signed for the user (proxy.py); the box
never sees a cookie or a password.

Run: uvicorn web_deploy.gateway.app:create_production_app --factory --host 127.0.0.1 --port 8000 (behind Caddy)
"""

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from web_deploy.common.signing import Principal
from web_deploy.common.tenancy import InvalidSlug
from web_deploy.gateway.box import BoxController, BoxState, box_ssl_context, build_vast
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.idle import IdleMonitor
from web_deploy.gateway.mailer import Mailer
from web_deploy.gateway.passwords import WeakPassword
from web_deploy.gateway.proxy import forward
from web_deploy.gateway.store import (
    OWNER_FIRM, SESSION_SECONDS, Store, StoreError, User, normalize_email,
)

logger = logging.getLogger(__name__)

COOKIE = "onyx_session"
FIRM_COOKIE = "onyx_firm"
# A firm pass lets a browser on to the account step. Short unless the device is remembered.
FIRM_PASS_SECONDS = 20 * 60
FIRM_REMEMBER_SECONDS = 30 * 24 * 3600
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 8
# Higher per firm than per address: the firm password is shared, so anyone could otherwise lock a whole firm out
FIRM_MAX_FAILURES = 30
PUBLIC_WINDOW_SECONDS = 3600
PUBLIC_MAX_PER_ADDRESS = 5
DAILY_MAIL_CAP = 100
# A person takes longer than this to fill in either public form; a script does not
MIN_FILL_MS = 2500
SIGNUP_MAX_BYTES = 32 * 1024
SUPPORT_MAX_BYTES = 16 * 1024
LANGUAGES = ("en", "sr")

_PROFILE_TEXT = ("industry", "sub_domains", "description", "data_kinds", "mishearings")
_PROPER_NOUNS = ("people", "clients", "products", "places", "acronyms", "laws")
_PROFILE_LANGUAGES = ("share", "english_terms", "latin")


class Limiter:
    """
    At most limits[kind] events per key of that kind in a sliding window: failed sign-ins per address and per
    account, or submissions of a public form per address. In memory, for the gateway's single process.
    """

    def __init__(self, window_seconds: float, limits: Dict[str, int], clock: Callable[[], float] = time.monotonic):
        self._window = window_seconds
        self._limits = limits
        self._clock = clock
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    def _count(self, key: Tuple[str, str]) -> int:
        times = self._events.get(key)
        if times is None:
            return 0
        cutoff = self._clock() - self._window
        while times and times[0] < cutoff:
            times.popleft()
        if not times:
            del self._events[key]  # never grows with addresses that have gone quiet
            return 0
        return len(times)

    def blocked(self, **keys: str) -> bool:
        return any(self._count((kind, value)) >= self._limits[kind] for kind, value in keys.items())

    def hit(self, **keys: str) -> None:
        now = self._clock()
        for kind, value in keys.items():
            self._count((kind, value))
            self._events[(kind, value)].append(now)

    def clear(self, kind: str, value: str) -> None:
        self._events.pop((kind, value), None)


class PayloadError(ValueError):
    """A public form that cannot be accepted as sent: the message is shown to the person who sent it."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _text(data: Dict[str, Any], key: str, limit: int, required: str = "") -> str:
    value = data.get(key)
    value = value.strip() if isinstance(value, str) else ""
    if required and not value:
        raise PayloadError(required)
    if len(value) > limit:
        raise PayloadError(f"'{key}' is too long (at most {limit} characters).")
    return value


def _section(data: Any, keys: Tuple[str, ...], limit: int) -> Dict[str, str]:
    """Only the fields the form has, as text: anything else a client sends is dropped."""
    data = data if isinstance(data, dict) else {}
    return {key: _text(data, key, limit) for key in keys}


def signup_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
    return {
        "firm_name": _text(data, "firm_name", 200, "Enter your firm's name."),
        "contact_name": _text(data, "contact_name", 200, "Enter your name."),
        "contact_role": _text(data, "contact_role", 200),
        "phone": _text(data, "phone", 50),
        "profile": {
            **_section(profile, _PROFILE_TEXT, 4000),
            "proper_nouns": _section(profile.get("proper_nouns"), _PROPER_NOUNS, 4000),
            "languages": _section(profile.get("languages"), _PROFILE_LANGUAGES, 2000),
        },
    }


def support_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": _text(data, "name", 200),
        "firm": _text(data, "firm", 100),
        "subject": _text(data, "subject", 150, "Enter a subject."),
        "message": _text(data, "message", 8000, "Describe the problem."),
    }


async def read_limited(request: Request, max_bytes: int) -> bytes:
    """The body, refused once it passes max_bytes (whatever Content-Length claimed)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise PayloadError("That is more than the form can take.", 413)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise PayloadError("That is more than the form can take.", 413)
    return bytes(body)


def create_app(settings: GatewaySettings, store: Store, box: BoxController, idle: IdleMonitor,
               http: httpx.AsyncClient, run_idle_monitor: bool = True, mailer: Optional[Mailer] = None) -> Starlette:
    mailer = mailer or Mailer(settings.mail, store)
    account_limiter = Limiter(LOGIN_WINDOW_SECONDS, {"ip": LOGIN_MAX_FAILURES, "account": LOGIN_MAX_FAILURES})
    firm_limiter = Limiter(LOGIN_WINDOW_SECONDS, {"ip": LOGIN_MAX_FAILURES, "firm": FIRM_MAX_FAILURES})
    public_limiter = Limiter(PUBLIC_WINDOW_SECONDS, {"signup": PUBLIC_MAX_PER_ADDRESS,
                                                     "support": PUBLIC_MAX_PER_ADDRESS})

    # ---------- who ----------

    def address_of(request: Request) -> str:
        return request.client.host if request.client else "?"

    def current_user(request: Request) -> Optional[User]:
        """The signed-in user, while their account and their firm are enabled. The session alone decides the firm."""
        user = store.session_user(request.cookies.get(COOKIE))
        if user is None:
            return None
        if user.firm_slug != OWNER_FIRM:
            firm = store.firm(user.firm_slug)
            if firm is None or firm.disabled:
                return None
        return user

    def same_origin(request: Request) -> bool:
        """State-changing requests must come from this site's own pages (CSRF): browsers always send Origin."""
        origin = request.headers.get("origin")
        if settings.public_url:
            return origin == settings.public_url
        host = request.headers.get("host", "")
        scheme = "https" if settings.https_only else request.url.scheme
        return origin == f"{scheme}://{host}"

    def refused() -> Response:
        return JSONResponse({"error": "Cross-site request refused."}, status_code=403)

    def guarded(handler: Callable[[Request, User], Awaitable[Response]], owner: bool = False,
                firm_admin: bool = False):
        async def wrapper(request: Request) -> Response:
            if request.method not in ("GET", "HEAD") and not same_origin(request):
                return refused()
            user = current_user(request)
            if user is None:
                return JSONResponse({"error": "Please sign in.", "kind": "auth"}, status_code=401)
            if owner and user.role != "owner":
                return JSONResponse({"error": "Only the owner can do that."}, status_code=403)
            if firm_admin and (user.role != "admin" or user.firm_slug == OWNER_FIRM):
                return JSONResponse({"error": "Only your firm's administrators can do that."}, status_code=403)
            try:
                return await handler(request, user)
            except (StoreError, WeakPassword, InvalidSlug) as exc:
                return JSONResponse({"error": str(exc), "kind": "invalid"}, status_code=400)
        return wrapper

    async def json_body(request: Request) -> dict:
        try:
            data = await request.json()
        except Exception:
            data = None
        if not isinstance(data, dict):
            raise StoreError("Expected a JSON object.")
        return data

    def session_cookie(response: Response, token: str) -> None:
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, secure=settings.https_only,
                            samesite="lax", path="/")

    def firm_public(slug: str) -> dict:
        if slug == OWNER_FIRM:
            return {"slug": OWNER_FIRM, "name": "Admin console"}
        firm = store.firm(slug)
        return {"slug": firm.slug, "name": firm.name, "default_language": firm.default_language}

    # ---------- signing in ----------

    async def me(request: Request) -> Response:
        """Where this browser is: no firm yet, a firm but no account, or signed in."""
        base = {"languages": list(LANGUAGES)}
        user = current_user(request)
        if user is not None:
            console = "admin" if user.firm_slug == OWNER_FIRM else "firm"
            return JSONResponse({**base, "stage": "signed_in", "console": console,
                                 "firm": firm_public(user.firm_slug), "user": user.public()})
        slug = store.firm_pass(request.cookies.get(FIRM_COOKIE))
        if slug is not None:
            return JSONResponse({**base, "stage": "account", "console": "admin" if slug == OWNER_FIRM else "firm",
                                 "firm": firm_public(slug), "user": None})
        return JSONResponse({**base, "stage": "firm", "console": None, "firm": None, "user": None})

    async def firm_signin(request: Request) -> Response:
        """Step one: the firm's sign-in name and shared password give this browser a firm pass."""
        if not same_origin(request):
            return refused()
        try:
            data = await json_body(request)
        except StoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        slug = str(data.get("firm") or "").strip().lower()[:64]
        if not slug:
            return JSONResponse({"error": "Enter your firm's sign-in name."}, status_code=400)
        address = address_of(request)
        if firm_limiter.blocked(ip=address, firm=slug):
            return JSONResponse({"error": "Too many attempts. Wait a few minutes and try again."}, status_code=429)
        verified = await asyncio.to_thread(store.verify_firm, slug, str(data.get("password") or ""))
        if verified is None:
            firm_limiter.hit(ip=address, firm=slug)
            return JSONResponse({"error": "Wrong firm name or password."}, status_code=401)
        firm_limiter.clear("firm", slug)
        seconds = FIRM_REMEMBER_SECONDS if data.get("remember") else FIRM_PASS_SECONDS
        token = store.create_firm_pass(verified, seconds)
        store.end_firm_pass(request.cookies.get(FIRM_COOKIE))
        response = JSONResponse({"firm": firm_public(verified)})
        response.set_cookie(FIRM_COOKIE, token, max_age=int(seconds), httponly=True, secure=settings.https_only,
                            samesite="strict", path="/")
        return response

    async def forget_firm(request: Request) -> Response:
        """Switch firm: this browser forgets both the firm pass and any session."""
        if not same_origin(request):
            return refused()
        store.end_session(request.cookies.get(COOKIE))
        store.end_firm_pass(request.cookies.get(FIRM_COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        response.delete_cookie(FIRM_COOKIE, path="/")
        return response

    async def login(request: Request) -> Response:
        """Step two: a person's own email and password, within the firm the pass is for (never one in the body)."""
        if not same_origin(request):
            return refused()
        firm = store.firm_pass(request.cookies.get(FIRM_COOKIE))
        if firm is None:
            return JSONResponse({"error": "Sign in as your firm first.", "kind": "firm_stage"}, status_code=401)
        try:
            data = await json_body(request)
        except StoreError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        email = str(data.get("email") or "").strip().lower()
        address = address_of(request)
        account = f"{firm}:{email}"
        if account_limiter.blocked(ip=address, account=account):
            return JSONResponse({"error": "Too many attempts. Wait a few minutes and try again."}, status_code=429)
        user = await asyncio.to_thread(store.authenticate, firm, email, str(data.get("password") or ""))
        if user is None:
            account_limiter.hit(ip=address, account=account)
            return JSONResponse({"error": "Wrong email or password."}, status_code=401)
        account_limiter.clear("account", account)
        store.end_session(request.cookies.get(COOKIE))
        response = JSONResponse({"user": user.public()})
        session_cookie(response, store.create_session(user.id))
        if firm != OWNER_FIRM and settings.box_mode == "vast":
            # Signing in wakes the rented box; a local box is always on
            idle.touch()
            box.start()
        return response

    async def logout(request: Request) -> Response:
        """Signs the person out; the firm pass stays, so the next person on this device starts at their account."""
        if not same_origin(request):
            return refused()
        store.end_session(request.cookies.get(COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    async def change_password(request: Request, user: User) -> Response:
        data = await json_body(request)
        if await asyncio.to_thread(store.authenticate, user.firm_slug, user.email,
                                   str(data.get("current") or "")) is None:
            return JSONResponse({"error": "The current password is wrong."}, status_code=400)
        await asyncio.to_thread(store.set_password, user.id, str(data.get("new") or ""))
        response = JSONResponse({"ok": True})
        session_cookie(response, store.create_session(user.id))  # set_password signed every session out
        return response

    # ---------- the box ----------

    async def box_status(request: Request, user: User) -> Response:
        idle.touch()
        return JSONResponse((await box.status()).as_json())

    async def box_start(request: Request, user: User) -> Response:
        idle.touch()
        box.start()
        return JSONResponse((await box.status()).as_json(), status_code=202)

    async def api(request: Request, user: User) -> Response:
        if user.firm_slug == OWNER_FIRM:
            return JSONResponse({"error": "The admin console has no appliance of its own."}, status_code=404)
        status = await box.status()
        if status.state is not BoxState.READY:
            if status.state is BoxState.STOPPED:
                box.start()
            return JSONResponse({"error": status.as_json()["message"], "kind": "box", "box": status.as_json()},
                                status_code=503)
        address = await box.api_address()
        principal = Principal(firm=user.firm_slug, user_id=user.id, role=user.role)
        cm = idle.in_flight()
        cm.__enter__()
        return await forward(request, http, f"{box.scheme}://{address}", settings.api_secret, principal,
                             on_close=lambda: cm.__exit__(None, None, None))

    # ---------- a firm's administrators ----------

    def firm_target(admin: User, user_id: str) -> Optional[User]:
        """A user of the admin's own firm; anyone else does not exist as far as this admin can tell."""
        return store.firm_user(admin.firm_slug, user_id)

    def is_last_admin(target: User) -> bool:
        return (target.role == "admin" and not target.disabled
                and store.enabled_admin_count(target.firm_slug) <= 1)

    async def firm_users(request: Request, user: User) -> Response:
        if request.method == "POST":
            data = await json_body(request)
            created = await asyncio.to_thread(
                store.create_user, user.firm_slug, str(data.get("email") or ""), str(data.get("password") or ""),
                str(data.get("role") or "member"), str(data.get("name") or ""))
            return JSONResponse(created.public(), status_code=201)
        return JSONResponse({"users": [u.public() for u in store.users(user.firm_slug)]})

    async def firm_user_action(request: Request, user: User) -> Response:
        target = firm_target(user, request.path_params["user_id"])
        if target is None:
            return JSONResponse({"error": "No such user."}, status_code=404)
        data = await json_body(request)
        action = request.path_params["action"]
        if action == "password":
            await asyncio.to_thread(store.set_password, target.id, str(data.get("password") or ""))
        elif action in ("disable", "enable"):
            if action == "disable":
                if target.id == user.id:
                    raise StoreError("You can't disable your own account.")
                if is_last_admin(target):
                    raise StoreError("Your firm needs at least one administrator.")
            store.set_user_disabled(target.id, action == "disable")
        elif action == "role":
            role = str(data.get("role") or "")
            if target.id == user.id:
                raise StoreError("You can't change your own role.")
            if role != "admin" and is_last_admin(target):
                raise StoreError("Your firm needs at least one administrator.")
            store.set_user_role(target.id, role)
        else:
            return JSONResponse({"error": "Unknown action."}, status_code=404)
        return JSONResponse(store.user(target.id).public())

    async def firm_password(request: Request, user: User) -> Response:
        """The firm's shared password. The admin confirms with their own password first."""
        data = await json_body(request)
        if await asyncio.to_thread(store.authenticate, user.firm_slug, user.email,
                                   str(data.get("account_password") or "")) is None:
            return JSONResponse({"error": "Your account password is wrong."}, status_code=400)
        await asyncio.to_thread(store.set_firm_password, user.firm_slug, str(data.get("password") or ""))
        return JSONResponse({"ok": True})

    # ---------- the owner's console ----------

    async def admin_firms(request: Request, user: User) -> Response:
        if request.method == "POST":
            data = await json_body(request)
            language = str(data.get("default_language") or "en")
            if language not in LANGUAGES:
                raise StoreError(f"The default language is one of: {', '.join(LANGUAGES)}.")
            password = str(data.get("firm_password") or "")
            if not password:
                raise StoreError("Give the firm a password: its people need it to sign in.")
            firm = await asyncio.to_thread(store.create_firm, str(data.get("slug") or ""),
                                           str(data.get("name") or ""), language, bool(data.get("is_demo")),
                                           password)
            return JSONResponse(firm.public(), status_code=201)
        firms = [{**firm.public(), "users": len(store.users(firm.slug))} for firm in store.firms()]
        return JSONResponse({"firms": firms, "console_has_password": store.has_console_password()})

    async def admin_firm_action(request: Request, user: User) -> Response:
        slug, action = request.path_params["slug"], request.path_params["action"]
        if store.firm(slug) is None:
            return JSONResponse({"error": "No such firm."}, status_code=404)
        if action == "password":
            data = await json_body(request)
            await asyncio.to_thread(store.set_firm_password, slug, str(data.get("password") or ""))
        elif action in ("disable", "enable"):
            store.set_firm_disabled(slug, action == "disable")
        else:
            return JSONResponse({"error": "Unknown action."}, status_code=404)
        return JSONResponse(store.firm(slug).public())

    async def admin_console_password(request: Request, user: User) -> Response:
        data = await json_body(request)
        if await asyncio.to_thread(store.authenticate, OWNER_FIRM, user.email,
                                   str(data.get("account_password") or "")) is None:
            return JSONResponse({"error": "Your account password is wrong."}, status_code=400)
        await asyncio.to_thread(store.set_firm_password, OWNER_FIRM, str(data.get("password") or ""))
        return JSONResponse({"ok": True})

    async def admin_users(request: Request, user: User) -> Response:
        slug = request.path_params["slug"]
        if store.firm(slug) is None:
            return JSONResponse({"error": "No such firm."}, status_code=404)
        if request.method == "POST":
            data = await json_body(request)
            created = await asyncio.to_thread(
                store.create_user, slug, str(data.get("email") or ""), str(data.get("password") or ""),
                str(data.get("role") or "member"), str(data.get("name") or ""))
            return JSONResponse(created.public(), status_code=201)
        return JSONResponse({"users": [u.public() for u in store.users(slug)]})

    async def admin_user(request: Request, user: User) -> Response:
        target = store.user(request.path_params["user_id"])
        if target is None or target.firm_slug == OWNER_FIRM:
            return JSONResponse({"error": "No such user."}, status_code=404)
        data = await json_body(request)
        action = request.path_params["action"]
        if action == "password":
            await asyncio.to_thread(store.set_password, target.id, str(data.get("password") or ""))
        elif action in ("disable", "enable"):
            store.set_user_disabled(target.id, action == "disable")
        elif action == "role":
            store.set_user_role(target.id, str(data.get("role") or ""))
        else:
            return JSONResponse({"error": "Unknown action."}, status_code=404)
        return JSONResponse(store.user(target.id).public())

    async def admin_requests(request: Request, user: User) -> Response:
        kind = request.query_params.get("kind") or None
        status = request.query_params.get("status") or None
        return JSONResponse({"requests": [r.public() for r in store.requests(kind, status)],
                             "mail_enabled": mailer.enabled})

    async def admin_request_action(request: Request, user: User) -> Response:
        inbound = store.request(request.path_params["request_id"])
        if inbound is None:
            return JSONResponse({"error": "No such request."}, status_code=404)
        action = request.path_params["action"]
        if action == "status":
            data = await json_body(request)
            store.set_request_status(inbound.id, str(data.get("status") or ""))
        elif action == "resend":
            if not mailer.enabled:
                raise StoreError("Email is not set up on this gateway (SMTP_HOST).")
            store.set_mail_result(inbound.id, "pending")
            mailer.submit(inbound)
        else:
            return JSONResponse({"error": "Unknown action."}, status_code=404)
        return JSONResponse(store.request(inbound.id).public())

    async def admin_box(request: Request, user: User) -> Response:
        if request.method == "POST":
            action = request.path_params.get("action")
            if action == "start":
                box.start()
            elif action == "stop":
                box.stop()
        return JSONResponse((await box.status()).as_json())

    # ---------- the public pages: a firm asking for Izvoria, and help ----------

    def public_form(kind: str, max_bytes: int, parse: Callable[[Dict[str, Any]], Dict[str, Any]]):
        async def handler(request: Request) -> Response:
            if not same_origin(request):
                return refused()
            address = address_of(request)
            if public_limiter.blocked(**{kind: address}):
                return JSONResponse({"error": "Too many requests from here. Please try again later."},
                                    status_code=429)
            try:
                raw = await read_limited(request, max_bytes)
                try:
                    data = json.loads(raw or b"null")
                except ValueError:
                    data = None
                if not isinstance(data, dict):
                    raise PayloadError("Expected a JSON object.")
                elapsed = data.get("elapsed_ms")
                if data.get("website") or not isinstance(elapsed, (int, float)) or elapsed < MIN_FILL_MS:
                    # The hidden field was filled in, or the form was sent faster than a person can type: a bot.
                    # It is told it worked, and nothing is kept.
                    return JSONResponse({"ok": True}, status_code=202)
                email = normalize_email(str(data.get("email") or ""))
                payload = parse(data)
            except PayloadError as exc:
                return JSONResponse({"error": str(exc), "kind": "invalid"}, status_code=exc.status_code)
            except StoreError:
                return JSONResponse({"error": "Enter a valid email address.", "kind": "invalid"}, status_code=400)
            public_limiter.hit(**{kind: address})
            language = data.get("language") if data.get("language") in LANGUAGES else "en"
            over_cap = store.mailed_since(time.time() - 24 * 3600) >= DAILY_MAIL_CAP
            if kind == "signup":
                inbound = store.add_request(kind, email, payload, name=payload["contact_name"],
                                            firm=payload["firm_name"], language=language, client_ip=address)
            else:
                inbound = store.add_request(kind, email, {"message": payload["message"]}, name=payload["name"],
                                            firm=payload["firm"], subject=payload["subject"], language=language,
                                            client_ip=address)
            if over_cap:
                logger.warning("Daily email cap reached: %s request %s is stored, not emailed.", kind, inbound.id)
                store.set_mail_result(inbound.id, "off", "Daily email limit reached.")
            else:
                mailer.submit(inbound)
            return JSONResponse({"ok": True}, status_code=202)
        return handler

    # ---------- the single-page app ----------

    index = settings.static_dir / "index.html"

    async def spa(request: Request) -> Response:
        if not index.exists():
            return Response("The web app has not been built (web_deploy/frontend: npm run build).",
                            status_code=503, media_type="text/plain")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        locked = store.firms_without_password()
        if locked:
            logger.warning("Nobody can sign in to these firms until they have a firm password "
                           "(admin_cli set-firm-password): %s", ", ".join(locked))
        if not store.has_console_password():
            logger.warning("The owner's console has no firm password yet (admin_cli set-firm-password admin).")
        if not mailer.enabled:
            logger.warning("Email is off (no SMTP_HOST/MAIL_FROM/SIGNUP_TO): requests are only stored.")
        task = asyncio.create_task(idle.run_forever()) if run_idle_monitor else None
        try:
            yield
        finally:
            if task:
                task.cancel()
            await box.wait()
            await asyncio.to_thread(mailer.shutdown)

    g = guarded
    assets = settings.static_dir / "assets"
    routes = [
        Route("/auth/me", me, methods=["GET"]),
        Route("/auth/firm", firm_signin, methods=["POST"]),
        Route("/auth/firm/forget", forget_firm, methods=["POST"]),
        Route("/auth/login", login, methods=["POST"]),
        Route("/auth/logout", logout, methods=["POST"]),
        Route("/auth/password", g(change_password), methods=["POST"]),
        Route("/public/signup", public_form("signup", SIGNUP_MAX_BYTES, signup_payload), methods=["POST"]),
        Route("/public/support", public_form("support", SUPPORT_MAX_BYTES, support_payload), methods=["POST"]),
        Route("/box/status", g(box_status), methods=["GET"]),
        Route("/box/start", g(box_start), methods=["POST"]),
        Route("/api/{path:path}", g(api), methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/firm/api/users", g(firm_users, firm_admin=True), methods=["GET", "POST"]),
        Route("/firm/api/users/{user_id}/{action}", g(firm_user_action, firm_admin=True), methods=["POST"]),
        Route("/firm/api/password", g(firm_password, firm_admin=True), methods=["POST"]),
        Route("/admin/api/firms", g(admin_firms, owner=True), methods=["GET", "POST"]),
        Route("/admin/api/firms/{slug}/users", g(admin_users, owner=True), methods=["GET", "POST"]),
        Route("/admin/api/firms/{slug}/{action}", g(admin_firm_action, owner=True), methods=["POST"]),
        Route("/admin/api/users/{user_id}/{action}", g(admin_user, owner=True), methods=["POST"]),
        Route("/admin/api/console-password", g(admin_console_password, owner=True), methods=["POST"]),
        Route("/admin/api/requests", g(admin_requests, owner=True), methods=["GET"]),
        Route("/admin/api/requests/{request_id}/{action}", g(admin_request_action, owner=True), methods=["POST"]),
        Route("/admin/api/box", g(admin_box, owner=True), methods=["GET"]),
        Route("/admin/api/box/{action}", g(admin_box, owner=True), methods=["POST"]),
    ]
    if assets.is_dir():
        routes.append(Mount("/assets", StaticFiles(directory=str(assets)), name="assets"))
    routes.append(Route("/{path:path}", spa, methods=["GET"]))
    return Starlette(routes=routes, lifespan=lifespan)


def create_production_app() -> Starlette:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = GatewaySettings.from_env()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(settings.database_path)
    http = httpx.AsyncClient(verify=box_ssl_context(settings), timeout=30.0)
    box = BoxController(settings, build_vast(settings), http)
    idle = IdleMonitor(box, settings.idle_minutes * 60)
    # Dev hardware has no rented box to stop
    return create_app(settings, store, box, idle, http, run_idle_monitor=settings.box_mode == "vast")
