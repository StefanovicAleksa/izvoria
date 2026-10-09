"""
Firms, their users, sign-in state and the requests that arrive from the public pages, in SQLite on the gateway VPS.

The gateway is always on and the GPU box is not, so who may sign in is decided here. The box only ever learns a
firm slug and an opaque user id, in a signed header.

Signing in takes two steps. A firm signs in with its sign-in name (the slug) and a password the whole firm shares,
which gives the browser a short-lived "firm pass"; a person then signs in with their own email and password within
that firm. The owner's console is the pseudo-firm "admin", whose shared password is kept in `meta`. There is no
self sign-up: a firm asks for access on the public request page, and the owner creates it (admin_cli.py, or the
console). A firm's administrators then manage its own accounts.

Sessions and firm passes are server-side rows keyed by a hash of the cookie's token, so disabling a user or
resetting a password signs them out everywhere at once, and changing a firm's password closes every device's pass.
"""

import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from web_deploy.common.tenancy import validate_slug
from web_deploy.gateway.passwords import DUMMY_HASH, hash_password, verify_password

SESSION_SECONDS = 14 * 24 * 3600
OWNER_FIRM = "admin"  # the owner's own accounts live under the admin console's "firm"
USER_ROLES = ("member", "admin")
REQUEST_KINDS = ("signup", "support")
REQUEST_STATUSES = ("new", "handled", "spam")
MAIL_STATUSES = ("pending", "sent", "failed", "off")
_CONSOLE_PASSWORD = "console_password_hash"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS firms (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    default_language TEXT NOT NULL DEFAULT 'en',
    is_demo INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    firm_slug TEXT NOT NULL,
    email TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE (firm_slug, email)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS firm_passes (
    token_hash TEXT PRIMARY KEY,
    firm_slug TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_firm_passes_firm ON firm_passes(firm_slug);
CREATE TABLE IF NOT EXISTS inbound_requests (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    email TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    firm TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    client_ip TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'new',
    mail_status TEXT NOT NULL DEFAULT 'pending',
    mail_error TEXT NOT NULL DEFAULT '',
    mailed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_inbound_created ON inbound_requests(created_at);
"""

# Columns added after the first release, for databases made before them
_FIRM_COLUMNS = {"password_hash": "TEXT", "password_changed_at": "REAL"}


class StoreError(ValueError):
    pass


@dataclass(frozen=True)
class Firm:
    slug: str
    name: str
    default_language: str
    is_demo: bool
    disabled: bool
    has_password: bool = False

    def public(self) -> dict:
        return {"slug": self.slug, "name": self.name, "default_language": self.default_language,
                "is_demo": self.is_demo, "disabled": self.disabled, "has_password": self.has_password}


@dataclass(frozen=True)
class User:
    id: str
    firm_slug: str
    email: str
    name: str
    role: str
    disabled: bool

    def public(self) -> dict:
        return {"id": self.id, "firm": self.firm_slug, "email": self.email, "name": self.name,
                "role": self.role, "disabled": self.disabled}


@dataclass(frozen=True)
class InboundRequest:
    """A sign-up request or a help request from the public pages."""
    id: str
    kind: str
    created_at: float
    email: str
    name: str
    firm: str
    subject: str
    payload: Dict
    language: str
    client_ip: str
    status: str
    mail_status: str
    mail_error: str

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "created_at": self.created_at, "email": self.email,
                "name": self.name, "firm": self.firm, "subject": self.subject, "payload": self.payload,
                "language": self.language, "status": self.status, "mail_status": self.mail_status,
                "mail_error": self.mail_error}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if "@" not in email or len(email) > 254 or " " in email:
        raise StoreError(f"'{email}' is not an email address.")
    return email


class Store:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Brings a database from an earlier release up to this one. Safe to run on every start."""
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(firms)").fetchall()}
        for column, kind in _FIRM_COLUMNS.items():
            if column not in existing:
                self._db.execute(f"ALTER TABLE firms ADD COLUMN {column} {kind}")

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._db.execute(sql, params).rowcount

    # ---------- firms ----------

    def create_firm(self, slug: str, name: str, default_language: str = "en", is_demo: bool = False,
                    password: Optional[str] = None) -> Firm:
        """A firm. Without a password nobody can sign in to it until set_firm_password gives it one."""
        slug = validate_slug(slug)
        if not (name or "").strip():
            raise StoreError("A firm needs a name.")
        password_hash = hash_password(password) if password is not None else None
        now = self._clock()
        try:
            self._execute("INSERT INTO firms (slug, name, default_language, is_demo, created_at, password_hash, "
                          "password_changed_at) VALUES (?,?,?,?,?,?,?)",
                          (slug, name.strip()[:200], default_language, int(is_demo), now, password_hash,
                           now if password_hash else None))
        except sqlite3.IntegrityError:
            raise StoreError(f"A firm called '{slug}' already exists.")
        return self.firm(slug)

    def firm(self, slug: str) -> Optional[Firm]:
        rows = self._query("SELECT * FROM firms WHERE slug = ?", (slug,))
        return _firm(rows[0]) if rows else None

    def firms(self) -> List[Firm]:
        return [_firm(r) for r in self._query("SELECT * FROM firms ORDER BY slug")]

    def set_firm_disabled(self, slug: str, disabled: bool) -> None:
        if not self._execute("UPDATE firms SET disabled = ? WHERE slug = ?", (int(disabled), slug)):
            raise StoreError(f"No firm called '{slug}'.")
        if disabled:
            self._execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE firm_slug = ?)", (slug,))
            self._execute("DELETE FROM firm_passes WHERE firm_slug = ?", (slug,))

    def set_firm_password(self, slug: str, password: str) -> None:
        """
        The password a firm shares for its first sign-in step ("admin": the owner's console). Every device's firm
        pass ends, so each must pass the first step again; people already signed in stay signed in.
        """
        password_hash = hash_password(password)
        if slug == OWNER_FIRM:
            self._execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (_CONSOLE_PASSWORD, password_hash))
        elif not self._execute("UPDATE firms SET password_hash = ?, password_changed_at = ? WHERE slug = ?",
                               (password_hash, self._clock(), slug)):
            raise StoreError(f"No firm called '{slug}'.")
        self._execute("DELETE FROM firm_passes WHERE firm_slug = ?", (slug,))

    def has_console_password(self) -> bool:
        return bool(self._query("SELECT 1 FROM meta WHERE key = ?", (_CONSOLE_PASSWORD,)))

    def verify_firm(self, slug: str, password: str) -> Optional[str]:
        """
        The firm's slug, if it is enabled and the password is its shared one. Always exactly one password check,
        so an unknown firm, a firm without a password and a wrong password take the same time.
        """
        slug = (slug or "").strip().lower()
        stored = None
        if slug == OWNER_FIRM:
            rows = self._query("SELECT value FROM meta WHERE key = ?", (_CONSOLE_PASSWORD,))
            stored = rows[0]["value"] if rows else None
        elif slug:
            rows = self._query("SELECT password_hash, disabled FROM firms WHERE slug = ?", (slug,))
            if rows and not rows[0]["disabled"]:
                stored = rows[0]["password_hash"]
        if stored is None:
            verify_password(password, DUMMY_HASH)
            return None
        return slug if verify_password(password, stored) else None

    def firms_without_password(self) -> List[str]:
        return [r["slug"] for r in self._query("SELECT slug FROM firms WHERE password_hash IS NULL ORDER BY slug")]

    # ---------- firm passes (the first sign-in step) ----------

    def create_firm_pass(self, slug: str, seconds: float) -> str:
        token = secrets.token_urlsafe(32)
        now = self._clock()
        self._execute("INSERT INTO firm_passes (token_hash, firm_slug, expires_at, created_at) VALUES (?,?,?,?)",
                      (_token_hash(token), slug, now + seconds, now))
        return token

    def firm_pass(self, token: Optional[str]) -> Optional[str]:
        """The firm a pass is for, while it is unexpired and the firm can still sign in."""
        if not token:
            return None
        rows = self._query("SELECT firm_slug FROM firm_passes WHERE token_hash = ? AND expires_at > ?",
                           (_token_hash(token), self._clock()))
        if not rows:
            return None
        slug = rows[0]["firm_slug"]
        if slug != OWNER_FIRM:
            firm = self.firm(slug)
            if firm is None or firm.disabled:
                return None
        return slug

    def end_firm_pass(self, token: Optional[str]) -> None:
        if token:
            self._execute("DELETE FROM firm_passes WHERE token_hash = ?", (_token_hash(token),))

    # ---------- users ----------

    def create_user(self, firm_slug: str, email: str, password: str, role: str = "member", name: str = "") -> User:
        if firm_slug == OWNER_FIRM:
            if role != "owner":
                raise StoreError("Accounts on the admin console are owners.")
        else:
            if self.firm(firm_slug) is None:
                raise StoreError(f"No firm called '{firm_slug}'.")
            if role not in USER_ROLES:
                raise StoreError(f"A firm user's role is one of: {', '.join(USER_ROLES)}.")
        user_id = uuid.uuid4().hex
        try:
            self._execute(
                "INSERT INTO users (id, firm_slug, email, name, password_hash, role, created_at) VALUES (?,?,?,?,?,?,?)",
                (user_id, firm_slug, normalize_email(email), (name or "").strip()[:200], hash_password(password),
                 role, self._clock()))
        except sqlite3.IntegrityError:
            raise StoreError(f"{email} already has an account at {firm_slug}.")
        return self.user(user_id)

    def user(self, user_id: str) -> Optional[User]:
        rows = self._query("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(rows[0]) if rows else None

    def users(self, firm_slug: str) -> List[User]:
        return [_user(r) for r in self._query("SELECT * FROM users WHERE firm_slug = ? ORDER BY email", (firm_slug,))]

    def firm_user(self, firm_slug: str, user_id: str) -> Optional[User]:
        """The user, only if they belong to this firm: what a firm administrator may act on."""
        rows = self._query("SELECT * FROM users WHERE id = ? AND firm_slug = ?", (user_id, firm_slug))
        return _user(rows[0]) if rows else None

    def enabled_admin_count(self, firm_slug: str) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM users WHERE firm_slug = ? AND role = 'admin' AND disabled = 0",
                           (firm_slug,))
        return int(rows[0]["n"])

    def authenticate(self, firm_slug: str, email: str, password: str) -> Optional[User]:
        """The user, if the email and password match an enabled account of an enabled firm. Constant-ish time."""
        try:
            email = normalize_email(email)
        except StoreError:
            verify_password(password, DUMMY_HASH)
            return None
        rows = self._query("SELECT * FROM users WHERE firm_slug = ? AND email = ?", (firm_slug, email))
        if not rows:
            verify_password(password, DUMMY_HASH)
            return None
        row = rows[0]
        if not verify_password(password, row["password_hash"]) or row["disabled"]:
            return None
        if firm_slug != OWNER_FIRM:
            firm = self.firm(firm_slug)
            if firm is None or firm.disabled:
                return None
        return _user(row)

    def set_password(self, user_id: str, password: str) -> None:
        if not self._execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), user_id)):
            raise StoreError("No such user.")
        self._execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def set_user_disabled(self, user_id: str, disabled: bool) -> None:
        if not self._execute("UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id)):
            raise StoreError("No such user.")
        if disabled:
            self._execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def set_user_role(self, user_id: str, role: str) -> None:
        if role not in USER_ROLES:
            raise StoreError(f"A firm user's role is one of: {', '.join(USER_ROLES)}.")
        if not self._execute("UPDATE users SET role = ? WHERE id = ? AND firm_slug != ?", (role, user_id, OWNER_FIRM)):
            raise StoreError("No such user.")

    # ---------- sessions ----------

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = self._clock()
        self._execute("INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?,?,?,?)",
                      (_token_hash(token), user_id, now + SESSION_SECONDS, now))
        return token

    def session_user(self, token: Optional[str]) -> Optional[User]:
        if not token:
            return None
        rows = self._query(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.disabled = 0",
            (_token_hash(token), self._clock()))
        return _user(rows[0]) if rows else None

    def end_session(self, token: Optional[str]) -> None:
        if token:
            self._execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    def purge_sessions(self) -> int:
        now = self._clock()
        return (self._execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
                + self._execute("DELETE FROM firm_passes WHERE expires_at <= ?", (now,)))

    # ---------- requests from the public pages ----------

    def add_request(self, kind: str, email: str, payload: Dict, name: str = "", firm: str = "", subject: str = "",
                    language: str = "en", client_ip: str = "", mail_status: str = "pending") -> InboundRequest:
        if kind not in REQUEST_KINDS:
            raise StoreError(f"A request is one of: {', '.join(REQUEST_KINDS)}.")
        if mail_status not in MAIL_STATUSES:
            raise StoreError("Unknown mail status.")
        request_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO inbound_requests (id, kind, created_at, email, name, firm, subject, payload_json, language, "
            "client_ip, mail_status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (request_id, kind, self._clock(), normalize_email(email), name[:200], firm[:200], subject[:200],
             json.dumps(payload, ensure_ascii=False), language, client_ip[:64], mail_status))
        return self.request(request_id)

    def request(self, request_id: str) -> Optional[InboundRequest]:
        rows = self._query("SELECT * FROM inbound_requests WHERE id = ?", (request_id,))
        return _request(rows[0]) if rows else None

    def requests(self, kind: Optional[str] = None, status: Optional[str] = None,
                 limit: int = 200) -> List[InboundRequest]:
        sql, params = "SELECT * FROM inbound_requests WHERE 1 = 1", []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [_request(r) for r in self._query(sql, tuple(params))]

    def set_request_status(self, request_id: str, status: str) -> None:
        if status not in REQUEST_STATUSES:
            raise StoreError(f"A request's status is one of: {', '.join(REQUEST_STATUSES)}.")
        if not self._execute("UPDATE inbound_requests SET status = ? WHERE id = ?", (status, request_id)):
            raise StoreError("No such request.")

    def set_mail_result(self, request_id: str, mail_status: str, error: str = "") -> None:
        if mail_status not in MAIL_STATUSES:
            raise StoreError("Unknown mail status.")
        self._execute("UPDATE inbound_requests SET mail_status = ?, mail_error = ?, mailed_at = ? WHERE id = ?",
                      (mail_status, error[:500], self._clock() if mail_status == "sent" else None, request_id))

    def mailed_since(self, since: float) -> int:
        """How many requests were emailed (or are about to be) since a moment: the daily cap on outgoing mail."""
        rows = self._query("SELECT COUNT(*) AS n FROM inbound_requests WHERE created_at >= ? "
                           "AND mail_status IN ('pending', 'sent', 'failed')", (since,))
        return int(rows[0]["n"])


def _firm(row: sqlite3.Row) -> Firm:
    return Firm(slug=row["slug"], name=row["name"], default_language=row["default_language"],
                is_demo=bool(row["is_demo"]), disabled=bool(row["disabled"]),
                has_password=row["password_hash"] is not None)


def _user(row: sqlite3.Row) -> User:
    return User(id=row["id"], firm_slug=row["firm_slug"], email=row["email"], name=row["name"], role=row["role"],
                disabled=bool(row["disabled"]))


def _request(row: sqlite3.Row) -> InboundRequest:
    return InboundRequest(id=row["id"], kind=row["kind"], created_at=row["created_at"], email=row["email"],
                          name=row["name"], firm=row["firm"], subject=row["subject"],
                          payload=json.loads(row["payload_json"]), language=row["language"],
                          client_ip=row["client_ip"], status=row["status"], mail_status=row["mail_status"],
                          mail_error=row["mail_error"])
