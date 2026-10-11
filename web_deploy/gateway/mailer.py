"""
Email for the two public pages: a firm asking for Izvoria, and a help request. Plain SMTP from the standard library.

Mail never goes out on a request thread. The request is stored first (the owner's console lists every one, so a
lost email loses nothing), then a single background worker sends it and records whether that worked. Without
SMTP_HOST mail is off and requests are only stored.
"""

import logging
import smtplib
import ssl
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Callable, Dict, List, Optional, Tuple

from web_deploy.gateway.store import InboundRequest, Store

logger = logging.getLogger(__name__)

SUBJECT_MAX = 150


@dataclass(frozen=True)
class MailSettings:
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_ssl: bool = False  # implicit TLS (port 465) instead of STARTTLS
    mail_from: str = ""
    signup_to: str = ""
    support_to: str = ""

    @property
    def enabled(self) -> bool:
        # A user without a password is a half-filled form (an app password still to be made): off, not failing
        return bool(self.smtp_host and self.mail_from and self.signup_to
                    and (not self.smtp_user or self.smtp_password))

    def recipient(self, kind: str) -> str:
        return self.signup_to if kind == "signup" else (self.support_to or self.signup_to)


def smtp_send(settings: MailSettings, message: EmailMessage) -> None:
    context = ssl.create_default_context()
    if settings.smtp_ssl:
        client = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=20, context=context)
    else:
        client = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20)
    with client as smtp:
        if settings.smtp_starttls and not settings.smtp_ssl:
            smtp.starttls(context=context)
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(message)


def _header(value: str) -> str:
    """One line, no control characters: nothing a requester types can add a header."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in (value or "")).split())


# The sign-up form's profile, in the order of docs/glossary_authoring/firm_profile_template.md, so the email can be
# pasted straight into the glossary creator prompt as {{FIRM_PROFILE}}.
PROFILE_SECTIONS: List[Tuple[str, str]] = [
    ("industry", "2. Industry"),
    ("sub_domains", "3. Sub-industries / specialised domains"),
    ("description", "4. Firm description"),
    ("data_kinds", "5. What the data will be"),
]
PROPER_NOUNS: List[Tuple[str, str]] = [
    ("people", "People"),
    ("clients", "Clients / counterparties / institutions"),
    ("products", "Products, funds, drugs, devices, software the firm uses"),
    ("places", "Places"),
    ("acronyms", "Internal acronyms and in-house jargon"),
    ("laws", "Laws, regulations, standards, case names cited often"),
]
LANGUAGE_ITEMS: List[Tuple[str, str]] = [
    ("share", "Languages and share"),
    ("english_terms", "Do Serbian speakers use English terms as-is?"),
    ("latin", "Latin in speech?"),
]


def _block(text: str) -> str:
    return (text or "").strip() or "(not given)"


def signup_body(request: InboundRequest) -> str:
    p: Dict = request.payload
    profile: Dict = p.get("profile") or {}
    lines = [
        "A firm asked for Izvoria.",
        "",
        f"Firm: {p.get('firm_name', '')}",
        f"Contact: {request.name} <{request.email}>",
        f"Phone: {p.get('phone') or '-'}",
        f"Role: {p.get('contact_role') or '-'}",
        f"Page language: {request.language}",
        "",
        "--- Firm profile (paste into the glossary creator prompt as {{FIRM_PROFILE}}) ---",
        "",
        "## 1. Firm name",
        "",
        _block(p.get("firm_name", "")),
    ]
    for key, title in PROFILE_SECTIONS:
        lines += ["", f"## {title}", "", _block(profile.get(key, ""))]
    nouns: Dict = profile.get("proper_nouns") or {}
    lines += ["", "## 6. Proper nouns the recordings will contain", ""]
    if any((nouns.get(k) or "").strip() for k, _ in PROPER_NOUNS):
        lines += [f"- **{title}:** {(nouns.get(key) or '').strip()}" for key, title in PROPER_NOUNS]
    else:
        lines.append("(left out by the firm: add these names by hand at onboarding)")
    languages: Dict = profile.get("languages") or {}
    lines += ["", "## 7. Languages spoken in the recordings", ""]
    lines += [f"- **{title}:** {(languages.get(key) or '').strip()}" for key, title in LANGUAGE_ITEMS]
    lines += ["", "## 8. Terms you already know Whisper gets wrong", "", _block(profile.get("mishearings", ""))]
    lines += ["", "---", f"Request {request.id}. Reply to this email to answer the contact."]
    return "\n".join(lines) + "\n"


def support_body(request: InboundRequest) -> str:
    return "\n".join([
        "A help request from the Izvoria web app.",
        "",
        f"From: {request.name or '-'} <{request.email}>",
        f"Firm: {request.firm or '-'}",
        f"Subject: {request.subject}",
        f"Page language: {request.language}",
        "",
        (request.payload.get("message") or "").strip(),
        "",
        "---",
        f"Request {request.id}. Reply to this email to answer them.",
    ]) + "\n"


def build_message(settings: MailSettings, request: InboundRequest) -> EmailMessage:
    message = EmailMessage()
    if request.kind == "signup":
        subject = f"Izvoria access request: {request.payload.get('firm_name', '')}"
        body = signup_body(request)
    else:
        subject = f"Izvoria support: {request.subject}"
        body = support_body(request)
    message["Subject"] = _header(subject)[:SUBJECT_MAX]
    message["From"] = _header(settings.mail_from)
    message["To"] = _header(settings.recipient(request.kind))
    message["Reply-To"] = _header(request.email)
    message.set_content(body)
    return message


class Mailer:
    def __init__(self, settings: MailSettings, store: Store,
                 transport: Callable[[MailSettings, EmailMessage], None] = smtp_send,
                 executor: Optional[Executor] = None) -> None:
        self.settings = settings
        self._store = store
        self._transport = transport
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="mailer")

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def submit(self, request: InboundRequest) -> None:
        """Sends the request's email in the background; the outcome lands in the request's mail_status."""
        if not self.enabled:
            self._store.set_mail_result(request.id, "off")
            return
        self._executor.submit(self._send, request)

    def _send(self, request: InboundRequest) -> None:
        try:
            self._transport(self.settings, build_message(self.settings, request))
        except Exception as exc:  # noqa: BLE001 - any failure is recorded, never raised into a request
            logger.warning("Could not email %s request %s: %s", request.kind, request.id, exc)
            self._store.set_mail_result(request.id, "failed", f"{type(exc).__name__}: {exc}")
            return
        self._store.set_mail_result(request.id, "sent")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
