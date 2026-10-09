"""
The owner's command line on the gateway VPS: the first owner account, and firms and users without the browser.

  python -m web_deploy.gateway.admin_cli create-owner you@example.com          # also the console's firm password
  python -m web_deploy.gateway.admin_cli create-firm legal-demo "Legal demo" --language en --demo
  python -m web_deploy.gateway.admin_cli set-firm-password legal-demo [--generate]   # "admin": the console's
  python -m web_deploy.gateway.admin_cli create-user legal-demo alice@firm.com --role admin
  python -m web_deploy.gateway.admin_cli reset-password legal-demo alice@firm.com
  python -m web_deploy.gateway.admin_cli disable-user legal-demo alice@firm.com
  python -m web_deploy.gateway.admin_cli list
  python -m web_deploy.gateway.admin_cli list-requests [--kind signup|support] [--status new|handled|spam]
  python -m web_deploy.gateway.admin_cli mark-request <id> handled

Signing in is two steps: the firm (its sign-in name and the password its people share), then a person's own email
and password. The owner signs in as the firm "admin".

Passwords are prompted for, never taken on the command line (they would land in shell history). Set
ONYX_ADMIN_PASSWORD to script it (and ONYX_FIRM_PASSWORD for a firm's shared password).
"""

import argparse
import getpass
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from web_deploy.common.tenancy import InvalidSlug
from web_deploy.gateway.passwords import WeakPassword, check_strength
from web_deploy.gateway.store import (
    OWNER_FIRM, REQUEST_KINDS, REQUEST_STATUSES, Store, StoreError, normalize_email,
)


def _password(prompt: str, variable: str = "ONYX_ADMIN_PASSWORD") -> str:
    scripted = os.environ.get(variable) or os.environ.get("ONYX_ADMIN_PASSWORD")
    if scripted:
        return scripted
    first = getpass.getpass(prompt)
    if first != getpass.getpass("Again: "):
        raise StoreError("The passwords don't match.")
    return first


def _find_user(store: Store, firm: str, email: str):
    email = normalize_email(email)
    for user in store.users(firm):
        if user.email == email:
            return user
    raise StoreError(f"No user {email} at {firm}.")


def _firm_password(slug: str, generate: bool) -> str:
    if generate:
        password = secrets.token_urlsafe(15)
        print(f"Firm password for {slug} (shown once, give it to the firm): {password}")
        return password
    return _password(f"Firm password for {slug} (shared by everyone at the firm): ", "ONYX_FIRM_PASSWORD")


def main(argv: Optional[list] = None, store: Optional[Store] = None) -> int:
    parser = argparse.ArgumentParser(prog="admin_cli", description="Manage firms and users of the hosted Onyx.")
    parser.add_argument("--db", default=os.environ.get("GATEWAY_DB", "/var/lib/onyx-web/gateway.sqlite3"))
    sub = parser.add_subparsers(dest="command", required=True)

    owner = sub.add_parser("create-owner")
    owner.add_argument("email")

    firm = sub.add_parser("create-firm")
    firm.add_argument("slug")
    firm.add_argument("name")
    firm.add_argument("--language", default="en", choices=["en", "sr"])
    firm.add_argument("--demo", action="store_true")
    firm.add_argument("--generate", action="store_true", help="make up the firm password and print it once")
    firm.add_argument("--no-password", action="store_true", help="nobody can sign in until set-firm-password")

    firm_password = sub.add_parser("set-firm-password")
    firm_password.add_argument("slug", help='a firm, or "admin" for the owner\'s console')
    firm_password.add_argument("--generate", action="store_true")

    for name in ("enable-firm", "disable-firm"):
        sub.add_parser(name).add_argument("slug")

    user = sub.add_parser("create-user")
    user.add_argument("firm")
    user.add_argument("email")
    user.add_argument("--role", default="member", choices=["member", "admin"])
    user.add_argument("--name", default="")

    for name in ("reset-password", "disable-user", "enable-user"):
        p = sub.add_parser(name)
        p.add_argument("firm")
        p.add_argument("email")

    sub.add_parser("list")

    requests = sub.add_parser("list-requests")
    requests.add_argument("--kind", choices=REQUEST_KINDS)
    requests.add_argument("--status", choices=REQUEST_STATUSES)

    mark = sub.add_parser("mark-request")
    mark.add_argument("request_id")
    mark.add_argument("status", choices=REQUEST_STATUSES)
    args = parser.parse_args(argv)

    if store is None:
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        store = Store(Path(args.db))
    try:
        if args.command == "create-owner":
            # Every password is asked for before anything is written, so a mistyped one leaves nothing half made
            password = _password("Owner password: ")
            console = None if store.has_console_password() else _password(
                "The console's firm password (the first sign-in step, as firm 'admin'): ", "ONYX_FIRM_PASSWORD")
            check_strength(password)
            if console is not None:
                check_strength(console)
            created = store.create_user(OWNER_FIRM, args.email, password, role="owner")
            if console is not None:
                store.set_firm_password(OWNER_FIRM, console)
            print(f"Owner {created.email} created. Sign in as firm 'admin', then with this email.")
        elif args.command == "create-firm":
            password = None if args.no_password else _firm_password(args.slug, args.generate)
            created = store.create_firm(args.slug, args.name, args.language, args.demo, password)
            print(f"Firm '{created.slug}' created. Its database is made on the GPU box at its first sign-in.")
        elif args.command == "set-firm-password":
            if args.slug != OWNER_FIRM and store.firm(args.slug) is None:
                raise StoreError(f"No firm called '{args.slug}'.")
            store.set_firm_password(args.slug, _firm_password(args.slug, args.generate))
            print(f"{args.slug}: firm password set; every device signs in as the firm again.")
        elif args.command in ("enable-firm", "disable-firm"):
            store.set_firm_disabled(args.slug, args.command == "disable-firm")
            print(f"{args.slug}: {'disabled' if args.command == 'disable-firm' else 'enabled'}.")
        elif args.command == "create-user":
            created = store.create_user(args.firm, args.email, _password(f"Password for {args.email}: "),
                                        args.role, args.name)
            print(f"{created.email} ({created.role}) can now sign in at {args.firm}.")
        elif args.command == "reset-password":
            target = _find_user(store, args.firm, args.email)
            store.set_password(target.id, _password(f"New password for {target.email}: "))
            print(f"Password reset; {target.email} is signed out everywhere.")
        elif args.command in ("disable-user", "enable-user"):
            target = _find_user(store, args.firm, args.email)
            store.set_user_disabled(target.id, args.command == "disable-user")
            print(f"{target.email}: {'disabled' if args.command == 'disable-user' else 'enabled'}.")
        elif args.command == "list":
            for f in store.firms():
                flags = " (demo)" if f.is_demo else ""
                flags += " [disabled]" if f.disabled else ""
                flags += "" if f.has_password else " [no firm password]"
                print(f"{f.slug}{flags}: {f.name}, answers in {f.default_language}")
                for u in store.users(f.slug):
                    print(f"    {u.email} {u.role}{' [disabled]' if u.disabled else ''}")
            owners = store.users(OWNER_FIRM)
            print(f"owners: {', '.join(u.email for u in owners) or 'none'}"
                  + ("" if store.has_console_password() else " [no console password]"))
        elif args.command == "list-requests":
            for r in store.requests(args.kind, args.status):
                when = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M")
                about = r.firm if r.kind == "signup" else r.subject
                print(f"{r.id} {when} {r.kind} [{r.status}, mail {r.mail_status}] {r.email}: {about}")
        elif args.command == "mark-request":
            store.set_request_status(args.request_id, args.status)
            print(f"{args.request_id}: {args.status}.")
    except (StoreError, WeakPassword, InvalidSlug) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
