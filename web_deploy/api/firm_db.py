"""
A firm's own database, created on first use: by the API at a firm's first request, and by an ingestion job that
reaches a firm before anyone has signed in to it (a seeded demo firm). Idempotent and safe to race: a concurrent
CREATE DATABASE fails with "already exists", which is what we wanted.
"""

import logging

logger = logging.getLogger(__name__)


def ensure_firm_database(name: str, host: str, port: int, user: str, password: str,
                         admin_database: str = "postgres") -> None:
    import psycopg2
    from psycopg2 import errors, sql

    from app.core.database.data.credentials import current_password
    # On the GPU box the password comes from the gateway (web_deploy/common/box_credentials.py), not the environment
    params = dict(host=host, port=port, user=user, password=current_password() or password)
    admin = psycopg2.connect(dbname=admin_database, **params)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cursor.fetchone() is None:
                try:
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                    logger.info(f"Created database {name}.")
                except errors.DuplicateDatabase:
                    pass
    finally:
        admin.close()
    firm_db = psycopg2.connect(dbname=name, **params)
    firm_db.autocommit = True
    try:
        with firm_db.cursor() as cursor:
            # Already there when the cluster's template1 has it (provision.sh), which a non-superuser needs
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        firm_db.close()
