#!/bin/bash
# The appliance database. One database holds every language (see CLAUDE.md "Languages").
# Idempotent. Needs postgres running and POSTGRES_USER / POSTGRES_PASSWORD in the environment (secrets.env).
set -euo pipefail

: "${POSTGRES_USER:?}" "${POSTGRES_PASSWORD:?}"

psql_admin() { su postgres -c "psql -v ON_ERROR_STOP=1 -tAq $*"; }

if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'\"")" ]; then
    psql_admin "-c \"CREATE ROLE ${POSTGRES_USER} LOGIN PASSWORD '${POSTGRES_PASSWORD}'\""
fi

for db in onyx; do
    if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_database WHERE datname='${db}'\"")" ]; then
        psql_admin "-c \"CREATE DATABASE ${db} OWNER ${POSTGRES_USER}\""
    fi
    # The same statement scripts/init.sql runs for the Docker databases. Schema comes from Alembic at engine start.
    psql_admin "-d ${db} -c \"CREATE EXTENSION IF NOT EXISTS vector\""
done
echo "Database onyx is ready."
