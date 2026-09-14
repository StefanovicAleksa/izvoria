-- Canvas: init_sql_v2
-- Path: scripts/init.sql

-- This script runs automatically on the first boot of the Postgres container
-- (and is re-applied by scripts/reset_dev_environment.sh after a schema reset).
CREATE EXTENSION IF NOT EXISTS vector;
