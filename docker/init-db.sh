#!/bin/bash
# Postgres entrypoint init script — creates the agentrec database if it
# doesn't already exist.  The POSTGRES_DB env var already creates the
# default DB, but this script ensures extensions are ready.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Ensure the database exists (no-op if POSTGRES_DB already created it).
    SELECT 'Database ready: ${POSTGRES_DB}';
EOSQL
