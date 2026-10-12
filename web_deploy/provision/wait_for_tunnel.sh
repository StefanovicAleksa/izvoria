#!/bin/bash
# Blocks until the tunnel's end of the VPS's Postgres answers here (box_tunnel.sh), so the API and the ingestion
# worker never start against a database they can't reach. Run with api.env loaded.
port="${POSTGRES_PORT:-15432}"
until (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; do
    sleep 2
done
