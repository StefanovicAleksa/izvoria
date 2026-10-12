#!/bin/bash
# Vast "on-start script": runs on every start, including each one the gateway triggers.
set -euo pipefail
if pgrep -x supervisord >/dev/null; then
    supervisorctl reread && supervisorctl update
else
    # A fresh start: nothing a pipeline left in its temporary folder outlives the stop. (Job scratch under
    # /workspace/tenants is pushed to the VPS if a stop interrupted it, then wiped, by the ingestion worker as it
    # starts: staging.py, `recover`.)
    rm -rf /workspace/tmp && install -d -m 700 /workspace/tmp
    exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
fi
