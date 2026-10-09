#!/bin/bash
# Vast "on-start script": runs on every start, including each one the gateway triggers.
set -euo pipefail
if pgrep -x supervisord >/dev/null; then
    supervisorctl reread && supervisorctl update
else
    exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
fi
