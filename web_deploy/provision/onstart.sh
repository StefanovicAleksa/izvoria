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
    install -d /workspace/logs
    # The current version of the app first (web_deploy/gateway/releases.py): a box that missed a deploy updates itself
    # here, before any service runs its old code. rsync replaces files by renaming, so this script, already read
    # by bash, is not disturbed if the update changes it.
    PYTHON=/workspace/onyx_v3/venv/bin/python
    [ -x "${PYTHON}" ] || PYTHON=python3
    status=0
    (cd /workspace/onyx_v3 && "${PYTHON}" web_deploy/provision/self_update.py) >> /workspace/logs/self_update.log 2>&1 \
        || status=$?
    if [ "${status}" = 10 ]; then
        echo "Started only to be updated: no service runs; the gateway stops this box." >> /workspace/logs/self_update.log
        exit 0
    elif [ "${status}" != 0 ]; then
        echo "The update failed half-way: no service runs on half-updated code." >> /workspace/logs/self_update.log
        exit 1
    fi
    exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
fi
