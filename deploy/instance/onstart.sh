#!/bin/bash
# Vast "on-start script" for the instance: runs on every start, including each one the gateway triggers.
# Vast containers have no systemd, so supervisord is the init for postgres, the X desktop, noVNC and the agent.
set -euo pipefail

# A stale X lock from the previous run (the container was stopped, not shut down) would stop Xvnc starting.
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
install -d -m 700 /var/run/onyx-agent

if pgrep -x supervisord >/dev/null; then
    supervisorctl reread && supervisorctl update
else
    exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
fi
