#!/bin/bash
# supervisord's tunnel-files: a second outbound SSH session to the VPS, used only for moving one ingestion job's files.
#
#   -R 127.0.0.1:18022 on the VPS -> this box's own sshd, which the VPS uses to send a job's files into the scratch
#      and take the results back (gateway/staging_server.py, with the VPS's staging key, which the ingest worker lets
#      in at start). The box itself has no key that reaches the VPS's files.
#
# A connection of its own, beside box_tunnel.sh's: a multi-gigabyte transfer then never queues behind, or ahead of,
# the database traffic and API calls chat runs on. Same key, same limits, same "only while started" rule.
set -uo pipefail
set -a; . /workspace/web/api.env; set +a

ssh -N -T \
    -i "${TUNNEL_SSH_KEY:-/workspace/web/tunnel_ed25519}" -o IdentitiesOnly=yes \
    -p "${VPS_SSH_PORT:-22}" \
    -o UserKnownHostsFile="${STAGING_KNOWN_HOSTS:-/workspace/web/known_hosts}" -o StrictHostKeyChecking=yes \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -R "127.0.0.1:${BOX_FILES_PORT:-18022}:127.0.0.1:22" \
    "izvoria-box@${VPS_HOST:?set VPS_HOST in /workspace/web/api.env}"
echo "files tunnel closed (exit $?); retrying in 5 s"
sleep 5
exit 1
