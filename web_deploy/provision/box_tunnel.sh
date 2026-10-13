#!/bin/bash
# supervisord's tunnel: the box's only link to the data, an outbound SSH session to the Hetzner VPS.
#
#   -L 127.0.0.1:15432 -> the VPS's Postgres (every firm database and the ingestion queue)
#   -L 127.0.0.1:18000 -> the VPS's gateway (this start's database password, asking for file transfers)
#   -R 127.0.0.1:18095 on the VPS -> this box's API, which the gateway calls there (BOX_LINK=tunnel)
# File transfers go through a connection of their own (box_tunnel_files.sh), so they never delay chat.
#
# No privileges needed, unlike WireGuard (a Vast container has no /dev/net/tun). The key may do nothing else on the
# VPS (authorized_keys: permitopen/permitlisten for exactly these ports, no shell), and is accepted only while the
# gateway has this box started (izvoria-box-keys). The VPS's host key is pinned in known_hosts by provision.sh.
set -uo pipefail
set -a; . /workspace/web/api.env; set +a

ssh -N -T \
    -i "${TUNNEL_SSH_KEY:-/workspace/web/tunnel_ed25519}" -o IdentitiesOnly=yes \
    -p "${VPS_SSH_PORT:-22}" \
    -o UserKnownHostsFile="${STAGING_KNOWN_HOSTS:-/workspace/web/known_hosts}" -o StrictHostKeyChecking=yes \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${POSTGRES_PORT:-15432}:127.0.0.1:5432" \
    -L "127.0.0.1:${BOX_GATEWAY_PORT:-18000}:127.0.0.1:8000" \
    -R "127.0.0.1:${BOX_TUNNEL_PORT:-18095}:127.0.0.1:8095" \
    "izvoria-box@${VPS_HOST:?set VPS_HOST in /workspace/web/api.env}"
echo "tunnel closed (exit $?); retrying in 5 s"
sleep 5
exit 1
