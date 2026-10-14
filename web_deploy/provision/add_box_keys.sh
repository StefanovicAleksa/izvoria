#!/bin/bash
# Run on the VPS, as root, with the public key the GPU box's provision.sh printed (installed as
# /usr/local/sbin/izvoria-add-box-keys by install_gateway.sh):
#   izvoria-add-box-keys 'ssh-ed25519 AAAA… izvoria-box tunnel'           adds this box beside the others (spares)
#   izvoria-add-box-keys --only 'ssh-ed25519 AAAA… izvoria-box tunnel'    this box alone: every other key is removed
# Every provisioned box (the primary and its spares, docs/web_deploy.md "Spare boxes") has its own key; the gateway
# runs at most one of them at a time. Use --only when retiring boxes.
#
# The key may only open the box's tunnels, nothing else (no shell, no files):
#   -L to this machine's Postgres (5432) and the gateway (8000: the box's database password, file staging)
#   -R back to the box's API (18095) and the box's own sshd (18022: the VPS moves one job's files through it)
# and it is offered to sshd only while the gateway has a box started (izvoria-box-keys).
set -euo pipefail
ONLY=0
if [ "${1:-}" = "--only" ]; then ONLY=1; shift; fi
TUNNEL="${1:?usage: izvoria-add-box-keys [--only] '<tunnel key>'}"
case "${TUNNEL}" in ssh-ed25519\ *) ;; *) echo "Not an ed25519 public key: ${TUNNEL}" >&2; exit 1 ;; esac
KEYS=/etc/onyx-web/box_authorized_keys
MATERIAL="$(echo "${TUNNEL}" | awk '{print $2}')"
TMP="$(mktemp)"
# Keep the other boxes' keys (unless --only); this box's key, if already there, is written afresh below
if [ "${ONLY}" = 0 ] && [ -f "${KEYS}" ]; then
    grep -v -F " ${MATERIAL}" "${KEYS}" | grep 'ssh-ed25519 ' > "${TMP}" || true
fi
cat >> "${TMP}" <<KEYS_EOF
restrict,port-forwarding,permitopen="127.0.0.1:5432",permitopen="127.0.0.1:8000",permitlisten="127.0.0.1:18095",permitlisten="127.0.0.1:18022",command="/usr/sbin/nologin" ${TUNNEL}
KEYS_EOF
install -m 644 -o root -g root "${TMP}" "${KEYS}"
rm -f "${TMP}"
echo "The box's key is installed ($(grep -c 'ssh-ed25519 ' "${KEYS}") box key(s) in all). It works only while the gateway has a box started."
