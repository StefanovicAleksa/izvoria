#!/bin/bash
# Run on the VPS, as root, with the two public keys the GPU box's provision.sh printed (installed as
# /usr/local/sbin/izvoria-add-box-keys by install_gateway.sh):
#   izvoria-add-box-keys 'ssh-ed25519 AAAA… izvoria-box tunnel' 'ssh-ed25519 AAAA… izvoria-box sync'
# Replaces the keys of any earlier box: one rented box at a time.
#
# Each key may do one thing only:
#   tunnel: forward this machine's Postgres to the box, and the box's API back to 127.0.0.1:18095 here; nothing else
#   sync:   rsync inside /workspace/tenants (rrsync), to stage one ingestion job's files in and its results out
set -euo pipefail
TUNNEL="${1:?usage: izvoria-add-box-keys '<tunnel key>' '<sync key>'}"
SYNC="${2:?usage: izvoria-add-box-keys '<tunnel key>' '<sync key>'}"
for key in "${TUNNEL}" "${SYNC}"; do
    case "${key}" in ssh-ed25519\ *) ;; *) echo "Not an ed25519 public key: ${key}" >&2; exit 1 ;; esac
done
RRSYNC="$(cat /etc/onyx-web/rrsync-path)"
TMP="$(mktemp)"
cat > "${TMP}" <<KEYS
restrict,port-forwarding,permitopen="127.0.0.1:5432",permitlisten="127.0.0.1:18095",command="/usr/sbin/nologin" ${TUNNEL}
restrict,command="${RRSYNC} /workspace/tenants" ${SYNC}
KEYS
install -m 644 -o root -g root "${TMP}" /etc/onyx-web/box_authorized_keys
rm -f "${TMP}"
echo "The box's keys are installed. They work only while the gateway has the box started."
