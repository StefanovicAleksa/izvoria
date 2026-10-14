#!/bin/bash
# Run on the VPS, as root, with the public key the GPU box's provision.sh printed (installed as
# /usr/local/sbin/izvoria-add-box-keys by install_gateway.sh):
#   izvoria-add-box-keys 'ssh-ed25519 AAAA… izvoria-box tunnel'
# Replaces the key of any earlier box: one rented box at a time.
#
# The key may only open the box's tunnels, nothing else (no shell, no files):
#   -L to this machine's Postgres (5432) and the gateway (8000: the box's database password, file staging)
#   -R back to the box's API (18095) and the box's own sshd (18022: the VPS moves one job's files through it)
# and it is offered to sshd only while the gateway has the box started (izvoria-box-keys).
set -euo pipefail
TUNNEL="${1:?usage: izvoria-add-box-keys '<tunnel key>'}"
case "${TUNNEL}" in ssh-ed25519\ *) ;; *) echo "Not an ed25519 public key: ${TUNNEL}" >&2; exit 1 ;; esac
TMP="$(mktemp)"
cat > "${TMP}" <<KEYS
restrict,port-forwarding,permitopen="127.0.0.1:5432",permitopen="127.0.0.1:8000",permitlisten="127.0.0.1:18095",permitlisten="127.0.0.1:18022",command="/usr/sbin/nologin" ${TUNNEL}
KEYS
install -m 644 -o root -g root "${TMP}" /etc/onyx-web/box_authorized_keys
rm -f "${TMP}"
echo "The box's key is installed. It works only while the gateway has the box started."
