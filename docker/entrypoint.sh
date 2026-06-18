#!/usr/bin/env bash
set -e

echo "[iSurgical] starting..."

# The host's ~/.ssh is bind-mounted read-only at /root/.ssh-host (see README / dashboard.py),
# since its files are owned by the host user, not root, and OpenSSH refuses to use keys/config
# with the wrong owner ("Bad owner or permissions"). Copy it into a root-owned, correctly
# permissioned, writable /root/.ssh before running anything that shells out to ssh/rsync.
if [ -d /root/.ssh-host ]; then
    mkdir -p /root/.ssh
    cp -a /root/.ssh-host/. /root/.ssh/
    chown -R root:root /root/.ssh
    chmod 700 /root/.ssh
    find /root/.ssh -type f -exec chmod 600 {} \;
fi

exec "$@"