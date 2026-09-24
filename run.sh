#!/usr/bin/env bash
#
# Launch the iSurgical container on the Jetson.
#
# Usage:
#   ./run.sh                          # run configs/app.yaml
#   ./run.sh configs/other.yaml       # run a different config
#   IMAGE=isurgical-app:usm-501 ./run.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-isurgical-app:usm-500}"
CONFIG="${1:-configs/app.yaml}"
DISPLAY="${DISPLAY:-:1}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"

# Holoviz talks to the X server as root inside the container.
xhost +local:docker >/dev/null
trap 'xhost -local:docker >/dev/null 2>&1 || true' EXIT

docker_args=(
    --rm -it
    --runtime nvidia
    --privileged
    --network host
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -e "DISPLAY=$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix
)

# The image ships with configs/ and data/ baked in. Shadow them with the working tree when
# it is there (development), and fall back to the baked copies when run.sh is on its own
# next to a loaded image (deployment).
for dir in configs data; do
    if [ -d "$REPO_ROOT/$dir" ]; then
        docker_args+=(-v "$REPO_ROOT/$dir:/workspace/iSurgical/$dir")
    fi
done

# Only present when running off the capture card; replay configs don't need it, and
# docker refuses to start if the device node is missing.
if [ -e "$VIDEO_DEVICE" ]; then
    docker_args+=(--device "$VIDEO_DEVICE:$VIDEO_DEVICE")
else
    echo "[run.sh] $VIDEO_DEVICE not present - starting without capture device" >&2
fi

# Mounted read-only and re-owned by entrypoint.sh; used for rsync-ing recordings off-device.
if [ -d "$HOME/.ssh" ]; then
    docker_args+=(-v "$HOME/.ssh:/root/.ssh-host:ro")
fi

exec docker run "${docker_args[@]}" "$IMAGE" ./bin/iSurgical "$CONFIG"
