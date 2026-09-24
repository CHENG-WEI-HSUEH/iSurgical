#!/usr/bin/env bash
#
# Package iSurgical as a single self-contained docker image tarball.
#
# The image carries the compiled binary, configs/, the ONNX models, the prebuilt TensorRT
# engine and the replay clip. It carries no source code - the runtime stage of
# docker/Dockerfile never sees src/.
#
# Usage:
#   ./scripts/package.sh /media/pandora/Transcend           # build, then save
#   ./scripts/package.sh --no-build /media/pandora/Transcend
#   IMAGE=isurgical-app:usm-501 ./scripts/package.sh /path/to/drive
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-isurgical-app:usm-500}"
BUILD=1

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0; shift ;;
        -h|--help)  sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "unknown option: $1" >&2; exit 2 ;;
        *)          OUT_DIR="$1"; shift ;;
    esac
done

if [ -z "${OUT_DIR:-}" ]; then
    echo "usage: $0 [--no-build] <output-directory>" >&2
    echo "  the repo disk is full - point this at external media" >&2
    exit 2
fi

[ -d "$OUT_DIR" ] || { echo "no such directory: $OUT_DIR" >&2; exit 1; }

OUT_FILE="$OUT_DIR/${IMAGE//[:\/]/-}.tar.gz"

if [ "$BUILD" -eq 1 ]; then
    echo "==> building $IMAGE"
    docker build --network=host -f docker/Dockerfile -t "$IMAGE" .
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "image $IMAGE not found - drop --no-build to build it first" >&2
    exit 1
}

# `docker save` writes every layer uncompressed, so the pipe needs roughly the image size
# before gzip gets to it. Refuse up front rather than dying half-written.
image_bytes=$(docker image inspect "$IMAGE" --format '{{.Size}}')
avail_bytes=$(( $(df --output=avail "$OUT_DIR" | tail -1) * 1024 ))
echo "==> image $IMAGE is $(numfmt --to=iec "$image_bytes"), $(numfmt --to=iec "$avail_bytes") free on $OUT_DIR"
if [ "$avail_bytes" -lt "$image_bytes" ]; then
    echo "not enough free space on $OUT_DIR (want at least the uncompressed image size)" >&2
    exit 1
fi

# dockerd stages the whole export under its root dir before streaming a byte, so the
# internal disk needs the image size free too - running it dry takes the system down with it.
docker_root=$(docker info --format '{{.DockerRootDir}}')
root_avail=$(( $(df --output=avail "$docker_root" | tail -1) * 1024 ))
echo "==> $(numfmt --to=iec "$root_avail") free on $docker_root for docker's staging copy"
if [ "$root_avail" -lt "$image_bytes" ]; then
    echo "not enough free space on $docker_root (want at least the uncompressed image size)" >&2
    exit 1
fi

# Split into parts below FAT32's 4 GiB per-file cap so the package fits on any USB stick.
PART_PREFIX="$OUT_FILE.part-"
rm -f "$PART_PREFIX"*
echo "==> saving to ${PART_PREFIX}*"
# Half-written parts look like a valid deliverable, so drop them unless the pipe finishes.
trap 'rm -f "$PART_PREFIX"*' EXIT
docker save "$IMAGE" | gzip -1 | split -b 3900M -a 2 - "$PART_PREFIX"
trap - EXIT
cp run.sh "$OUT_DIR/run.sh"

cat <<EOF

Done: $(ls "$PART_PREFIX"* | wc -l) parts, $(du -ch "$PART_PREFIX"* | tail -1 | cut -f1) total
$(ls -1 "$PART_PREFIX"* | xargs -n1 basename | sed 's/^/  /')

On the target machine:
  cat $(basename "$PART_PREFIX")* | gunzip | docker load
  ./run.sh                      # uses the configs/ and data/ baked into the image

To override the baked-in config or models, put a configs/ or data/ directory next to
run.sh and it will be mounted over the image's copy.
EOF
