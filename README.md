# iSurgical

Real-time surgical video inference pipeline built on NVIDIA Holoscan SDK, designed for Jetson AGX platforms with Yuan SC0710 capture cards.

## Prerequisites

- NVIDIA Jetson AGX Orin (JetPack 6.x)
- Docker with NVIDIA runtime
- Yuan SC0710 capture card (or compatible V4L2 device)
- HDMI input source

## Project Structure

```
src/
├── app/
│   ├── main.cpp                 # Entry point
│   └── pipeline/
│       ├── pipeline.cpp         # InferencePipeline implementation
│       └── pipeline.hpp         # InferencePipeline class definition
├── operators/
│   └── yuan_capture_source/
│       ├── yuan_capture_source.cpp
│       └── yuan_capture_source.hpp
├── interface/
└── modules/
configs/
└── app.yaml                     # Runtime configuration
data/
├── models/                      # .onnx + generated .engine.fp16
└── inputs/                      # Raw.gxf_entities / Raw.gxf_index replay clip
docker/
├── Dockerfile
└── entrypoint.sh
scripts/
└── package.sh                   # Build + save the deployable image tarball
run.sh                           # Launch the container
```

## Configuration

Edit `configs/app.yaml`:

```yaml
Device: /dev/video0    # Use "test" for SMPTE test pattern
Width: 1920
Height: 1080
Channels: 3
```

## Build

```bash
docker build --network=host -f docker/Dockerfile -t isurgical-app:usm-500 .
docker build --no-cache --network=host -f docker/Dockerfile -t isurgical-app:usm-500 .
```

## Run

```bash
./run.sh                        # configs/app.yaml
./run.sh configs/other.yaml
IMAGE=isurgical-app:usm-501 ./run.sh
```

`run.sh` handles `xhost +local:docker`, the NVIDIA runtime flags and the X11 socket. It
mounts the working tree's `configs/` and `data/` over the copies baked into the image, and
skips `--device /dev/video0` when the capture card isn't present (replay configs don't need
it). Override the device with `VIDEO_DEVICE=`, the display with `DISPLAY=`.

## Package for deployment

```bash
./scripts/package.sh /media/pandora/Transcend
./scripts/package.sh --no-build /media/pandora/Transcend    # save the existing image as-is
```

Builds the image and writes `isurgical-app-usm-500.tar.gz` — split into
`.part-aa`, `.part-ab`, … chunks under FAT32's 4 GB file limit — plus a copy of `run.sh` to the
given directory. The tarball is self-contained — compiled binary, `configs/`, the ONNX
models, the prebuilt TensorRT engine and the replay clip — and contains **no source code**,
since the runtime stage of the Dockerfile never sees `src/`. The image is ~17 GB
uncompressed, so this needs external media, and dockerd also stages a full copy under
`/var/lib/docker` first, so the internal disk needs that much free as well.

On the target:

```bash
cat isurgical-app-usm-500.tar.gz.part-* | gunzip | docker load
./run.sh
```

The TensorRT engine is built for this exact Orin + TensorRT 10.3 combination. On different
hardware it is ignored and regenerated from the ONNX on first run (~10 minutes).

## Profiling

```bash
  xhost +local:docker
  docker run --rm -it \
      --runtime nvidia \
      --privileged \
      --network host \
      --ipc=host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -e DISPLAY=:1 \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      --device /dev/video0:/dev/video0 \
      -v $(pwd)/configs:/workspace/iSurgical/configs \
      -v $(pwd)/data:/workspace/iSurgical/data \
      -v $(pwd)/scripts:/workspace/iSurgical/scripts \
      -v $HOME/.ssh:/root/.ssh-host:ro \
      -v $(pwd)/nsight_reports:/workspace/iSurgical/nsight_reports \
      -v /opt/nvidia/nsight-systems/2024.5.4:/opt/nvidia/nsight-systems/2024.5.4 \
      isurgical-app:usm-500 \
      bash -c "/opt/nvidia/nsight-systems/2024.5.4/bin/nsys profile \
        --output /workspace/iSurgical/nsight_reports/isurgical_report \
        --trace cuda,nvtx,osrt \
        --force-overwrite true \
        ./bin/iSurgical configs/app.yaml"

```
## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Authorization required` / `EGL failed` | Run `xhost +local:docker` before starting the container |
| `No input source was detected` | No HDMI signal — check cable and source device |
| `Failed to open QCAP device` | Verify device exists: `v4l2-ctl -d /dev/video0 --get-input` |
| `bad any_cast` for width/height | Ensure config values are integers, not strings |
| Test without hardware | Set `Device: test` in `configs/app.yaml` |
