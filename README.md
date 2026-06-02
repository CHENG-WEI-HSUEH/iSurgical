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
docker/
├── Dockerfile
└── entrypoint.sh
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
docker build --network=host -f docker/Dockerfile -t isurgical-app:latest .
docker build --no-cache --network=host -f docker/Dockerfile -t isurgical-app:latest .
```

## Run

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
      -v $(pwd)/outputs:/workspace/iSurgical/outputs \
      isurgical-app:latest \
      ./bin/iSurgical configs/app.yaml
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Authorization required` / `EGL failed` | Run `xhost +local:docker` before starting the container |
| `No input source was detected` | No HDMI signal — check cable and source device |
| `Failed to open QCAP device` | Verify device exists: `v4l2-ctl -d /dev/video0 --get-input` |
| `bad any_cast` for width/height | Ensure config values are integers, not strings |
| Test without hardware | Set `Device: test` in `configs/app.yaml` |
