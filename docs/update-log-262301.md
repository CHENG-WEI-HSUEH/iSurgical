# Update Log — 2026-05-29

## Summary

Completed code restructure from monolithic single-file app to modular multi-file architecture, fixed all build errors, and validated the Holoscan pipeline runs end-to-end inside Docker on Jetson.

---

## Issues Solved

### 1. CMakeLists.txt — Wrong Source Paths
- **Problem:** `add_executable` referenced root-level files (`main.cpp`, `pipeline.cpp`, `yuan_capture_source.cpp`) that no longer existed after restructure.
- **Fix:** Updated paths to `src/app/main.cpp`, `src/app/pipeline/pipeline.cpp`, `src/operators/yuan_capture_source/yuan_capture_source.cpp`. Added proper include directories.

### 2. CMakeLists.txt — Invalid Library Target
- **Problem:** Linked against `holoscan::ops::yuan_capture_source` which doesn't exist (it's a custom operator, not a Holoscan built-in).
- **Fix:** Removed it. Added `${CUDA_LIBRARIES}` for cuda_runtime usage.

### 3. CMakeLists.txt — `pkg_check_modules` Case Sensitivity
- **Problem:** `imported_target` (lowercase) was treated as a package name instead of a CMake keyword.
- **Fix:** Changed to `IMPORTED_TARGET` (uppercase).

### 4. Header File — Double Dot in Filename
- **Problem:** `yuan_capture_source..hpp` (double dot) didn't match the `#include "yuan_capture_source.hpp"` in the .cpp.
- **Fix:** Renamed file to `yuan_capture_source.hpp`.

### 5. yuan_capture_source.cpp — `override` on Definitions
- **Problem:** `override` keyword used on method definitions in .cpp (only valid in class declarations).
- **Fix:** Removed `override` from all .cpp method definitions.

### 6. yuan_capture_source — `holoscan::CudaStreamHandler` Does Not Exist
- **Problem:** `CudaStreamHandler` is not part of Holoscan SDK v3.3.0.
- **Fix:** Replaced with a locally managed `cudaStream_t` (create in `start()`, destroy in `stop()`).

### 7. pipeline.cpp — Multiple Compile Errors
- **Problem:** Wrong include (`"pipeline.h"` vs `"pipeline.hpp"`), `void` return on constructor, double `<<` in template, `make_resource`/`make_operator` called as free functions.
- **Fix:** Rewrote the file with correct syntax and proper member function calls.

### 8. main.cpp — Non-existent Includes
- **Problem:** Included `"config.hpp"` and `"main.hpp"` which didn't exist.
- **Fix:** Replaced with correct includes (`<yaml-cpp/yaml.h>`, `"pipeline.hpp"`).

### 9. HolovizOp — `bad any_cast` for width/height
- **Problem:** `int` type passed to HolovizOp parameters that expect `uint32_t`.
- **Fix:** Changed to pass `uint32_t` values using `holoscan::Arg("width", win_width)` syntax.

### 10. OpenCV GStreamer — Pipeline Fails with RGB Format
- **Problem:** OpenCV's `CAP_GSTREAMER` appsink requires BGR format. Specifying `format=RGB` causes pipeline negotiation failure ("Internal data stream error").
- **Fix:** Changed appsink format to `BGR`, then convert to RGB with `cv::cvtColor` before emitting tensor.

### 11. Dockerfile — COPY of Empty/Missing Directories
- **Problem:** `COPY tests ./tests` and `COPY data/models ./data/models` failed (empty dirs, git-ignored).
- **Fix:** Removed those COPY lines from Dockerfile.

### 12. Docker Runtime — X11 Authorization / EGL Initialization
- **Problem:** "Authorization required" and "EGL failed to initialize" inside container.
- **Fix:** Run `xhost +local:docker` on host. Use `--privileged --ipc=host` flags. Set `-e DISPLAY=:1`.

---

## Current Status

- Pipeline builds and runs successfully in Docker
- Test pattern (`Device: test`) displays SMPTE color bars via HoloViz — confirmed working
- Real capture card (`Device: /dev/video0`) requires active HDMI input signal
- Yuan SC0710 currently shows "no signal, no hsync lock" — awaiting physical HDMI source connection

---

## Docker Commands

**Build:**
```bash
docker build --no-cache --network=host -f docker/Dockerfile -t isurgical-app:latest .
```

**Run:**
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
    isurgical-app:latest \
    ./bin/iSurgical configs/app.yaml
```

---

## File Structure After Restructure

```
src/
├── app/
│   ├── main.cpp              # Entry point
│   └── pipeline/
│       ├── pipeline.cpp      # InferencePipeline implementation
│       └── pipeline.hpp      # InferencePipeline class definition
├── operators/
│   └── yuan_capture_source/
│       ├── yuan_capture_source.cpp   # GStreamer V4L2 capture operator
│       └── yuan_capture_source.hpp   # Operator class definition
├── interface/                # (placeholder)
└── modules/                  # (placeholder)
```
