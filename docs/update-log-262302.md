# iSurgical Pipeline - Update Log

**Date:** 2026-05-30  
**Branch:** `refactor/code-restructure`  
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0  
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

Resolved all compilation, linking, Docker build, and runtime errors to bring the refactored iSurgical inference pipeline from non-compiling state to fully operational 4K real-time segmentation.

---

## Issues Resolved

### 1. Compilation Errors — pipeline.cpp Function Signature Mismatch

**Error Messages:**
```
error: invalid types '<unresolved overloaded function type>[const char [7]]' for array subscript
error: no declaration matches 'void InferencePipeline::ConfigurePostProcessorOperator(const YAML::Node&, ...)'
note: candidate is: 'void InferencePipeline::ConfigurePostProcessorOperator()'
```

**Root Cause:** The `.cpp` defined functions as free functions with parameters (`config`, `cuda_stream_pool`) while the header declared them as parameterless member functions using `config_` and `cuda_stream_pool_` members.

**Fix (pipeline.cpp):**
- Changed all `Configure*` functions to match the header — no parameters, using member variables `config_` and `cuda_stream_pool_` directly
- Fixed `config` → `config_` everywhere (the unresolved overload error)
- Fixed `this->InferOp` → `inference_op_` (matching header member)
- Fixed `model_path_map` declared as `model_path` → consistently `model_path_map`
- Fixed `device_memory_pool` → created local `device_allocator`
- Removed `holoscan::Fragment::make_operator` → inherited `make_operator`

### 2. Compilation Errors — Missing Includes

**Error Messages:**
```
error: 'holoscan::ops::InferenceOp' has not been declared
```

**Fix (pipeline.cpp lines 1-12):**
```cpp
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>
#include "./operators/yuan_capture_source/yuan_capture_source.hpp"
#include "./operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.hpp"
```

### 3. Linker Error — Missing Libraries

**Error Messages:**
```
/usr/bin/ld: cannot find -lyuan_capture_source: No such file or directory
/usr/bin/ld: cannot find -lyolo_seg_postprocessor_gpu: No such file or directory
```

**Root Cause:** CMakeLists.txt linked against `yuan_capture_source` and `yolo_seg_postprocessor_gpu` as if they were separate library targets, but they are compiled directly as source files in `add_executable()`.

**Fix (CMakeLists.txt):** Removed them from `target_link_libraries`.

### 4. Linker Error — Undefined CUDA Kernel References

**Error Messages:**
```
undefined reference to `UploadClassColorsToGPU'
undefined reference to `ConfidenceFilterAndNmsGPU'
undefined reference to `RenderMasksToOverlayGPU'
```

**Root Cause:** The CUDA kernel file `CUDA_Postprocess.cu` was not included in the build.

**Fix (CMakeLists.txt line 26):**
```cmake
add_executable(iSurgical
    ...
    src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess.cu
)
```

### 5. Linker Error — nvdla/cudla Undefined References

**Error Messages:**
```
/usr/bin/ld: warning: libnvdla_compiler.so, needed by /lib/aarch64-linux-gnu/libnvinfer.so.10, not found
/usr/bin/ld: /lib/aarch64-linux-gnu/libnvinfer.so.10: undefined reference to `nvdla::INetwork::addSlice(...)'
/usr/bin/ld: /usr/local/cuda/lib64/libcudla.so.1: undefined reference to `cudlaDrvDeviceGetCount'
```

**Root Cause:** The Holoscan iGPU container ships `libnvinfer.so` with DT_NEEDED entries for DLA (Deep Learning Accelerator) stub libraries (`libnvdla_compiler.so`, `libnvcudla.so`) that are not present in the container. These are optional transitive dependencies not needed for GPU-only inference.

**Fix (CMakeLists.txt line 52):**
```cmake
target_link_options(iSurgical PRIVATE "-Wl,--allow-shlib-undefined")
```

### 6. Docker Build — DNS Resolution Failure

**Error Messages:**
```
Temporary failure resolving 'ports.ubuntu.com'
E: Unable to locate package libopencv-dev
```

**Root Cause:** Docker daemon configured with `"iptables": false` in `/etc/docker/daemon.json` disables NAT, preventing containers from reaching the internet during build.

**Fix:** Build with `--network host` flag and use `RUN --network=host` in Dockerfile:
```dockerfile
# syntax=docker/dockerfile:1
RUN --network=host apt-get update && ...
```

**Build command:**
```bash
docker build --network host -f docker/Dockerfile -t isurgical-app:latest .
```

### 7. Docker Build — Package Name Errors

**Error Messages:**
```
E: Unable to locate package libgles2-mesa
E: Unable to locate package libegl1-mesa
```

**Root Cause:** Ubuntu 24.04 (Noble) renamed packages — the `-mesa` suffix was removed.

**Fix (Dockerfile):** `libgles2-mesa` → `libgles2`, `libegl1-mesa` → `libegl1`

### 8. Runtime Error — Missing Shared Libraries (libavcodec.so.60, libdc1394.so.25, etc.)

**Error Messages:**
```
./bin/iSurgical: error while loading shared libraries: libavcodec.so.60: cannot open shared object file
```

**Root Cause:** Multi-stage Docker build copied only `libopencv_*.so` but not their transitive dependencies (FFmpeg, GStreamer, gphoto2, tbb, etc. — 65+ libraries).

**Fix (Dockerfile):** Used `ldd` to collect ALL runtime dependencies from the built binary and all GStreamer plugins:
```dockerfile
RUN mkdir -p /runtime-libs /runtime-gst-plugins && \
    ldd build/iSurgical | grep "=> /" | awk '{print $3}' | \
    xargs -I{} cp -L {} /runtime-libs/ 2>/dev/null; \
    cp -a /usr/lib/aarch64-linux-gnu/gstreamer-1.0/ /runtime-gst-plugins/; \
    for lib in /usr/lib/aarch64-linux-gnu/gstreamer-1.0/*.so; do \
        ldd "$lib" 2>/dev/null | grep "=> /" | awk '{print $3}' | \
        xargs -I{} cp -Ln {} /runtime-libs/ 2>/dev/null; \
    done; true
```

### 9. Runtime Error — Bad Any Cast for resize_height/resize_width

**Error Messages:**
```
[error] [argument_setter-inl.hpp:336] Bad any cast exception caught for argument 'resize_height': bad any_cast
```

**Root Cause:** `FormatConverterOp` declares `resize_height_` / `resize_width_` as `Parameter<int32_t>`. The code passed `uint32_t` from YAML which failed the type cast.

**Fix (pipeline.cpp line 33-34):**
```cpp
holoscan::Arg("resize_height") = config_["InferenceResizeHeight"].as<int>(),
holoscan::Arg("resize_width") = config_["InferenceResizeWidth"].as<int>(),
```

### 10. Runtime Warning — ScoreThreshold Parameter Not Found

**Error Messages:**
```
[warning] component 'YoloSegPostprocesserGPUOp': Arg 'ScoreThreshold' not found in spec_.params().
The defined parameters are (MemoryAllocator, NumClass, ScoreThredshold).
```

**Root Cause:** The operator's `setup()` registers the parameter with a typo: `ScoreThredshold` (missing 'o'). The pipeline was passing `ScoreThreshold`.

**Fix (pipeline.cpp line 68):** Match the operator's registered name:
```cpp
holoscan::Arg("ScoreThredshold") = config_["ScoreThreshold"].as<float>(),
```

### 11. Runtime Error — GStreamer v4l2src Memory Allocation Failure

**Error Messages:**
```
Embedded video playback halted; module v4l2src0 reported: Failed to allocate required memory.
OpenCV | GStreamer warning: unable to start pipeline
```

**Root Cause:** The SC0710 capture card's native format is YUYV, but the pipeline didn't specify a capture format, causing GStreamer to negotiate a format requiring oversized kernel buffers.

**Fix (yuan_capture_source.cpp line 33):** Explicitly request the card's native YUYV format:
```cpp
"v4l2src device=" + device + " io-mode=mmap do-timestamp=true ! "
"video/x-raw,format=YUY2,width=3840,height=2160,framerate=30/1 ! "
"queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
"videoconvert n-threads=2 ! "
"video/x-raw,format=BGR ! "
"appsink drop=true max-buffers=2 sync=false";
```

### 12. Runtime Error — Model File Not Found

**Error Messages:**
```
[error] Invalid file path: /opt/models/yolo_seg.onnx for model: segmentation inference
```

**Fix (configs/app.yaml):** Updated path to actual model location:
```yaml
ModelPath: "/workspace/iSurgical/data/models/urology_yolov9.onnx"
```

### 13. Runtime — Black Screen (No Video Displayed)

**Root Cause:** The source video was only sent to the pre-processor, not to the visualizer. HoloViz only received the overlay tensor.

**Fix (pipeline.cpp line 102):** Added video flow to visualizer:
```cpp
add_flow(source_op_, visualizer_op_, {{"output", "receivers"}});
```

### 14. Runtime — Block Memory Pool Allocation Error

**Error Messages:**
```
[error] Requested 33177600 bytes of memory in a pool with block size 24883200
```

**Root Cause:** Post-processor outputs RGBA (4 channels) but pool was sized for RGB (3 channels).

**Fix (pipeline.cpp line 64):** Changed `3` → `4`:
```cpp
auto src_fc_block_size = config_["Width"].as<uint32_t>() * config_["Height"].as<uint32_t>() * 4 * sizeof(float);
```

---

## Final File States

### configs/app.yaml
```yaml
# Source
Device: /dev/video0
Width: 3840
Height: 2160

# Pre-processor (FormatConverter)
InferenceFormat: "float32"
InferenceResizeHeight: 640
InferenceResizeWidth: 640
NumBlocks: 4

# Inference
ModelPath: "/workspace/iSurgical/data/models/urology_yolov9.onnx"

# Post-processor
ScoreThreshold: 0.5
NumClass: 80
```

### CMakeLists.txt (Key Sections)
```cmake
add_executable(iSurgical
    src/app/main.cpp
    src/app/pipeline/pipeline.cpp
    src/operators/yuan_capture_source/yuan_capture_source.cpp
    src/operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.cpp
    src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess.cu
)

target_include_directories(iSurgical
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/app
        ${CMAKE_SOURCE_DIR}/src/app/pipeline
        ${CMAKE_SOURCE_DIR}/src/operators
        ${CMAKE_SOURCE_DIR}/src/operators/yuan_capture_source
        ${CMAKE_SOURCE_DIR}/src/operators/yolo_seg_postprocessor_gpu
)

target_link_libraries(iSurgical
    PRIVATE
        holoscan::core
        holoscan::ops::format_converter
        holoscan::ops::holoviz
        holoscan::ops::inference
        PkgConfig::YAML_CPP
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
)

target_link_options(iSurgical PRIVATE "-Wl,--allow-shlib-undefined")
```

---

## Build Instructions

### Prerequisites
- Docker with NVIDIA Container Toolkit
- `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu` base image pulled

### Build Command
```bash
cd ~/Documents/iSurgical
docker build --network host -f docker/Dockerfile -t isurgical-app:latest .
```

- `--network host` — Required because the Docker daemon has `iptables: false`, which disables container NAT. This allows the build to reach Ubuntu package mirrors via the host network stack.

---

## Run Instructions

### Run Command
```bash
xhost +local:docker

docker run --rm --runtime nvidia --privileged --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e DISPLAY=:1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/video0:/dev/video0 \
    -v "$(pwd)/configs:/workspace/iSurgical/configs" \
    -v "$(pwd)/data/models:/workspace/iSurgical/data/models" \
    isurgical-app:latest
```

### Run Command Arguments Explained

| Argument | Purpose |
|----------|---------|
| `--rm` | Automatically remove container when it exits |
| `--runtime nvidia` | Use NVIDIA Container Runtime for GPU access |
| `--privileged` | Grant full device access (needed for V4L2 capture card) |
| `--network host` | Share host network namespace (avoids NAT issues) |
| `--ipc=host` | Share host IPC namespace (required for shared memory used by Holoscan) |
| `--ulimit memlock=-1` | Unlimited locked memory (needed for CUDA pinned memory allocations) |
| `--ulimit stack=67108864` | 64MB stack size (required by Holoscan SDK for deep GXF call stacks) |
| `-e DISPLAY=:1` | Set X11 display for HoloViz Vulkan rendering window |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | Mount X11 socket for GUI display |
| `--device /dev/video0:/dev/video0` | Pass through SC0710 capture card device |
| `-v "$(pwd)/configs:..."` | Mount config directory (allows editing without rebuild) |
| `-v "$(pwd)/data/models:..."` | Mount model files (ONNX + cached TensorRT engines) |

### Test Mode (No Hardware Required)
Edit `configs/app.yaml`:
```yaml
Device: test
```
This uses GStreamer's `videotestsrc` with an SMPTE color bar pattern instead of the capture card.

---

## Pipeline Data Flow

```
YuanCaptureSource ──┬──→ FormatConverterOp ──→ InferenceOp ──→ YoloSegPostprocesserGPUOp ──┐
        (output)    │         (tensor)           (transmitter)            (out)              │
                    │                                                                        ▼
                    └───────────────────────────────────────────────────────────────→  HoloVizOp
                                              (output → receivers)                    (receivers)
```

- **YuanCaptureSource**: Captures 4K YUYV frames via GStreamer, converts to BGR, uploads to GPU
- **FormatConverterOp**: Resizes to 640x640, converts to float32 for inference
- **InferenceOp**: Runs YOLOv9 segmentation via TensorRT FP16
- **YoloSegPostprocesserGPUOp**: NMS, mask rendering, overlay generation (RGBA)
- **HoloVizOp**: Composites video + overlay via Vulkan, displays on screen
