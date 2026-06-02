# iSurgical Pipeline - Update Log

**Date:** 2026-06-02  
**Branch:** `refactor/code-restructure`  
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0  
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

Added an asynchronous recording operator (`AsyncRecorderOp`) that captures both the raw video frame and the segmentation overlay from the same pipeline cycle, writing them to separate MP4 files without blocking the main inference/visualization path. Encountered and resolved multiple issues during integration.

---

## What Was Added

### New Operator: `src/operators/async_recorder/`

| File | Purpose |
|------|---------|
| `async_recorder.hpp` | Class declaration with background writer thread, frame queue, raw buffer, and dual VideoWriter |
| `async_recorder.cpp` | Implementation: receive tensors, D2H copy, resize, alpha-composite, enqueue to writer thread |

### Design Decisions

1. **Same-frame guarantee** — The recorder receives the raw source frame (`source_op_ → "output"`) and the segmentation overlay (`post_processor_op_ → "out"`) as two inputs. Raw frames are buffered, and when a segmentation result arrives, it is paired with the most recent raw frame.

2. **Non-blocking (asynchronous) recording** — The `compute()` method copies GPU tensors to CPU and enqueues them into a bounded queue (max 30 frames). A background `std::thread` consumes the queue and performs the slow disk I/O (`cv::VideoWriter::write`). If the writer falls behind, frames are dropped to prevent memory blow-up.

3. **Resolution handling** — The raw source can be any resolution (e.g. 1920×1080 or 3840×2160). The recorder downscales the raw frame to match the overlay resolution (1920×1080) so both MP4s are comparable.

4. **Segmentation video compositing** — The overlay (float32 RGBA) is alpha-blended onto the downscaled raw frame to produce a "raw + segmentation" composite video.

5. **Source operator now config-driven** — Width/height passed from YAML to the GStreamer pipeline (no longer hardcoded 3840×2160).

### Pipeline Wiring (in `compose()`)

```
source_op_ ──"output"──→ recorder_op_ "raw_input"       (raw video, kNone condition)
post_processor_op_ ──"out"──→ recorder_op_ "seg_input"   (segmentation overlay, default condition)
```

### YAML Config Addition

```yaml
# Recording
RecordingPath: "/workspace/iSurgical/outputs"
```

### CMakeLists.txt Changes

- Added `src/operators/async_recorder/async_recorder.cpp` to `add_executable()`
- Added `${CMAKE_SOURCE_DIR}/src/operators/async_recorder` to `target_include_directories()`
- Added `stdc++fs` to `target_link_libraries()` for `<filesystem>` support

---

## Issues Encountered and Resolutions

### Issue 1: `nvidia::gxf::Entity::get<nvidia::gxf::Tensor>()` SFINAE Failure

**Error Message:**
```
error: no matching function for call to 'holoscan::gxf::Entity::get<nvidia::gxf::Tensor>()'
note: candidate: 'template<class DataT, class> std::shared_ptr<_Tp> holoscan::gxf::Entity::get(const char*, bool) const'
note: template argument deduction/substitution failed:
/usr/include/c++/13/type_traits:2610:11: error: no type named 'type' in 'struct std::enable_if<false, void>'
```

**Root Cause:**  
`holoscan::gxf::Entity::get<T>()` has a SFINAE `enable_if` constraint (at `entity.hpp:68`) that only accepts types in the Holoscan type system (types that have a `holoscan::Tensor` wrapper). `nvidia::gxf::Tensor` is a low-level GXF type that does NOT satisfy this constraint, so the template substitution fails.

**Fix:**  
Receive as `holoscan::TensorMap` and look up tensors by their registered names:

```cpp
// WRONG — fails at compile time:
auto entity = raw_msg.value();
auto tensor = entity.get<nvidia::gxf::Tensor>();  // SFINAE error

// CORRECT — receive as TensorMap, lookup by name:
auto raw_msg = input.receive<holoscan::TensorMap>("raw_input");
auto& raw_tmap = raw_msg.value();
auto raw_tensor = raw_tmap["video"];  // shared_ptr<holoscan::Tensor>
void* gpu_ptr = raw_tensor->data();  // access GPU memory
```

**Key Lesson:**  
Always receive messages as `holoscan::TensorMap` when you need to access tensor data. The Holoscan framework automatically unwraps GXF entities into TensorMaps regardless of how the upstream operator emitted them.

---

### Issue 2: Scheduler Deadlock — Recorder Never Executes

**Error Message:**
```
[info] [greedy_scheduler.cpp:372] Scheduler stopped: Some entities are waiting for execution,
but there are no periodic or async entities to get out of the deadlock.
```

The recorder's `start()` logged `recording to /workspace/iSurgical/outputs` but `compute()` was never called. Zero frames written.

**Root Cause:**  
The `GreedyScheduler` requires ALL input port conditions to be satisfied before scheduling an operator. With default conditions on both `raw_input` and `seg_input` (both requiring `MessageAvailableCondition(min_size=1)`), the operator only runs when BOTH ports have messages available simultaneously.

The timing issue:
1. `source_op_` produces frames at ~30fps (fast)
2. Those frames go through `pre_processor → inference → post_processor` (slow, ~100-200ms per frame)
3. By the time `post_processor` emits to `seg_input`, the scheduler has already moved on
4. The `raw_input` and `seg_input` messages never co-exist in the same scheduler tick
5. Deadlock: the recorder waits for both, but the scheduler has no way to make progress

**Fix:**  
Set `ConditionType::kNone` on `raw_input` so it doesn't gate scheduling:

```cpp
void AsyncRecorderOp::setup(holoscan::OperatorSpec& spec) {
  // kNone: doesn't block scheduling, messages queue up and are drained in compute()
  spec.input<holoscan::gxf::Entity>("raw_input")
      .condition(holoscan::ConditionType::kNone);
  
  // Default condition: operator is scheduled ONLY when seg_input has a message
  spec.input<holoscan::gxf::Entity>("seg_input");
  ...
}
```

Then in `compute()`, drain all queued raw frames first, buffer the latest one, and pair it with the arriving segmentation:

```cpp
void AsyncRecorderOp::compute(...) {
  // Drain all queued raw frames (they arrive faster than seg)
  while (true) {
    auto raw_msg = input.receive<holoscan::TensorMap>("raw_input");
    if (!raw_msg) break;
    // ... copy and buffer the latest raw frame ...
  }

  // This triggers compute() — seg arrived
  auto seg_msg = input.receive<holoscan::TensorMap>("seg_input");
  if (!seg_msg) return;
  
  // Pair with latest buffered raw frame
  // ... composite and enqueue ...
}
```

**Key Lesson:**  
In Holoscan with `GreedyScheduler`:
- Default input condition = `MessageAvailableCondition(min_size=1)` → blocks scheduling until message arrives
- `ConditionType::kNone` = no condition → port does NOT block scheduling
- If you have a fast producer and slow producer feeding the same operator, set `kNone` on the fast one and let the slow one gate scheduling
- Messages on `kNone` ports queue up and can be consumed via `receive()` during compute

---

### Issue 3: `std::atomic<uint64_t>` Not Formattable by `fmt`

**Error Message:**
```
/opt/nvidia/holoscan/include/fmt/core.h:1576:63: error: 'fmt::v10::detail::type_is_unformattable_for<
    std::atomic<long unsigned int>, char> _' has incomplete type
fmt/core.h:1580:7: error: static assertion failed: Cannot format an argument.
    To make type T formattable provide a formatter<T> specialization
```

Triggered at:
```cpp
HOLOSCAN_LOG_INFO("AsyncRecorderOp: stopped, wrote {} frames", frames_written_);
```

**Root Cause:**  
Holoscan's logging uses `fmt` v10, which cannot directly format `std::atomic<T>` types. The `fmt` library requires either a built-in type or a custom `formatter<T>` specialization.

**Fix:**  
Call `.load()` to extract the plain integer value before passing to the formatter:

```cpp
// WRONG — atomic type is not formattable:
HOLOSCAN_LOG_INFO("wrote {} frames", frames_written_);

// CORRECT — .load() returns uint64_t which fmt can format:
HOLOSCAN_LOG_INFO("wrote {} frames", frames_written_.load());
```

**Key Lesson:**  
Never pass `std::atomic<T>` directly to `HOLOSCAN_LOG_*` macros, `fmt::format()`, or `fmt::print()`. Always call `.load()` first.

---

### Issue 4: Capture Card Resolution Mismatch

**Error Message:**
```
[ WARN:0@0.557] global cap_gstreamer.cpp (2401) handleMessage OpenCV | GStreamer warning:
Embedded video playback halted; module v4l2src0 reported: Device '/dev/video0' cannot capture at 3840x2160
[ WARN:0@0.558] global cap_gstreamer.cpp (1356) open OpenCV | GStreamer warning: unable to start pipeline
[error] Exception occurred when starting operator: 'YuanCaptureSource' - Failed to open capture source via GStreamer: /dev/video0
```

**Root Cause:**  
The GStreamer pipeline in `yuan_capture_source.cpp` had the resolution **hardcoded** to 3840×2160:

```cpp
pipeline = "v4l2src device=" + device + " ! "
           "video/x-raw,format=YUY2,width=3840,height=2160,framerate=30/1 ! " ...
```

But the actual signal on the capture card was 1920×1080 (verified via `v4l2-ctl --query-dv-timings` showing `Active width: 0, Active height: 0` for no signal, and `./tests/unit/qcap/qcap_fullscreen` showing 1920×1080).

**Diagnosis Steps:**
```bash
# Check what the card currently detects:
v4l2-ctl --device=/dev/video0 --query-dv-timings

# Check supported modes:
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Test capture independently:
./tests/unit/qcap/qcap_fullscreen  # reported "Frame size: 1920 x 1080"
```

**Fix:**  
Made the source operator read width/height from YAML config instead of hardcoding:

```cpp
// yuan_capture_source.cpp — setup()
spec.param(this->width_, "width", "Width", "Capture width", 1920);
spec.param(this->height_, "height", "Height", "Capture height", 1080);

// yuan_capture_source.cpp — start()
int w = width_.get();
int h = height_.get();
pipeline = "v4l2src device=" + device + " io-mode=mmap do-timestamp=true ! "
           "video/x-raw,format=YUY2,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=30/1 ! " ...
```

Pipeline configuration passes the values:
```cpp
// pipeline.cpp
source_op_ = make_operator<YuanCaptureSourceOp>(
    "YuanCaptureSource",
    holoscan::Arg("device") = config_["Device"].as<std::string>(),
    holoscan::Arg("width") = config_["Width"].as<int>(),
    holoscan::Arg("height") = config_["Height"].as<int>(),
    holoscan::Arg("allocator") = source_allocator
);
```

**Key Lesson:**  
- Never hardcode resolution in GStreamer pipelines — always parameterize from config
- Use `v4l2-ctl --query-dv-timings` to check the ACTUAL input signal resolution
- The card supports many resolutions (check `--list-formats-ext`) but can only capture at the resolution the source is currently outputting
- If `Active width: 0, Active height: 0`, there is no signal connected

---

### Issue 5: HoloViz GLFW Initialization Failure (X11 Display)

**Error Message:**
```
error: XDG_RUNTIME_DIR is invalid or not set in the environment.
Authorization required, but no authorization protocol specified
Glfw Error 65550: Failed to detect any supported platform
[error] Exception occurred when starting operator: 'HoloViz' - Failed to initialize glfw
```

**Root Cause:**  
The Docker container cannot connect to the host's X11 display server. Two conditions must be met:
1. `xhost` must allow the container's connection
2. The `DISPLAY` environment variable must match the host's active display

**Fix:**  
Before running the container:
```bash
# Allow docker containers to access X11:
xhost +local:docker
```

Verify the correct display:
```bash
echo $DISPLAY  # Usually :0 or :1
```

If it's `:0`, change the docker run flag to `-e DISPLAY=:0`.

**Key Lesson:**  
- Always run `xhost +local:docker` before launching containers that need display access
- The display number (`:0` vs `:1`) must match the host — check with `echo $DISPLAY`
- If running headless (SSH without desktop), there's no X server — use `Xvfb :1 &` to create a virtual display, or skip HoloViz entirely

---

### Issue 6: Container Running Old Binary After Source Changes

**Symptom:**  
After fixing the source code (e.g., removing hardcoded resolution), the container still showed the old error because the Docker image hadn't been rebuilt.

**Root Cause:**  
Source code changes only exist on the host filesystem. The Docker image contains a compiled binary from the last `docker build`. Volume-mounting only mounts `configs/` and `data/` — NOT the binary.

**Fix:**  
Always rebuild after source changes:
```bash
docker build --network host -f docker/Dockerfile -t isurgical-app:latest .
```

**Key Lesson:**  
- YAML config changes take effect immediately (volume-mounted)
- C++ source code changes require a full Docker rebuild
- If you see an error that should have been fixed, the first question is: "Did I rebuild?"

---

## Holoscan Data Types: `nvidia::gxf::Entity` vs `holoscan::TensorMap` vs `holoscan::Tensor`

### Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Holoscan Application Layer (C++ operators)             │
│                                                         │
│  holoscan::TensorMap  ←  what you receive/emit          │
│  holoscan::Tensor     ←  what you read data from        │
├─────────────────────────────────────────────────────────┤
│  GXF Runtime Layer (execution engine)                   │
│                                                         │
│  nvidia::gxf::Entity  ←  message container              │
│  nvidia::gxf::Tensor  ←  raw memory descriptor          │
└─────────────────────────────────────────────────────────┘
```

### `nvidia::gxf::Entity`

| Aspect | Detail |
|--------|--------|
| **Level** | GXF runtime (low-level) |
| **What it is** | A generic container that holds named components (tensors, timestamps, metadata) |
| **When to use** | When creating entities from scratch (e.g., `Entity::New(context.context())`) or emitting raw GXF messages. Used in source operators and post-processors that construct output entities manually. |
| **Key methods** | `add<T>(name)` — add a typed component; `get<T>(name)` — retrieve component handle |
| **Example in project** | `yuan_capture_source.cpp` creates a GXF entity, adds a `nvidia::gxf::Tensor`, reshapes it, and emits it |

```cpp
auto entity = nvidia::gxf::Entity::New(context.context());
auto tensor = entity.value().add<nvidia::gxf::Tensor>("video");
tensor.value()->reshape<uint8_t>(shape, kDevice, allocator);
// ... cudaMemcpyAsync into tensor.value()->pointer() ...
op_output.emit(holoscan::gxf::Entity(std::move(entity.value())), "output");
```

### `holoscan::TensorMap`

| Aspect | Detail |
|--------|--------|
| **Level** | Holoscan application layer (high-level) |
| **What it is** | A `std::unordered_map<std::string, std::shared_ptr<holoscan::Tensor>>` — a named collection of tensors extracted from a GXF Entity |
| **When to use** | When **receiving** messages in `compute()`. The Holoscan framework automatically unwraps the incoming GXF Entity into a TensorMap. This is the standard receive type for most operators. |
| **Key methods** | `find(name)`, `operator[](name)` — standard map access |
| **Example in project** | `MP4Writer.cpp` and `async_recorder.cpp` receive frames this way |

```cpp
auto maybe_msg = input.receive<holoscan::TensorMap>("in_frame");
auto& tmap = maybe_msg.value();
auto frame = tmap["video_rgb"];  // shared_ptr<holoscan::Tensor>
```

### `holoscan::Tensor`

| Aspect | Detail |
|--------|--------|
| **Level** | Holoscan application layer (high-level) |
| **What it is** | A wrapper around `nvidia::gxf::Tensor` that provides safe access to tensor metadata and data pointers |
| **When to use** | After extracting from a TensorMap — use it to get shape, dtype, strides, and the raw `data()` pointer for CUDA operations |
| **Key methods** | `data()` — void* to GPU/CPU memory; `shape()` — DLTensor dimensions; `nbytes()` — total byte size; `ndim()` — rank |
| **Example in project** | Post-processor accesses inference outputs |

```cpp
auto predictions = tmap["outputs"];           // shared_ptr<holoscan::Tensor>
const float* gpu_ptr = static_cast<const float*>(predictions->data());
// Launch CUDA kernels on gpu_ptr...
```

### Decision Table: Which Type to Use Where

| Scenario | Type | Reason |
|----------|------|--------|
| Receiving input in `compute()` | `holoscan::TensorMap` | Framework handles Entity→TensorMap unwrapping automatically |
| Accessing tensor data after receive | `holoscan::Tensor` (via `shared_ptr`) | Provides `data()`, `shape()`, type-safe access |
| Creating new output from scratch | `nvidia::gxf::Entity` | Need to allocate GPU memory via GXF allocator, add tensor components |
| Emitting existing entity (pass-through) | `nvidia::gxf::Entity` / `holoscan::gxf::Entity` | Re-emit an entity you constructed or received without modification |
| Registering input/output ports in `setup()` | Template type in `spec.input<T>()` | Use `holoscan::gxf::Entity` for port declaration, `holoscan::TensorMap` for receive |

### Scheduling Conditions on Input Ports

| Condition | Behavior | When to Use |
|-----------|----------|-------------|
| Default (none specified) | `MessageAvailableCondition(min_size=1)` — blocks scheduling until at least one message is available on this port | Standard case: operator must have input before running |
| `ConditionType::kNone` | No condition — port does NOT block scheduling | Fast-producer input that you want to buffer; "optional" inputs |
| `ConditionType::kMessageAvailable` | Explicit message condition with custom min_size | When you need N messages before running |

---

## Updated Pipeline Data Flow

```
                                                                                    ┌──────────────┐
YuanCaptureSource ──┬──→ FormatConverterOp ──→ InferenceOp ──→ PostprocessorGPU ──┬──→  HoloVizOp  │
     (output)       │         (tensor)          (transmitter)       (out)          │   (receivers)  │
                    │                                                               │               │
                    └───────────────────────────────────────────────────────────────┼──→ HoloVizOp  │
                    │                         (output → receivers)                   └──────────────┘
                    │                                                               
                    │                                                      ┌────────────────────────┐
                    └───────("output" → "raw_input" [kNone])─────────────→ │    AsyncRecorderOp     │
                                                                           │  raw_input: buffered    │
PostprocessorGPU ──────────("out" → "seg_input" [default])───────────────→ │  seg_input: triggers    │
                                                                           │  writer_thread → MP4    │
                                                                           └────────────────────────┘
```

### Output Files

| File | Content | Resolution |
|------|---------|-----------|
| `{RecordingPath}/raw.mp4` | Raw captured video (no segmentation), BGR | 1920×1080 |
| `{RecordingPath}/segmentation.mp4` | Raw + segmentation overlay alpha-composited | 1920×1080 |

---

## Docker Run Command

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

Note: `-v $(pwd)/outputs:/workspace/iSurgical/outputs` persists recorded MP4s on the host.

---

## Quick Troubleshooting Guide

| Symptom | Check | Fix |
|---------|-------|-----|
| `cannot capture at WxH` | `v4l2-ctl --query-dv-timings` | Match `Width`/`Height` in YAML to actual signal |
| `Failed to initialize glfw` | `echo $DISPLAY` on host | Run `xhost +local:docker`, match `-e DISPLAY=:X` |
| Scheduler deadlock (no frames written) | Check if both inputs have default conditions | Use `kNone` on fast-producer input port |
| No files in outputs/ | Check volume mount | Add `-v $(pwd)/outputs:/workspace/iSurgical/outputs` |
| Old bug still appears after fix | Binary is stale | Rebuild: `docker build --network host -f docker/Dockerfile -t isurgical-app:latest .` |
| `type_is_unformattable_for<std::atomic<...>>` | Passing atomic to LOG macro | Use `.load()`: `frames_written_.load()` |
| `enable_if<false, void>` on Entity::get | Using GXF type in Holoscan API | Receive as `holoscan::TensorMap`, not `nvidia::gxf::Tensor` |
