# iSurgical Pipeline - Update Log

**Date:** 2026-06-02 ~ 2026-06-03  
**Branch:** `refactor/code-restructure`  
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0  
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

Added a `SurgicalPhaseOp` operator that performs periodic surgical phase recognition using a temporal ONNX model (`surgical_phase_model.onnx`) via TensorRT. The operator runs fully asynchronously — it does not block the main pipeline. Also resolved several issues including OpenCV DNN dynamic shape limitation, TensorRT 10.x API changes, operator shutdown hangs, and a 4K capture resolution mismatch.

---

## What Was Added

### New Operator: `src/operators/surgical_phase/`

| File | Purpose |
|------|---------|
| `surgical_phase.hpp` | Class declaration with background worker thread, TensorRT engine, frame buffer |
| `surgical_phase.cpp` | Implementation: TensorRT engine build/load, periodic frame sampling, inference, phase output |

### Design Decisions

1. **Fully asynchronous** — The operator's `compute()` only stores the latest frame (GPU→CPU copy). A background `worker_thread_` handles engine initialization, periodic timing, frame sampling, and inference — none of this blocks the main pipeline.

2. **Interruptible waits** — All sleep/wait operations use `std::condition_variable::wait_for` instead of `std::this_thread::sleep_for`. This allows the operator to shut down immediately on Ctrl+C without hanging.

3. **TensorRT inference** — OpenCV DNN (v4.6) cannot handle ONNX models with dynamic shapes. TensorRT natively supports dynamic shapes via optimization profiles, and is significantly faster on Jetson.

4. **Engine caching** — First run builds the TensorRT engine from ONNX (takes several minutes) and saves it to `surgical_phase_model.onnx.engine`. Subsequent runs load the cached engine instantly.

5. **Periodic sampling** — Every `interval_ms` (default 60s, set to 10s for testing), the operator samples `seq_len` frames at 1-second intervals (1 fps × 5 seconds), resizes each to 224×224, normalizes to [0,1] float32, packs into shape `(1, 5, 3, 224, 224)`, and runs inference.

6. **Output** — Prints the most likely surgical phase to the log:
   ```
   [info] [SurgicalPhase] Current phase: CalotTriangleDissection
   ```

### Model Specification

| Property | Value |
|----------|-------|
| Input name | `frames` |
| Input shape | `(batch_size, seq_len, 3, 224, 224)` |
| Output name | `logits` |
| Output shape | `(batch_size, seq_len, 7)` |
| Phases | Preparation, CalotTriangleDissection, ClippingAndCutting, GallbladderDissection, GallbladderPackaging, CleaningAndCoagulation, GallbladderRetraction |

### Pipeline Wiring (in `compose()`)

```
source_op_ ──"output"──→ surgical_phase_op_ "frame_input"   (kNone condition, non-blocking)
```

### YAML Config Addition

```yaml
# Surgical Phase Recognition
PhaseModelPath: "/workspace/iSurgical/data/models/surgical_phase_model.onnx"
```

### CMakeLists.txt Changes

- Added `src/operators/surgical_phase/surgical_phase.cpp` to `add_executable()`
- Added `${CMAKE_SOURCE_DIR}/src/operators/surgical_phase` to `target_include_directories()`
- Added `nvinfer` and `nvonnxparser` to `target_link_libraries()`

---

## Issues Encountered and Resolutions

### Issue 1: OpenCV DNN Cannot Load Dynamic-Shape ONNX Model

**Error Message:**
```
[ERROR:0@1.193] global onnx_importer.cpp (2845) parseShape DNN/ONNX(Shape): dynamic 'zero' shapes are not supported, input frames [ 0 0 3 224 224 ]
error: (-215:Assertion failed) !isDynamicShape in function 'parseShape'
```

**Root Cause:**  
The ONNX model was exported with dynamic `batch_size` and `seq_len` dimensions (represented as 0 in shape). OpenCV DNN v4.6 does not support dynamic shapes — it requires all dimensions to be fixed at load time.

**Fix:**  
Replaced OpenCV DNN with TensorRT native inference. TensorRT supports dynamic shapes via optimization profiles:

```cpp
auto profile = builder->createOptimizationProfile();
nvinfer1::Dims input_dims{};
input_dims.nbDims = 5;
input_dims.d[0] = 1;    // batch
input_dims.d[1] = seq;  // seq_len
input_dims.d[2] = 3;    // channels
input_dims.d[3] = 224;  // height
input_dims.d[4] = 224;  // width
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
```

**Key Lesson:**  
OpenCV DNN is unsuitable for models with dynamic batch or sequence dimensions. Use TensorRT (available in the Holoscan container) for production deployment on Jetson.

---

### Issue 2: TensorRT 10.x API — `Dims5` Removed

**Error Message:**
```
error: 'Dims5' is not a member of 'nvinfer1'; did you mean 'Dims4'?
```

**Root Cause:**  
TensorRT 10.x removed the `Dims2`, `Dims3`, `Dims4`, `Dims5` convenience types. Only `nvinfer1::Dims` (with `nbDims` and `d[]` array) remains.

**Fix:**  
Use `nvinfer1::Dims` with explicit `nbDims`:

```cpp
// WRONG (TRT 8.x only):
nvinfer1::Dims5 input_dims{1, 5, 3, 224, 224};

// CORRECT (TRT 10.x):
nvinfer1::Dims input_dims{};
input_dims.nbDims = 5;
input_dims.d[0] = 1;
input_dims.d[1] = 5;
input_dims.d[2] = 3;
input_dims.d[3] = 224;
input_dims.d[4] = 224;
```

Also deprecated in TRT 10.x:
- `NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` → pass `0` to `createNetworkV2()` (explicit batch is now the default)
- `builder->platformHasFastFp16()` → just set `BuilderFlag::kFP16` (Jetson Orin always supports FP16)

---

### Issue 3: Operator Cannot Stop (Ctrl+C Hangs)

**Symptom:**  
After pressing Ctrl+C, the application hangs for 3 seconds then force-exits:
```
[error] Application did not shut down within 3 seconds of interrupt. Forcing exit...
```

**Root Cause:**  
`std::this_thread::sleep_for()` is not interruptible. When `stop()` sets `is_running_ = false`, the background thread is stuck inside a 60-second sleep and cannot wake until it expires.

**Fix:**  
Replace `sleep_for` with `condition_variable::wait_for` that wakes on stop signal:

```cpp
// In header:
std::mutex stop_mutex_;
std::condition_variable stop_cv_;

// Interruptible wait helper:
bool SurgicalPhaseOp::waitFor(std::chrono::milliseconds duration) {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  return !stop_cv_.wait_for(lock, duration, [this]() { return !is_running_.load(); });
}

// In stop():
{
  std::lock_guard<std::mutex> lock(stop_mutex_);
  is_running_ = false;
}
stop_cv_.notify_all();
```

**Key Lesson:**  
Never use `sleep_for` in Holoscan operators with long intervals. Use `condition_variable::wait_for` so `stop()` can wake the thread immediately. This same fix was also applied to `PeriodicCounterOp`.

---

### Issue 4: TensorRT Engine Build Blocks Pipeline Start

**Symptom:**  
Building the TensorRT engine from ONNX takes several minutes. During this time, the entire pipeline is frozen — no video display, no inference.

**Root Cause:**  
Engine build was in `start()`, which blocks the Holoscan scheduler from starting any other operators.

**Fix:**  
Moved engine build into the background `workerLoop()` thread. The `start()` method only spawns the thread and returns immediately:

```cpp
void SurgicalPhaseOp::start() {
  is_running_ = true;
  worker_thread_ = std::thread(&SurgicalPhaseOp::workerLoop, this);
  // Returns immediately — pipeline continues starting other operators
}

void SurgicalPhaseOp::workerLoop() {
  // Engine build happens here, in the background
  if (!buildEngine()) { return; }
  // Then periodic inference loop...
}
```

**Key Lesson:**  
Never perform heavy initialization (model loading, TensorRT builds) inside Holoscan `start()`. Use a background thread so the pipeline can begin processing immediately.

---

### Issue 5: HoloViz Display Enlarged (Only Top-Left Visible)

**Symptom:**  
After integrating the surgical phase operator, the HoloViz window shows a zoomed-in view — only the top-left portion of the frame is visible. Previously the full frame was displayed correctly.

**Root Cause:**  
The Yuan capture card's HDMI source changed from 1920×1080 to 3840×2160 (verified with `v4l2-ctl --all` showing `Width/Height: 3840/2160`). The GStreamer pipeline requested `YUY2,1920x1080,30fps` but the card now outputs `BGR3,3840x2160,60fps`. GStreamer may silently negotiate a different format/resolution, delivering 4K frames that get interpreted as 1080p by the tensor shape — causing a quarter-frame display.

**Status:**  
The source operator now includes a `cv::resize` fallback in `compute()` to scale frames to the configured resolution regardless of what the card delivers. If the card output changes resolution, the pipeline adapts:

```cpp
int target_w = width_.get();
int target_h = height_.get();
if (frame.cols != target_w || frame.rows != target_h) {
  cv::resize(frame, frame, cv::Size(target_w, target_h));
}
```

**Note:** This issue is independent of the surgical phase operator — it was caused by a change in the HDMI source resolution.

---

## Updated Pipeline Data Flow

```
                                                                                  ┌──────────────┐
YuanCaptureSource ──┬──→ FormatConverterOp ──→ InferenceOp ──→ PostprocessorGPU ─┬──→  HoloVizOp  │
     (output)       │         (tensor)          (transmitter)       (out)         │   (receivers)  │
                    │                                                              │               │
                    └──────────────────────────────────────────────────────────────┼──→ HoloVizOp  │
                    │                         (output → receivers)                  └──────────────┘
                    │
                    │                                                     ┌────────────────────────┐
                    ├───────("output" → "raw_input" [kNone])────────────→ │    AsyncRecorderOp     │
                    │                                                     └────────────────────────┘
                    │
                    │                                                     ┌────────────────────────┐
                    └───────("output" → "frame_input" [kNone])──────────→ │   SurgicalPhaseOp      │
                                                                          │  background worker:     │
                                                                          │  - build TRT engine     │
                                                                          │  - every 60s: sample    │
                                                                          │    5 frames @ 1fps      │
                                                                          │  - infer → print phase  │
                                                                          └────────────────────────┘
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/operators/surgical_phase/surgical_phase.hpp` | New — operator header |
| `src/operators/surgical_phase/surgical_phase.cpp` | New — operator implementation |
| `src/app/pipeline/pipeline.hpp` | Added `ConfigureSurgicalPhaseOperator()` and `surgical_phase_op_` member |
| `src/app/pipeline/pipeline.cpp` | Added operator configuration and `add_flow` wiring |
| `src/operators/periodic_counter/periodic_counter.hpp` | Added `condition_variable` for interruptible stop |
| `src/operators/periodic_counter/periodic_counter.cpp` | Replaced `sleep_for` with `wait_for` |
| `src/operators/yuan_capture_source/yuan_capture_source.cpp` | Added `cv::resize` fallback for resolution mismatch |
| `CMakeLists.txt` | Added surgical_phase source, include dir, and TensorRT link libraries |
| `configs/app.yaml` | Added `PhaseModelPath` |

---

## Docker Build & Run

```bash
# Build
docker build --network=host -f docker/Dockerfile -t isurgical-app:latest .

# Run
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

---

## Quick Reference

| Parameter | Default | Location |
|-----------|---------|----------|
| Phase model path | `/workspace/iSurgical/data/models/surgical_phase_model.onnx` | `configs/app.yaml` |
| Inference interval | 60000 ms (set to 10000 for testing) | `pipeline.cpp` |
| Sequence length | 5 frames (1 fps sampling) | `pipeline.cpp` |
| Engine cache | `{model_path}.engine` | auto-generated beside ONNX file |
