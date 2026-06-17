# iSurgical Pipeline - Update Log

**Date:** 2026-06-17
**Branch:** `main`
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

A surgical-phase-recognition refactor (`surgical_phase` → `lc_surgical_phase`, a `TemporalPhasePostprocessorOp` with sliding-window smoothing) had been started but not finished: the app crashed on startup, and the user additionally wanted phase recognition to run asynchronously — sampling 5x/second with a time window, decoupled from the live video rate, and to log the highest-probability phase once a minute, all without ever stalling real-time segmentation. This session:

1. **Fixed the startup crash** — an `InferenceOp` output-tensor mismatch for the phase model
2. **Fixed a latent "entity will never tick" scheduling bug** on the phase postprocessor's unused output port
3. **Added `PhaseFrameSamplerOp`** — decouples phase recognition's sample rate from the live video frame rate
4. **Added per-minute highest-probability phase reporting** to `TemporalPhasePostprocessorOp`
5. **Attempted true OS-thread parallelism** (`EventBasedScheduler` + dedicated `ThreadPool`) for the phase branch, which surfaced three further bugs in sequence, and was ultimately reverted in favor of a purely data-flow-based decoupling that achieves the same non-blocking guarantee without the crashes
6. Spent significant time distinguishing **real hangs from testing artifacts** (an unconsumed `appsink` in a `gst-launch-1.0` probe, and device contention between concurrently-running test containers) before settling on the final architecture

---

## Issue 1: Startup Crash — `InferenceOp` Output Tensor Size Mismatch

**Phenomenon:**
```
[error] [infer_manager.cpp:368] Size mismatch. Out tensor names: 1, Output node size: 2
[error] [holoinfer_constants.hpp:83] Output tensor size not equal to output nodes in model.
terminate called after throwing an instance of 'std::runtime_error'
  what():  Error in Inference Operator, Sub-module->Start, Parameters setup, Output tensor size not equal to output nodes in model.
```

**Root Cause:** `lc_phase_model.onnx` has **two** ONNX output nodes (`features`, a 2048-dim embedding, and `phase_logits`, 13 classes), confirmed via:
```python
import onnx
m = onnx.load("data/models/lc_phase_model.onnx")
# Outputs: features ['batch_size', 2048], phase_logits ['batch_size', 13]
```
but `ConfigurePhaseInferenceOperator()` in `pipeline.cpp` only declared one output tensor name (`phase_logits`) in the `InferenceOp::DataVecMap inference_map`. Holoscan's `InferenceOp` requires every model output to be listed, even ones that go unused downstream.

**Fix:** Added `"features"` to the inference map alongside `"phase_logits"`. The postprocessor only reads `phase_logits` from the resulting `TensorMap`; the extra `features` entry is simply ignored.

**Key Lesson:** When an `InferenceOp::Start` fails with "Output tensor size not equal to output nodes in model", inspect the ONNX graph's actual output list (`onnx.load(...).graph.output`) rather than assuming the inference map matches what the model was trained to emit — embedding/auxiliary outputs are easy to miss.

---

## Issue 2: "Entity Will Never Tick" Warning on the Phase Postprocessor

**Phenomenon:**
```
[error] [entity_executor.cpp:353] No receiver connected to transmitter of DownstreamReceptiveSchedulingTerm 94 of entity "SurgicalPhasePostprocessor". The entity will never tick.
```

**Root Cause:** `TemporalPhasePostprocessorOp` declares a `smoothed_phase` output port and calls `output.emit(...)` on it, but nothing in the graph connects to it — the gRPC handoff is done via a native C++ callback (`setPhaseCallback`), not a Holoscan port. The port's default `DownstreamMessageAffordableCondition` requires *some* connected, ready receiver before the operator is allowed to tick at all, which an unconnected port can never satisfy.

**Fix:** `spec.output<std::string>("smoothed_phase").condition(holoscan::ConditionType::kNone);` — disables the backpressure condition on that specific (intentionally unconsumed) port.

**Key Lesson:** Any Holoscan output port you emit on for diagnostic/optional purposes, without wiring it into the graph via `add_flow`, needs `condition(ConditionType::kNone)` or the operator can stop ticking entirely — this is a "the entity will never tick" warning, not a benign one.

---

## Feature: `PhaseFrameSamplerOp` — Rate-Decoupled Phase Recognition

**Goal:** Run phase recognition at a fixed ~5 Hz, independent of the live video frame rate (30–60 fps), without ever blocking frame capture for segmentation/recording.

**New operator** (`src/operators/phase_frame_sampler/`): a thin passthrough with:
- Input port using `IOSpec::QueuePolicy::kPop` (capacity 1) — a new frame silently replaces the previously buffered one instead of queuing or blocking.
- A `holoscan::PeriodicCondition` (period from `PhaseSampleRateHz` in `app.yaml`, default 5) gates how often it actually ticks and forwards a frame downstream.

Net effect: between ticks, incoming frames are simply dropped; on each tick, the most recent frame is forwarded.

**Wiring it without stalling the source:** Initially this sampler read from `YuanCaptureSourceOp`'s existing shared `"output"` port (also used by segmentation/raw-preview/recording). That port has a default backpressure condition shared across *all* its fan-out edges — if the slow 5 Hz consumer's queue filled up between ticks, the condition would block the source operator from ticking *at all*, stalling every branch, not just the phase one (see Holoscan's own `multi_branch_pipeline` example for the documented pattern). Fixed by giving `YuanCaptureSourceOp` a **second, dedicated output port**, `"phase_output"`, with backpressure explicitly disabled (`condition(ConditionType::kNone)`), while the original `"output"` port keeps its default backpressure for the branches that should still pace naturally. `compute()` emits the same frame entity on both ports (cheap — just two more references to the same GPU buffer).

**Key Lesson:** A condition set on a Holoscan output port applies to *all* edges fanned out from that port, not per-edge. If different consumers of the same source need different backpressure/rate behavior, give them separate named output ports rather than trying to special-case one edge.

---

## Feature: Per-Minute Highest-Probability Phase Report

`TemporalPhasePostprocessorOp` already smoothed phase logits over a short sliding window (`SequenceLength` ticks) for live gRPC updates. Added, independently:
- A running sum of per-tick softmax probabilities (`minute_prob_sum_`) and a tick counter, reset on a wall-clock timer (`std::chrono::steady_clock`).
- Every `ReportIntervalSec` (new parameter, default 60, configurable via `PhaseReportIntervalSec` in `app.yaml`), logs the phase with the highest accumulated average probability and its confidence, then resets the accumulator.

This is intentionally decoupled from the short-window smoothing — one drives the live per-tick gRPC push, the other is a once-a-minute summary log, on independent timescales.

---

## Issue 3: Regression — `Push failed on 'source_video'` Warning Spam

**Phenomenon (after an earlier, since-reverted attempt to disable backpressure globally):**
```
[warning] [double_buffer_receiver.cpp:80] Push failed on 'source_video'
[warning] [double_buffer_receiver.cpp:80] Push failed on 'source_video'
... (repeated, bursty)
```

**Root Cause:** Before introducing the dedicated `"phase_output"` port (see above), backpressure was disabled on the *single shared* `"output"` port. That port also feeds the segmentation, raw-preview, and recording branches, all consumed by stock `holoscan::ops::FormatConverterOp` instances using the SDK default queue policy (`kFault`: log a warning and reject on overflow). With backpressure removed, the source pushed to every consumer every tick regardless of readiness — so whenever segmentation/recording briefly fell behind, those pushes failed and got logged, and frames were silently dropped for branches that previously paced correctly via natural backpressure.

**Fix:** Superseded by the `"phase_output"` port split above — `"output"` keeps its default backpressure (segmentation/raw/recording behave exactly as before any of this work), only `"phase_output"` bypasses it.

**Key Lesson:** Don't disable backpressure broadly to fix one slow consumer; scope the fix to that consumer's own edge/port.

---

## Issue 4: Crash — `CudaStreamPool` Exhaustion

**Phenomenon:**
```
[error] [cuda_stream_pool.cpp:125] CudaStreamPool reached capacity(5), could not allocate more streams
[error] [cuda_stream_pool.cpp:192] allocate cudastream failed.
[error] [holoinfer_constants.hpp:83] Inference manager, Preprocessed data for tensor raw_video does not exist.
[error] [holoinfer_constants.hpp:83] Inference manager, Inference failed in execution for segmentation inference
... terminate called after throwing an instance of 'std::runtime_error'
```

**Root Cause:** All operators shared one `CudaStreamPool` (`cuda_stream_pool_`, `max_size=5`), sized for the original single-threaded `GreedyScheduler` where stream usage was naturally serialized. At this point in the session, the phase branch had been pinned to its own dedicated `ThreadPool` (via `EventBasedScheduler`) to give it a genuinely separate OS thread from segmentation. With true concurrency, both branches could request streams from the same fixed-size pool *at the same instant*, exceeding capacity 5 — causing the segmentation preprocessor's stream allocation to fail, which cascaded into a missing `raw_video` tensor and a fatal inference failure.

**Fix:** Added a second, separate `CudaStreamPool` (`phase_cuda_stream_pool_`, `max_size=4`) used only by the phase branch's `FormatConverterOp` and `InferenceOp`, so the two branches can no longer contend for the same pool.

**Key Lesson:** A shared, fixed-capacity GPU resource (stream pool, memory pool, etc.) that was "big enough" under single-threaded/serialized execution can be silently undersized the moment any part of the graph starts running on genuinely separate threads. Give concurrently-executing subgraphs their own resource instances.

---

## Issue 5: Crash — "Unknown Exception" From Concurrent TensorRT Engines

**Phenomenon:** Even after Issue 4's fix, the very first `compute()` tick of `phase_inference` (right as both TensorRT engines became active at the same time) crashed:
```
[error] [gxf_wrapper.cpp:118] Exception occurred for operator: 'phase_inference' - Error in Inference Operator, Sub-module->Compute, unknown exception
terminate called after throwing an instance of 'std::runtime_error'
  what():  Error in Inference Operator, Sub-module->Compute, unknown exception
```

**Root Cause:** Running the segmentation and phase TensorRT engines truly concurrently, on separate OS threads (via the dedicated `ThreadPool` + `EventBasedScheduler`), is not reliably safe on this Jetson/Holoscan/TensorRT combination — the underlying inference stack threw a non-`std::exception` error with no further detail, right at the point both engines first executed simultaneously.

**Fix:** Reverted the true-OS-thread-parallelism attempt entirely: removed the `ThreadPool` creation/pinning and the `EventBasedScheduler` configuration, back to the default single-threaded `GreedyScheduler`. The phase branch's non-blocking guarantee is instead enforced purely through data flow (Issue/Feature above: `PhaseFrameSamplerOp` + dedicated `"phase_output"` port + `kPop` queue) — with one scheduler thread, segmentation and phase recognition simply interleave cooperatively rather than truly running in parallel, but the phase branch's occasional, lightweight tick (5 Hz, on a small 224×224 model) adds negligible overhead and — critically — can never block frame *capture*, which was the actual requirement.

**Key Lesson:** "Make X asynchronous so it doesn't block Y" doesn't have to mean "put X on its own OS thread." Data-flow decoupling (rate-limiting + non-blocking queues/ports) can satisfy the same requirement with far less risk, especially when X and Y both drive GPU inference engines that may not tolerate true concurrent execution.

---

## Diagnostic Detours (No Code Change, But Worth Recording)

1. **Suspected camera-open hang that wasn't real load-bearing evidence:** An early test run sat silent for 6+ minutes after `AsyncRecorderOp` started, with no `YuanCaptureSourceOp started` log. Bisecting the exact `v4l2src ! queue ! videoconvert ! appsink` GStreamer pipeline via `gst-launch-1.0` seemed to confirm a hang — but it was a methodology artifact: `gst-launch-1.0` never pulls samples from `appsink`, and `appsink` (by default, `wait-on-eos=true`) won't post an `EOS` bus message until something does, so the pipeline looked stuck even when working correctly. The actual application uses OpenCV's `cv::VideoCapture` with the `CAP_GSTREAMER` backend, which *does* continuously pull samples — a fundamentally different consumption pattern that the `gst-launch-1.0` probe didn't reproduce.
2. **Device contention between concurrent test runs:** Separately, a background test container and an interactive run (started independently in another terminal) both touched `/dev/video0` around the same time, producing confusing, inconsistent results (clean-but-fast exits, apparent hangs) that had nothing to do with the code changes themselves.

**Key Lesson:** When a real-time capture pipeline behaves inconsistently across test runs, first rule out (a) test tooling that doesn't exercise the code path the same way the real app does, and (b) multiple processes/containers competing for the same physical device, before debugging the application logic itself.

---

## Known Open Item (Not Yet Addressed)

Closing the Holoviz window / sending `Ctrl+C` triggers:
```
[error] [gxf_executor.cpp:2432] Application did not shut down within 3 seconds of interrupt. Forcing exit...
```
The app does still exit, just via GXF's forced path after its 3-second grace period elapses, rather than a clean shutdown. Not yet root-caused or fixed — flagged to the user as a possible follow-up (candidates to investigate: the `PhaseGrpcClient` sender thread's per-RPC deadline interacting with shutdown, or general teardown coordination overhead of the additional worker threads added throughout this session).

---

## Files Modified

| File | Change |
|------|--------|
| `configs/app.yaml` | Added `PhaseSampleRateHz` (default 5) and `PhaseReportIntervalSec` (default 60) |
| `src/app/pipeline/pipeline.hpp` | Added `phase_frame_sampler_op_`, `phase_pre_processor_op_`, `phase_inference_op_` members; added `phase_cuda_stream_pool_`; added `Configure{PhaseFrameSampler,PhasePreProcessor,PhaseInference}Operator()` declarations |
| `src/app/pipeline/pipeline.cpp` | Fixed phase `inference_map` to include both ONNX outputs; added `ConfigurePhaseFrameSamplerOperator()`; added a second `CudaStreamPool` for the phase branch; rewired the phase branch through `PhaseFrameSamplerOp` reading from `"phase_output"`; passed `ReportIntervalSec` into `TemporalPhasePostprocessorOp`; removed the `EventBasedScheduler`/`ThreadPool` pinning that was added and then reverted this session |
| `src/operators/yuan_capture_source/yuan_capture_source.cpp` | Added a second output port, `"phase_output"`, with backpressure disabled; `"output"` keeps default backpressure; `compute()` emits the same frame entity on both ports |
| `src/operators/lc_surgical_phase/lc_surgical_phase.hpp` / `.cpp` | Disabled the default condition on the unused `smoothed_phase` output port; added `ReportIntervalSec` parameter and the per-minute highest-probability accumulation/log logic |
| `CMakeLists.txt` | Added `src/operators/phase_frame_sampler/phase_frame_sampler.cpp` to the build and its include directory |

## Files Added

| File | Purpose |
|------|---------|
| `src/operators/phase_frame_sampler/phase_frame_sampler.hpp` / `.cpp` | New operator: rate-limits and drops frames for the phase branch via `PeriodicCondition` + `kPop` queue policy, decoupling its sample rate from the live video rate without blocking the source |

---

## Quick Reference

| Symptom | Likely Cause | Where to Look |
|---------|--------------|----------------|
| `InferenceOp` "Output tensor size not equal to output nodes in model" | `inference_map` is missing one of the ONNX model's actual output nodes | `onnx.load(model_path).graph.output` vs. the `DataVecMap` in `pipeline.cpp` |
| "No receiver connected to transmitter... entity will never tick" | An output port is emitted on but never wired via `add_flow` | Add `.condition(ConditionType::kNone)` to that port in `setup()` |
| `Push failed on '<port>'` warning spam | Backpressure disabled on a port shared by multiple consumers with different rates | Give the rate-mismatched consumer its own dedicated output port instead |
| `CudaStreamPool reached capacity(N)` | A GPU resource pool sized for serialized/single-threaded execution is now used by genuinely concurrent operators | Give concurrently-running subgraphs separate resource instances |
| "unknown exception" inside `InferenceOp::Compute` right as two engines first run together | Two TensorRT engines executing concurrently on separate OS threads | Avoid true thread-level parallelism between inference engines; decouple via data flow (rate limiting + non-blocking queues) instead |
| Capture pipeline looks "hung" in a `gst-launch-1.0` probe but works in the real app | `appsink` doesn't post EOS until something pulls samples; `gst-launch-1.0` never does | Test with the actual consumption pattern (e.g. `cv::VideoCapture` + a `read()` loop), not a generic CLI probe |

---

## Build/Verify Workflow Used This Session

Same constraint as prior sessions: the host's CMake/Holoscan don't match what `docker/Dockerfile` builds against, so every change was verified by rebuilding and running the actual container image:

```bash
docker build --network=host -f docker/Dockerfile -t isurgical-app:latest .

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
    -v $(pwd)/outputs:/workspace/iSurgical/data/outputs \
    isurgical-app:latest \
    ./bin/iSurgical configs/app.yaml
```

Only one such container (or one interactive run) should touch `/dev/video0` at a time — see the device-contention detour above.
