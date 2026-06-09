# iSurgical Pipeline - Update Log

**Date:** 2026-06-09  
**Branch:** `main`  
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0  
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

This session addressed multiple improvements:
1. **Shutdown hang fix** — `SurgicalPhaseOp` now shuts down cleanly when Holoviz is closed
2. **Phase output consolidation** — Inference output reduced from per-frame predictions to a single majority-voted phase per cycle
3. **gRPC client operator** — New `PhaseGrpcClientOp` sends phase recognition results to an external server
4. **Surgery Dashboard** — New PySide6 Qt application (`ior/dashboard/`) that receives phase updates via gRPC and displays a real-time surgery monitoring dashboard

---

## Issue 1: SurgicalPhaseOp Cannot Stop When Holoviz Closes

**Phenomenon:** Closing the Holoviz window should terminate the entire pipeline, but the application hangs indefinitely. The `SurgicalPhaseOp` never completes its `stop()` method.

**Root Cause (multi-part):**

1. **`ConditionType::kNone` on input port** — The input was configured with:
   ```cpp
   spec.input<holoscan::gxf::Entity>("frame_input")
       .condition(holoscan::ConditionType::kNone);
   ```
   This tells the GXF scheduler "this operator is always ready to execute, regardless of input availability." Even after the source stops and Holoviz closes, the scheduler keeps ticking this operator and cannot declare the graph fully stopped.

2. **`std::async` timeout with detach** — The previous `stop()` used `std::async` to join the worker thread with a 3-second timeout. If the thread was blocked in `buildEngine()`, it was detached — but CUDA resources (`d_input_`, `d_output_`, `infer_stream_`) were freed while the detached thread might still be using them.

3. **No exit tracking** — No mechanism to know whether the worker thread had actually exited vs. being stuck in an uninterruptible call.

**Fix:**

1. **Removed `ConditionType::kNone`** — Default `kMessageAvailable` condition now applies. `compute()` only runs when a frame arrives from upstream. When the source stops, the scheduler recognizes this operator is idle and shuts down the graph cleanly.
   ```cpp
   spec.input<holoscan::gxf::Entity>("frame_input");  // default kMessageAvailable
   ```

2. **Added `worker_exited_` atomic flag** — The worker sets it `true` on every exit path, allowing `stop()` to know when it's safe to join.

3. **Revised `stop()` with poll-based join** — Polls `worker_exited_` for up to 5 seconds. If the worker exited (normal case), joins cleanly. Only detaches if truly stuck inside `buildEngine()`.
   ```cpp
   constexpr int kMaxWaitMs = 5000;
   constexpr int kPollMs = 50;
   int waited = 0;
   while (!worker_exited_.load() && waited < kMaxWaitMs) {
     std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
     waited += kPollMs;
   }
   if (worker_exited_.load()) {
     worker_thread_.join();
   } else {
     worker_thread_.detach();
   }
   ```

4. **Added `is_running_` checks around `buildEngine()`** — Worker checks the flag before and after engine build.

5. **Replaced busy-wait CUDA poll with `cudaStreamSynchronize()`** — Simpler and avoids tight sleep loop during inference.

**Key Lesson:**  
Never use `ConditionType::kNone` on an input port for operators that should follow the pipeline lifecycle. The default `kMessageAvailable` ensures the scheduler can properly determine when the operator is idle and allow graph shutdown.

---

## Issue 2: Multiple Phase Predictions Printed Per Inference Cycle

**Phenomenon:** Each inference cycle printed 5 separate phase results (one per sequence frame):
```
[info] [SurgicalPhase] Current phase: CalotTriangleDissection
[info] [SurgicalPhase] Current phase: Preparation
[info] [SurgicalPhase] Current phase: Preparation
[info] [SurgicalPhase] Current phase: CalotTriangleDissection
[info] [SurgicalPhase] Current phase: Preparation
```

**Root Cause:**  
The model outputs logits of shape `(1, seq_len, 7)` — a prediction for each of the 5 frames in the sequence. The code only used the last frame's prediction (`seq - 1`), but inference was being triggered multiple times across the session. The user expected a single consolidated prediction per cycle.

**Fix:**  
Replaced single-frame prediction with **majority voting** across all sequence steps:

```cpp
std::vector<int> votes(kNumClasses, 0);
for (int t = 0; t < seq; ++t) {
  float* logits = output_buffer.data() + t * kNumClasses;
  int pred = std::distance(logits, std::max_element(logits, logits + kNumClasses));
  votes[pred]++;
}
int best = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
HOLOSCAN_LOG_INFO("[SurgicalPhase] Current phase: {}", kPhaseNames[best]);
```

Now a single phase (the most common prediction across the 5-frame window) is reported per inference cycle.

---

## Feature 1: gRPC Client Operator (`PhaseGrpcClientOp`)

### Purpose

Send surgical phase recognition results from the Holoscan C++ pipeline to an external gRPC server (e.g., the Python dashboard) in real-time.

### Architecture

```
SurgicalPhaseOp (worker thread produces phase)
    │
    │ callback (thread-safe queue push)
    ▼
PhaseGrpcClientOp (sender thread drains queue → gRPC unary RPC)
    │
    ▼
gRPC Server @ localhost:50051
```

### New Files

| File | Purpose |
|------|---------|
| `protos/surgical_phase.proto` | Protobuf service definition with `SendPhase` and `StartSurgery` RPCs |
| `src/operators/phase_grpc_client/phase_grpc_client.hpp` | Operator header — gRPC channel, stub, sender thread, queue |
| `src/operators/phase_grpc_client/phase_grpc_client.cpp` | Implementation — async sender loop with 2s deadline per RPC |

### Proto Definition

```protobuf
service SurgicalPhaseService {
  rpc SendPhase (PhaseRequest) returns (PhaseResponse);
  rpc StartSurgery (SurgeryInfo) returns (PhaseResponse);
}

message PhaseRequest {
  string phase = 1;
  int64 timestamp_ms = 2;
}

message SurgeryInfo {
  string doctor_name = 1;
  string patient_id = 2;
  string procedure = 3;
  int64 start_time_ms = 4;
}
```

### Design Decisions

1. **Callback-based coupling** — `SurgicalPhaseOp` exposes a `setPhaseCallback()` method. The pipeline wires it to `PhaseGrpcClientOp::sendPhase()` via a lambda in `compose()`. No direct dependency between the two operator headers.

2. **Async sender thread** — gRPC calls happen on a dedicated thread, never blocking the pipeline. Phase strings are pushed to a `std::queue` protected by mutex + condition variable.

3. **Graceful shutdown** — `stop()` signals the condition variable; the sender drains any remaining items then exits. Thread is always joined (no detach).

4. **2-second RPC deadline** — Prevents the sender from blocking indefinitely if the server is down.

5. **`add_operator()` registration** — Since `PhaseGrpcClientOp` has no input/output port connections (it receives data via callback, not the graph), it is registered with `add_operator()` so Holoscan manages its lifecycle.

### Pipeline Wiring (in `compose()`)

```cpp
auto phase_op = std::dynamic_pointer_cast<SurgicalPhaseOp>(surgical_phase_op_);
auto grpc_op = std::dynamic_pointer_cast<PhaseGrpcClientOp>(phase_grpc_client_op_);
phase_op->setPhaseCallback([grpc_op](const std::string& phase) {
  grpc_op->sendPhase(phase);
});
add_operator(phase_grpc_client_op_);
```

### CMakeLists.txt Changes

- Added `find_package(Protobuf)` and `find_package(gRPC)`
- Added `add_custom_command()` to generate protobuf/gRPC C++ sources from `.proto`
- Created `surgical_phase_proto` static library target
- Added `phase_grpc_client.cpp` to `add_executable()`
- Added `${PROTO_SRC_DIR}` to include directories
- Linked `surgical_phase_proto` to the main executable

### Docker Requirements

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
```

---

## Feature 2: Surgery Dashboard (PySide6 + gRPC Server)

### Purpose

A real-time surgery monitoring dashboard that runs on the host (or any machine), receives phase updates from the iSurgical pipeline via gRPC, and displays:
- Doctor name, de-identified patient name, procedure type
- Total elapsed surgery time
- Per-phase time accumulation with visual highlighting of the current phase

### File Structure

```
ior/dashboard/
├── dashboard.py          # Main PySide6 GUI application + entry point
├── server.py             # gRPC server (embedded in the Qt app process)
├── test_client.py        # Standalone test client to simulate phase updates
├── generate_proto.sh     # Script to regenerate Python proto stubs
├── requirements.txt      # PySide6, grpcio, grpcio-tools, protobuf
└── generated/
    ├── __init__.py
    ├── surgical_phase_pb2.py
    └── surgical_phase_pb2_grpc.py
```

### Dashboard Features

| Feature | Description |
|---------|-------------|
| Dark theme | Professional surgical monitoring aesthetic |
| Info panel | Shows doctor name, de-identified patient (e.g., "J*** S***"), procedure, total elapsed time |
| Phase cards | 7 color-coded cards (one per LC phase), each showing accumulated time |
| Active phase highlight | Current phase card has colored border and bold text |
| Real-time clock | 1-second tick updates elapsed time and active phase duration |
| Patient de-identification | Names are masked: "John Smith" → "J*** S***" |

### Phase Color Mapping

| Phase | Color |
|-------|-------|
| Preparation | `#4FC3F7` (light blue) |
| Calot Triangle Dissection | `#FFB74D` (orange) |
| Clipping & Cutting | `#E57373` (red) |
| Gallbladder Dissection | `#81C784` (green) |
| Gallbladder Packaging | `#BA68C8` (purple) |
| Cleaning & Coagulation | `#FFD54F` (yellow) |
| Gallbladder Retraction | `#4DB6AC` (teal) |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   dashboard.py (Qt Main Thread)          │
│                                                         │
│  PhaseSignalBridge (QObject)                            │
│    ├── phase_received Signal ──→ SurgeryDashboard       │
│    └── surgery_started Signal ──→ SurgeryDashboard      │
│                                                         │
│  SurgeryDashboard (QMainWindow)                         │
│    ├── Info panel (doctor, patient, procedure, elapsed) │
│    ├── Current phase label                              │
│    └── 7 × PhaseCard widgets                            │
└─────────────────────────────────────────────────────────┘
         ▲
         │ Qt Signal (thread-safe)
         │
┌─────────────────────────────────────────────────────────┐
│             server.py (gRPC Thread Pool)                 │
│                                                         │
│  SurgicalPhaseServicer                                  │
│    ├── SendPhase() → emits phase_received signal        │
│    └── StartSurgery() → emits surgery_started signal    │
└─────────────────────────────────────────────────────────┘
         ▲
         │ gRPC (localhost:50051)
         │
┌─────────────────────────────────────────────────────────┐
│        iSurgical C++ Pipeline (PhaseGrpcClientOp)       │
│        OR test_client.py (for testing)                  │
└─────────────────────────────────────────────────────────┘
```

### How to Run

```bash
# Terminal 1 — Start the dashboard (includes gRPC server)
cd ior/dashboard
pip install -r requirements.txt
python dashboard.py

# Terminal 2 — Test with simulated phases
cd ior/dashboard
python test_client.py

# OR run the full iSurgical pipeline (sends real phases)
docker run ... isurgical-app:latest ./bin/iSurgical configs/app.yaml
```

### YAML Config Addition

```yaml
PhaseGrpcServer: "localhost:50051"
```

---

## Updated Pipeline Data Flow

```
                                                                               ┌──────────────┐
YuanCaptureSource ──┬──→ FormatConverterOp ──→ InferenceOp ──→ PostprocessGPU ─┬──→  HoloVizOp │
     (output)       │         (tensor)          (transmitter)       (out)      │   (receivers) │
                    │                                                           └──────────────┘
                    ├──→ RawFormatConverterOp ──→ HoloVizOp (receivers)
                    │
                    │                                              ┌─────────────────────────────┐
                    └──→ SurgicalPhaseOp ("frame_input")           │  background worker:          │
                              │                                    │  - build TRT engine           │
                              │                                    │  - every 7s: sample 5 frames  │
                              │                                    │  - infer → majority vote      │
                              │                                    │  - callback → gRPC client     │
                              │                                    └─────────────────────────────┘
                              │ phase_callback_
                              ▼
                         PhaseGrpcClientOp (add_operator, no ports)
                              │ sender thread → gRPC unary RPC
                              ▼
                    ┌─────────────────────────┐
                    │  Dashboard (PySide6)     │
                    │  gRPC server :50051      │
                    └─────────────────────────┘
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/operators/surgical_phase/surgical_phase.hpp` | Added `worker_exited_` atomic, `phase_callback_` function, `setPhaseCallback()` public method, `<functional>` include |
| `src/operators/surgical_phase/surgical_phase.cpp` | Removed `kNone` condition, revised `stop()` with poll-based join, added `worker_exited_` tracking on all exit paths, majority voting in `runInference()`, callback invocation |
| `src/operators/phase_grpc_client/phase_grpc_client.hpp` | New — gRPC client operator header |
| `src/operators/phase_grpc_client/phase_grpc_client.cpp` | New — async sender loop implementation |
| `src/app/pipeline/pipeline.hpp` | Added `ConfigurePhaseGrpcClientOperator()` and `phase_grpc_client_op_` member |
| `src/app/pipeline/pipeline.cpp` | Added gRPC client configuration, callback wiring, `add_operator()` |
| `protos/surgical_phase.proto` | New — service definition with `SendPhase` and `StartSurgery` RPCs |
| `configs/app.yaml` | Added `PhaseGrpcServer` setting |
| `CMakeLists.txt` | Added protobuf/gRPC find_package, proto codegen, `surgical_phase_proto` library, new source and include paths |
| `ior/dashboard/dashboard.py` | New — PySide6 surgery dashboard application |
| `ior/dashboard/server.py` | New — gRPC server with Qt signal bridge |
| `ior/dashboard/test_client.py` | New — test client for dashboard validation |
| `ior/dashboard/generate_proto.sh` | New — proto stub generation script |
| `ior/dashboard/requirements.txt` | New — Python dependencies |

---

## Quick Reference

| Parameter | Value | Location |
|-----------|-------|----------|
| Phase model path | `/workspace/iSurgical/data/models/surgical_phase_model.onnx` | `configs/app.yaml` |
| Inference interval | 7000 ms | `pipeline.cpp` |
| Sequence length | 5 frames (1 fps sampling) | `pipeline.cpp` |
| gRPC server address | `localhost:50051` | `configs/app.yaml` |
| RPC deadline | 2 seconds | `phase_grpc_client.cpp` |
| Dashboard port | 50051 | `ior/dashboard/dashboard.py` |
| Engine cache | `{model_path}.engine` | auto-generated beside ONNX file |

---

## Nsight Systems Profiling

To profile the application with Nsight Systems (installed on host via JetPack):

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
    -v $(pwd)/outputs:/workspace/iSurgical/data/outputs \
    -v $(pwd)/nsight_reports:/workspace/iSurgical/nsight_reports \
    -v /opt/nvidia/nsight-systems/2024.5.4:/opt/nvidia/nsight-systems/2024.5.4 \
    isurgical-app:latest \
    bash -c "/opt/nvidia/nsight-systems/2024.5.4/bin/nsys profile \
      --output /workspace/iSurgical/nsight_reports/isurgical_report \
      --trace cuda,nvtx,osrt \
      --force-overwrite true \
      ./bin/iSurgical configs/app.yaml"
```

The `nsys` binary is located at `/opt/nvidia/nsight-systems/2024.5.4/bin/nsys` on the Jetson host (installed with JetPack). Mount it into the container since the Holoscan container does not include it.
