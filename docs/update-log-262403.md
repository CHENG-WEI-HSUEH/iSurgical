# Update Log - 2026-06-03

## Issue 1: Application cannot stop after closing Holoviz

**Phenomenon:** When the user closes the Holoviz window, the application hangs indefinitely and never exits. Must be force-killed.

**Root cause:** Worker threads in `AsyncRecorderOp` and `SurgicalPhaseOp` block their `stop()` methods:
- `AsyncRecorderOp::writerLoop()` drained all queued frames before exiting, and `VideoWriter::write()` could block indefinitely.
- `SurgicalPhaseOp::runInference()` called `cudaStreamSynchronize()` which blocks until GPU inference completes (potentially seconds). Additionally, `buildEngine()` (TensorRT engine compilation) is completely non-interruptible and can take minutes on first run.
- Both `stop()` methods called `thread_.join()` with no timeout, blocking the entire shutdown.

**Fix:**
- `AsyncRecorderOp::writerLoop()` now exits immediately when `is_running_` is false (drops remaining frames instead of draining).
- `SurgicalPhaseOp::runInference()` replaced `cudaStreamSynchronize()` with a polling loop (`cudaStreamQuery()` every 5ms) that checks `is_running_` and returns early on shutdown.
- Both `stop()` methods now use a timed join (2s for recorder, 3s for phase recognition). If the thread doesn't exit in time, it is detached so the application can proceed.
- Added early-return guard in `AsyncRecorderOp::compute()` when `!is_running_`.

---

## Issue 2: Recorded videos are 262 bytes (empty) and cannot be opened

**Phenomenon:** MP4 files (`raw.mp4`, `segmentation.mp4`) are created but only contain the file header (262 bytes). Zero frames are written. Log showed repeated warnings: `AsyncRecorderOp: no raw frame buffered yet, skipping`.

**Root cause (multi-part):**

1. **Tensor name mismatch:** The pipeline routes raw frames through `RecordFormatConverterOp` which renames the tensor from `"video"` to `"tensor"`. The recorder was looking for `"video"` and never finding it.

2. **kNone receive failure:** With `ConditionType::kNone` on `raw_input`, the original code used `input.receive<holoscan::TensorMap>("raw_input")` which returned nothing. Messages on kNone ports need to be drained in a loop since they are not guaranteed to be available on any single call.

3. **Scheduler deadlock:** The greedy scheduler logged `"Some entities are waiting for execution, but there are no periodic or async entities to get out of the deadlock"` and stopped the pipeline before any frames could flow. This happened because:
   - The `PeriodicCounterOp` (which provides the async heartbeat) was commented out.
   - Without an async entity, if the source's first `compute()` returns empty (capture thread hasn't grabbed a frame yet), the scheduler sees all entities waiting and gives up.

**Fix:**
- Re-enabled `add_operator(periodic_counter_op_)` in the pipeline to keep the greedy scheduler alive.
- Recorder now searches for both `"tensor"` and `"video"` keys in the raw TensorMap.
- Raw input receive uses a drain loop: `while (raw_msg) { ... raw_msg = input.receive(...); }` to consume all pending messages and keep only the latest frame.
- Removed the spammy warning log (it's expected at startup before the first raw frame arrives).

---

## Issue 3: Output path configuration

**Phenomenon:** User wanted recordings saved to `iSurgical/data/outputs` on the host.

**Resolution:** Config already set `RecordingPath: "/workspace/iSurgical/data/outputs"`. Docker mount `-v $(pwd)/outputs:/workspace/iSurgical/data/outputs` maps this correctly. Added `std::filesystem::create_directories(path)` in `start()` to ensure the directory exists.

---

## Files modified

| File | Changes |
|------|---------|
| `src/operators/async_recorder/async_recorder.cpp` | Rewrote compute() with drain loop, tensor name fallback, timed join in stop(), immediate exit in writerLoop() |
| `src/operators/async_recorder/async_recorder.hpp` | Restored raw_buffer_ members for kNone buffering pattern |
| `src/operators/surgical_phase/surgical_phase.cpp` | Replaced cudaStreamSynchronize with polling loop, timed join in stop() |
| `src/app/pipeline/pipeline.cpp` | Re-enabled periodic_counter_op_ |
