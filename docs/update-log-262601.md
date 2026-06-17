# iSurgical Pipeline - Update Log

**Date:** 2026-06-17  
**Branch:** `main`  
**Platform:** NVIDIA Jetson Orin (aarch64) / Holoscan SDK 3.3.0  
**Container:** `nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu`

---

## Summary

A multi-procedure config refactor (`Procedure: "Urology" | "Laparoscopic Cholecystectomy"` with per-procedure settings in `configs/app.yaml`) had been started but was left mid-way: it didn't compile, and even after it compiled it crashed at runtime. This session finished that refactor, then iterated on the segmentation overlay's legend feature:

1. **Fixed the multi-procedure config refactor** — `configs/app.yaml` schema, `pipeline.cpp`/`.hpp`, `yolo_seg_postprocessor_gpu.cpp`/`.hpp`, `CUDA_Postprocess.cu`/`.h` — so the app builds and runs again
2. **Fixed a Holoscan parameter-timing bug** — values were read before the framework had bound them
3. **Fixed an oversized `cudaMemcpy2DAsync`** that was copying ~16x more rows than intended
4. **Added/fixed a class-color legend** on the overlay — corrected its position, made it static (always show all classes) instead of flickering with detections, and laid it out in two columns
5. **Removed bounding boxes and per-detection text labels** from the overlay — only segmentation masks + legend are now drawn

---

## Issue 1: `YAML::TypedBadConversion<unsigned int>` on Startup

**Phenomenon:**
```
[info] [fragment.cpp:765] Loading extensions from configs...
terminate called after throwing an instance of 'YAML::TypedBadConversion<unsigned int>'
  what():  bad conversion
```

**Root Cause (two layers):**

1. **Stale binary vs. new config** — `configs/app.yaml` had been restructured to move `Sys_Width`, `Sys_Height`, `NumClass`, `ScoreThreshold`, etc. out of the YAML root and into a per-procedure block, but the *currently built* `isurgical-app:latest` image still contained the *old* binary, which read those keys straight from the root (`config_["NumClass"].as<uint32_t>()`). A missing/null YAML node converted to `uint32_t` throws exactly this exception.
2. **The new C++ source didn't even compile** — see Issue 2. So simply rebuilding the image wasn't possible yet without further fixes.

**Key Lesson:**  
If you change a YAML config's *shape* (not just values), the currently-running container image is still running the *old* binary's expectations until you rebuild it. A config/binary mismatch produces this exact `YAML::TypedBadConversion` error — always rebuild (`docker buildx build ...`) after either side changes, and check `docker images <name> --format "{{.CreatedAt}}"` against your last source edit if you're not sure.

---

## Issue 2: New Procedure-Config Refactor Didn't Compile

**Phenomenon:** Building `src/app/pipeline/pipeline.cpp` and `src/operators/yolo_seg_postprocessor_gpu/*` failed outright.

**Root Causes:**

1. **`switch` on a `std::string`** — not legal C++:
   ```cpp
   switch(config_["Procedure"].as<std::string>()) {
   case "urology":   // also illegal — case labels must be integer constants
   ```
2. **Missing closing brace** on `ConfigurePostProcessorOperator()`, which silently swallowed the next function's opening brace.
3. **`ProcedureConfig` was a YAML sequence of single-key maps**, not a real map:
   ```yaml
   ProcedureConfig:
     - Laparoscopic Cholecystectomy:
       - Sys_Width: 1920    # <- each "- Key: value" line is its own one-entry map
   ```
   This can't be indexed as `config_["ProcedureConfig"]["Urology"]`.
4. **Header/implementation parameter mismatch** in `yolo_seg_postprocessor_gpu.hpp`/`.cpp` — `.cpp` registered a Holoscan parameter named `colour_labels_`, but the header declared a *different* member, `config_`:
   ```
   error: no match for 'operator[]' (operand types are
   'holoscan::Parameter<YAML::Node>' {aka 'holoscan::MetaParameter<YAML::Node>'}
   and 'const char [12]')
   ```
5. **`InitOverlayTensor()` signature mismatch** between header (5 args, including `width`/`height`) and the `.cpp` definition/call site (3 args, after `width`/`height` were removed during the refactor):
   ```
   error: no matching function for call to
   'YoloSegPostprocesserGPUOp::InitOverlayTensor(holoscan::ExecutionContext&,
   holoscan::Parameter<std::shared_ptr<holoscan::Allocator>>&, const char [14])'
   ```
6. **`UploadClassColorsToGPU` was hardcoded to `float[12][4]`** (`CUDA_Postprocess_API.h`), but the two procedures have different class counts (12 vs 16) and the YAML-derived data was now a `std::vector<std::vector<float>>`.

**Fix:**

- Replaced the `switch` with a single `procedure_config_` `YAML::Node` resolved once in the `InferencePipeline` constructor (`config_["ProcedureConfig"][procedure]`), used by every `Configure*Operator()` function instead of reading `Sys_Width`/`Sys_Height`/`NumBlocks`/`ModelPath`/etc. from the root.
- Rewrote `configs/app.yaml` so `ProcedureConfig` is a proper nested map (`Urology:` / `Laparoscopic Cholecystectomy:` as real keys with real key-value children), not a sequence of one-entry maps.
- Renamed the header member to `colour_labels_` to match what `.cpp` registers.
- Fixed `InitOverlayTensor()`'s declaration to match its 3-argument definition/call site (width/height now come from the resolved procedure config instead of being passed in).
- Changed `UploadClassColorsToGPU` to `(const float* colors, int num_classes)` against a `__constant__ float d_class_colors[32][4]` buffer (bumped from `[12][4]`), with the caller flattening/padding each class's color to 4 floats (defaulting missing alpha to `1.0f`) before upload.

**Key Lesson:**  
When restructuring a YAML config from a flat schema to a nested one, double-check whether each level is meant to be a **map** (`Key: value`) or a **list** (`- value`). A `- Key:` line nested under another `- Key:` line is *not* the same as `Key:` directly under another `Key:` — the former produces a sequence, which can't be indexed by name.

---

## Issue 3: `MetaParameter: value for 'colour_labels' is not set`

**Phenomenon:** After Issue 2 was fixed and the app compiled and ran, it crashed almost immediately:
```
[info] [fragment.cpp:765] Loading extensions from configs...
terminate called after throwing an instance of 'std::runtime_error'
  what():  MetaParameter: value for 'colour_labels' is not set
```

**Root Cause:**  
`YoloSegPostprocesserGPUOp::setup()` called `spec.param(this->colour_labels_, "colour_labels", ...)` and then *immediately* tried to read it with `colour_labels_.get()` in the same function. Holoscan only binds the actual argument values to `holoscan::Parameter<T>` members **after** `setup()` returns — reading `.get()` inside `setup()` itself always fails. This codebase's own `AsyncRecorderOp` follows the correct pattern: it registers parameters in `setup()` and only calls `.get()` later, in `start()`.

**Fix:**  
Added a `start()` override to `YoloSegPostprocesserGPUOp` (declared in the `.hpp`) and moved everything that depended on parameter values — reading `colour_labels_`, all the `cudaMalloc`/`cudaHostAlloc` buffer setup, and `UploadClassColorsToGPU()` — out of `setup()` and into `start()`. `setup()` now only declares ports and calls `spec.param(...)`.

**Key Lesson:**  
In a Holoscan `Operator`, never call `.get()` on a `holoscan::Parameter<T>` inside `setup()` — the framework hasn't bound argument values yet at that point. Do parameter-dependent initialization in `start()` instead. (Grep other operators in this repo for `.get()` to see the correct pattern before writing a new one.)

---

## Issue 4: Oversized `cudaMemcpy2DAsync` (Latent Memory-Safety Bug)

**Phenomenon:** No visible symptom yet (likely masked by the postprocessor's GPU memory pool block being oversized relative to what's actually needed), but caught during review of the overlay-copy code:

```cpp
cudaMemcpy2DAsync(
    this->d_overlay,
    this->width_ * this->num_blocks_ * sizeof(float),
    this->h_box_roi_,
    this->width_ * this->num_blocks_ * sizeof(float),
    this->width_ * this->num_blocks_ * sizeof(float),
    this->hight_ * this->num_blocks_ * sizeof(float),   // <- BUG: should just be this->hight_
    cudaMemcpyHostToDevice,
    this->cuda_stream
);
```

**Root Cause:**  
The `height` parameter of `cudaMemcpy2DAsync` is a **row count**, not a byte count. When the original literal `1080` was refactored into `this->hight_`, an extra `* this->num_blocks_ * sizeof(float)` was mistakenly carried over from the *pitch* parameters above it — inflating the row count ~16x (`num_blocks_ = 4`, `sizeof(float) = 4`). This reads/writes far past the intended buffer region on both the host (`h_box_roi_`) and device (`d_overlay`) sides.

**Fix:** Changed the height argument to just `this->hight_`.

**Key Lesson:**  
When refactoring literal pixel dimensions into named variables, re-derive each `cudaMemcpy2D*` argument from its *meaning* (pitch = bytes per row, width = bytes per row to copy, height = number of rows), not by pattern-matching the line above it.

---

## Issue 5: Legend Drawn in the Wrong Corner

**Phenomenon:** A `DrawLegendOnOverlay()` function was added to show a class-color legend, but it appeared at the top-left instead of the intended top-right.

**Root Cause:** Hardcoded position despite a comment claiming otherwise:
```cpp
// Position legend box at the top left corner
int start_x = 20;
int start_y = 20;
```

**Fix:**
```cpp
int margin = 20;
int start_x = this->width_ - legend_w - margin;
int start_y = margin;
```

---

## Issue 6: Legend Not Appearing At All (Even After the Position Fix)

**Phenomenon:** After fixing Issue 5 and rebuilding, the legend still didn't show — even though bounding boxes and per-box labels rendered correctly using the exact same detection data.

**Investigation:** Added a one-shot `HOLOSCAN_LOG_INFO` inside `DrawLegendOnOverlay()` and wrapped its call site in a `try`/`catch` logging any exception, to get hard evidence instead of guessing further. (This diagnostic step turned out to be moot — see Resolution.)

**Resolution:** The underlying design was the actual problem, not a bug: the legend only listed classes from the *current frame's kept detections* (`active_classes` built from `h_kept_boxes_`), so it would only appear when something was actively detected, and would change/disappear as detections changed — easy to mistake for "not working" if no detection happened to be on-screen at that moment. Per follow-up request, this was changed to a **static legend** that always lists every configured class (see Issue 7).

**Key Lesson:**  
Before assuming a "not appearing" UI element is a rendering bug, check whether its visibility is *conditional* on data that might not be present in the moment you're looking — add a log line to confirm what the function actually decided to do before chasing render-pipeline theories.

---

## Feature: Static, Two-Column Legend; Segmentation-Only Overlay

Per follow-up requests, the overlay was changed to:

1. **Static legend** — `DrawLegendOnOverlay()` now lists every class from `label_names_` (skipping index 0, `"Background"`) every frame, rather than only currently-detected classes:
   ```cpp
   std::vector<int> active_classes;
   for (int cls_id = 1; cls_id < static_cast<int>(this->label_names_.size()); ++cls_id) {
       active_classes.push_back(cls_id);
   }
   ```
2. **Two-column layout** — items are laid out left-to-right then top-to-bottom (`row = i / columns`, `col = i % columns`), roughly halving the panel's height for a given class count.
3. **No bounding boxes or per-detection text** — `DrawKeptBoxesOnOverlay()` no longer draws `cv::rectangle`/`cv::putText` per kept box; it only clears the CPU ROI buffer, draws the legend, and copies the result to the GPU overlay. Segmentation masks are unaffected — they're rendered separately on the GPU via `RenderMasksToOverlayGPU`.

---

## Files Modified

| File | Change |
|------|--------|
| `configs/app.yaml` | `ProcedureConfig` restructured from a sequence of single-key maps into a proper nested map per procedure; per-procedure `kclass_colours` normalized to 4 floats (RGBA) |
| `src/app/pipeline/pipeline.hpp` | Added `procedure_config_` member |
| `src/app/pipeline/pipeline.cpp` | Resolved `procedure_config_` once in the constructor; replaced the broken `switch`-on-`std::string` in `ConfigurePostProcessorOperator()`; every `Configure*Operator()` now reads `Sys_Width`/`Sys_Height`/`NumBlocks`/`ModelPath`/`InferenceFormat`/`InferenceResize*` from `procedure_config_` instead of the YAML root |
| `src/operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.hpp` | Renamed `config_` → `colour_labels_` parameter to match `.cpp`; added `start()` override; fixed `InitOverlayTensor()` signature; added `DrawLegendOnOverlay()` declaration |
| `src/operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.cpp` | Moved parameter-dependent init from `setup()` to `start()`; fixed YAML key casing (`ConfidenceThreshold`/`IoUThreshold`/`MaskThreshold`); fixed `label_names` → `label_names_` typo; fixed the oversized `cudaMemcpy2DAsync` height argument; added/fixed `DrawLegendOnOverlay()` (top-right position, static class list, two columns); removed bounding-box/text drawing from `DrawKeptBoxesOnOverlay()` |
| `src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess.cu` | `d_class_colors` bumped from `[12][4]` to `[MAX_CLASS_COLORS][4]`; `UploadClassColorsToGPU` now takes `(const float*, int num_classes)` |
| `src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess_API.h` | Added `MAX_CLASS_COLORS` constant; updated `UploadClassColorsToGPU` declaration |

---

## Quick Reference

| Symptom | Likely Cause | Where to Look |
|---------|--------------|----------------|
| `YAML::TypedBadConversion<unsigned int>` on startup | Config/binary schema mismatch — rebuild the image | `docker images isurgical-app --format "{{.CreatedAt}}"` vs. last source edit |
| `MetaParameter: value for '<name>' is not set` | A `holoscan::Parameter<T>::get()` call happened inside `setup()` instead of `start()` | Search the operator's `setup()` for `.get()` calls |
| Operator builds but overlay looks corrupted/garbage past a certain region | Check every `cudaMemcpy2D*` call's `height` argument is a **row count**, not bytes | `grep -n cudaMemcpy2D` in the operator's `.cpp` |
| New YAML nesting can't be indexed by key (`node["Foo"]` returns null) | The YAML level is a sequence (`- Foo:`) instead of a map (`Foo:`) | Re-check indentation/dashes in the `.yaml` file |
| New UI/overlay element "doesn't show up" | May be conditional on per-frame data that isn't always present | Add a one-shot log inside the function before assuming a render bug |

---

## Build/Verify Workflow Used This Session

Since the host's CMake (3.22) and Holoscan SDK (4.1.0) don't match what the project's `Dockerfile` builds against (CMake 3.30+, Holoscan 3.3.0), compilation was verified using the actual Docker builder stage rather than building on the host directly:

```bash
docker buildx build --allow network.host -f docker/Dockerfile --target builder -t isurgical-build-check .
```

`--allow network.host` is required because the `Dockerfile`'s `apt-get update` step uses `RUN --network=host`, which BuildKit blocks by default as an "insecure" entitlement.

To verify the full app (not just compilation), build and run the runtime image the same way the deployed container is run:
```bash
docker buildx build --allow network.host -f docker/Dockerfile -t isurgical-app:latest --load .
docker run --rm --runtime nvidia \
    -v $(pwd)/configs:/workspace/iSurgical/configs \
    -v $(pwd)/data:/workspace/iSurgical/data \
    isurgical-app:latest ./bin/iSurgical configs/app.yaml
```
