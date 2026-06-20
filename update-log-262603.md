# Update Log — 2626-06-03

Scope: bring the **Laparoscopic Cholecystectomy (LC)** segmentation path online end‑to‑end,
fix the build, and polish the overlay rendering. Touches the build system, the inference
pipeline, the LC post‑processor (C++/CUDA), the app config, and the model export.

---

## 1. Build fixes

### 1.1 CMake source path mismatch
`CMakeLists.txt` referenced a non‑existent directory `src/operators/lc_seg_postprocessor_gpu/`,
while the real directory is `src/operators/lc_seg_postprocessor/`. CMake failed with
*"Cannot find source file … / No SOURCES given to target: iSurgical"*.

- Fixed the two source entries (`lc_seg_postprocessor_gpu.cpp`, `lc_cuda_postprocess.cu`)
  and the `target_include_directories` entry to point at `lc_seg_postprocessor/`.

### 1.2 `pipeline.cpp` compile errors
`ConfigurePostProcessorOperator()` failed to compile:

- **`switch` on a `std::string`** ("switch quantity not an integer") — C++ `switch` only
  accepts integral types. Replaced with an `if / else if / else` chain. This also removed a
  latent fall‑through bug (the `Urology` case was missing a `break` and would have thrown
  in the `default` even on success).
- **`LcSegPostprocesserGPUOp was not declared`** — the operator header was never included.
  Added `#include "./operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.hpp"`.

---

## 2. LC segmentation produced **no / incorrect** masks — root cause and fix

The LC path was broken at every layer. Findings and fixes:

| # | Problem | Fix |
|---|---------|-----|
| 1 | `data/models/lc_seg_model.onnx` was actually a **phase‑classification** model (outputs `features`+`phase_logits`), not segmentation. | Installed the real segmentation model exported from `LC_SKH_2618.pth`. |
| 2 | LC config had **no `head_groups`** → `num_heads_ = 0` → render loop never ran (zero masks). | Added `head_groups` to `configs/app.yaml`. |
| 3 | `inference_map` was **hardcoded YOLO‑style** (`outputs`,`proto`); the real model has 5 outputs. | Made `inference_map` procedure‑specific. |
| 4 | CUDA decode used **per‑channel sigmoid + threshold** with an **off‑by‑one** channel→class map (background at the wrong end). | Rewrote the kernel to **argmax** decode with channel 0 = background, matching the reference `inference_video.py`. |
| 5 | Config said **224**, model is **512**; head logits are at inference resolution, not overlay resolution. | Set resize to 512 and added overlay→inference coordinate scaling in the kernel. |
| 6 | Model was **NCHW & un‑normalized**, but this Holoscan pipeline feeds **NHWC, [0,1]** tensors (InferenceOp does no transpose/normalize). | Re‑exported the ONNX with a deploy wrapper that takes NHWC[0,1] and bakes ImageNet normalize + NHWC→NCHW transpose. |

### Model architecture (reference)
`MultiHeadGCNFormer` (SegFormer + GCN refinement), `src/operators/lc_seg_postprocessor/model.py`.
Four prediction heads, each a `Conv2d(embed_dim, len(group)+1, 1)` softmax classifier
interpolated to input size. Head/class layout:

```
HEAD_GROUPS = [
  [1, 2, 3, 13],          # Gallbladder, Cystic duct, Cystic artery, Common bile duct
  [4, 15],                # Liver, Rouvieres sulcus
  [5, 12, 14],            # Hepatoduodenal ligament, Duodenum, Fat
  [6, 7, 8, 9, 10, 11],   # bipolar, clipper, grasper, hook, irrigator, scissors
]
```
Per head: channel 0 = background; channels 1..N map to the group's global class ids.

### ONNX re‑export (deploy build)
Added `convert_to_onnx_deploy.py` (in the training repo). `DeployWrapper`:
`NHWC [0,1] → (x-mean)/std → permute to NCHW → model → (head_0..head_3, correlation_scores)`.
Input becomes `image [batch, 512, 512, 3]`. Verified numerically equivalent to the reference
NCHW+normalize path (max abs diff ≈ 1e‑5). Installed as `data/models/lc_seg_model.onnx`;
deleted the stale `*.engine.fp16` so TensorRT rebuilds.

---

## 3. Functions added / changed

### `RenderLcHeadsToOverlayGPU` + `render_lc_heads_kernel` (`lc_cuda_postprocess.cu`)
Rewritten. New signature passes overlay dims **and** inference dims:
```c
void RenderLcHeadsToOverlayGPU(
    const float* d_head_logits, const int* d_channel_class_map, int num_channels,
    float* d_overlay, int out_w, int out_h, int infer_w, int infer_h, cudaStream_t stream);
```
Per overlay pixel: nearest‑neighbour map down to the head's inference grid, **argmax over
channels**, skip if background wins (leaves earlier heads intact), else paint the class
colour. Heads render sequentially → later heads overwrite earlier ones, matching the
reference's sequential `pred_mask` assignment.

### `LcSegPostprocesserGPUOp::compute` (`lc_seg_postprocessor_gpu.cpp`)
- Builds the correct `channel_class_map` (index 0 = background, `c+1` = `group[c]`).
- Threads `inference_resize_width_/height_` (512) into the render call.
- **Reordered** rendering so masks draw first and the legend is composited last.

### `LcSegPostprocesserGPUOp::DrawLegendOnOverlay` → returns `cv::Rect`
Now returns the frame‑clamped legend panel rectangle so the caller can blit just that region.

### `InferencePipeline::ConfigureInferenceOperator` (`pipeline.cpp`)
`inference_map` is procedure‑specific:
- LC: `head_out_0, head_out_1, head_out_2, head_out_3, corr_scores`
- Urology: `outputs, proto`

### `InferencePipeline::ConfigurePostProcessorOperator` (`pipeline.cpp`)
`switch` → `if/else`, correct operator selection per procedure, missing‑break bug removed.

---

## 4. Overlay polish

### 4.1 Legend always on top
The legend was being painted **before** the masks, so masks covered it. Reordered
`compute()` to: clear → render masks → draw legend → **blit only the legend's panel rect**
on top via `cudaMemcpy2DAsync` (a full‑frame copy would have wiped the masks elsewhere).
Both operations share `cuda_stream`, guaranteeing the legend lands last and is never covered.

### 4.2 Segmentation alpha
Mask blending alpha lowered **0.45 → 0.2** in `render_lc_heads_kernel` (more transparent
masks; legend swatches/text keep alpha 1.0).

---

## 5. Files touched

| File | Change |
|------|--------|
| `CMakeLists.txt` | Corrected LC operator source/include paths. |
| `src/app/pipeline/pipeline.cpp` | `switch`→`if/else`, include, procedure‑specific `inference_map`. |
| `configs/app.yaml` | `head_groups`, `InferenceResize` 224→512 for LC. |
| `src/operators/lc_seg_postprocessor/lc_cuda_postprocess.cu` | Argmax kernel rewrite, coord scaling, alpha 0.2. |
| `src/operators/lc_seg_postprocessor/lc_cuda_postprocess_api.h` | New `RenderLcHeadsToOverlayGPU` signature. |
| `src/operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.hpp` | Inference‑size members, `DrawLegendOnOverlay` returns `cv::Rect`. |
| `src/operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.cpp` | Channel map fix, render reorder, legend‑on‑top blit. |
| `data/models/lc_seg_model.onnx` | Replaced with the real NHWC deploy segmentation model; stale TRT engine removed. |
| `convert_to_onnx_deploy.py` *(training repo)* | New NHWC + baked‑preprocessing export script. |

---

## 6. Key constraint (for future models)

This Holoscan pipeline feeds models the raw `FormatConverterOp` output: **NHWC**, float32,
**[0,1]** range. `InferenceOp` does **not** transpose or normalize. Any new model must be
exported to accept NHWC[0,1] with all preprocessing baked into the graph (as Urology already
is, and as LC now is). After swapping an `.onnx`, delete the cached `*.engine.fp16` so
TensorRT rebuilds.

---

## 7. Build / verify

1. Rebuild in Docker.
2. First run rebuilds the LC TensorRT engine (slower startup).
3. Confirm visually: LC anatomy/tool masks appear, aligned, at ~0.2 opacity, with the legend
   card on top and never occluded.
4. If mask colours look swapped, reconcile RGB vs BGR between `kclass_colours` (RGB) and the
   reference's BGR tuples.
