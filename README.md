# iprovision

Real-time surgical scene segmentation with `lc_seg_model.onnx` (ONNX Runtime + OpenCV, C++17).
Frames come from a camera or an mp4 replayed in real time. Each frame is segmented, and the
result is printed, drawn over the frame, and optionally shown in a window or recorded.

## Requirements

| Component | Version used | Notes |
|---|---|---|
| GPU | GTX 1080 Ti (Pascal, sm_61) | |
| ONNX Runtime | `onnxruntime-linux-x64-gpu-1.20.1` | CUDA 12 build; 1.30 needs CUDA 13, which does not support Pascal |
| cuDNN | 9.5.1 | **Must be a Pascal-capable cuDNN.** The system cuDNN 9.26 fails on this GPU (`CUDNN_STATUS_EXECUTION_FAILED_CUDART`) |
| CUDA runtime | 12.x (system `libcudart.so.12`, `libcublas.so.12`) | |
| OpenCV | 4.6 (system) | |
| yaml-cpp | 0.8 (system, `libyaml-cpp-dev`) | reads `configs/config.yaml` |
| CMake | ≥ 3.20 | |

## Build

First build (configures CMake, then compiles):

```bash
cd /media/andyhsueh/Storage2/smartsurgery/isurgical
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT_DIR=/media/andyhsueh/Storage2/sideproject/Onnx/onnxruntime-linux-x64-gpu-1.20.1 \
      -DCUDNN_LIB_DIR=/media/andyhsueh/Storage2/sideproject/Onnx/cudnn-9.5.1/lib
cmake --build build -j$(nproc)
```

- `CUDNN_LIB_DIR` accepts either the cuDNN root or its `lib/` folder.
- The ONNX Runtime and cuDNN library paths are embedded in the binary (RPATH), so no `LD_LIBRARY_PATH` is needed.
- If `CUDNN_LIB_DIR` is empty or wrong, CMake warns. The system cuDNN is then loaded, GPU inference fails, and the program falls back to CPU.
- CMake caches these options. After the first configure, rebuild after any code change with:

  ```bash
  cmake --build build -j$(nproc)
  ```
- To start from scratch, delete `build/` and run both commands again.
- The executable is `build/iprovision`.

## Run

Run from the project root so the default model, config and input paths resolve.

```bash
cd /media/andyhsueh/Storage2/smartsurgery/isurgical

# Defaults: data/inputs/video01.mp4, configs/config.yaml (laparoscopic cholecystectomy), GPU, console output only
./build/iprovision

# Show the result full screen and record it (q / Esc to quit)
./build/iprovision --input data/inputs/video01.mp4 --output data/outputs/output.mp4 --show

# Live camera 0 in a resizable window
./build/iprovision --input 0 --show --windowed

# Segmentation only, on CPU
./build/iprovision --cpu --no-rsd

# List all options
./build/iprovision --help
```

`--output` overwrites an existing file. Its directory must already exist.

| Option | Default | Description |
|---|---|---|
| `--model <path>` | `data/models/lc_seg_model.onnx` | ONNX model |
| `--config <path>` | `configs/config.yaml` | class names, colours and thresholds, see [Config](#config-configsconfigyaml) |
| `--procedure <name>` | `laparoscopic cholecystectomy` | which procedure section of the config to use |
| `--input <src>` | `data/inputs/video01.mp4` | camera index, video file, stream URL or image |
| `--output <path>` | none | write the visualization (video for streams, image for an image input) |
| `--head <0-3>` | all heads fused | draw only one head |
| `--cpu` | off | run on CPU instead of CUDA |
| `--letterbox` | off | keep aspect ratio when resizing to 512×512 (default stretches) |
| `--crop-border` | off | crop the black border around the endoscope view before resizing |
| `--show` | off | display the visualization full screen, aspect ratio kept (q / Esc to quit) |
| `--windowed` | off | with `--show`, use a resizable window instead of full screen |
| `--no-rsd` | off | disable phase / RSD inference |
| `--shm <name>` | none | publish the phase / class tables and per-frame status to shared memory created by a parent process (layout in `src/common/isurgical_shm.hpp`, reader: iprovision's `AppManager`). Setting its `stop_requested` flag ends the run cleanly |

Each processed frame prints the inference time, the 7 correlation scores, the regions found per
head (class name, area, mean confidence), and the latency from capture to display. At exit the program prints how many frames
inference skipped. It exits with a non-zero code if inference fails.

### Running from your own code

`src/main.cpp` only fills an `iSurgicalConfig` and calls `iSurgical::Run()`. The application is the
`iSurgical` class in `src/isurgical/`. Each command-line option above is a field of `iSurgicalConfig`, so
you can set the options in code instead:

```cpp
#include "isurgical/isurgical.hpp"

int main() {
    iSurgicalConfig config;                 // defaults as in the table above
    config.input = "0";                     // --input 0
    config.procedure = "laparoscopic cholecystectomy";
    config.show = true;                     // --show
    config.windowed = true;                 // --windowed
    config.output = "data/outputs/camera.mp4";

    iSurgical app(config);
    return app.Run();                       // blocks until the input ends or q / Esc; returns 0 or -1
}
```

Rebuild with `cmake --build build -j$(nproc)` after editing `main.cpp`.

### Real-time replay

Video files are replayed at their own frame rate, as if they came from a camera. When inference is
slower than the source, it always takes the newest frame and skips the rest. On the 1080 Ti,
inference takes about 63 ms per frame on `video01.mp4` (854×480), so about 2 of every 3 frames
of the 25 fps video are processed. On CPU it takes about 1.2 s per frame, so about 1 in 36 is processed.

The recording keeps the source timeline. Each annotated frame is held until the next result, so
`--output` has the same frame rate and length as the input.

## Model: `data/models/lc_seg_model.onnx`

| Tensor | Shape | Description |
|---|---|---|
| `image` (input) | `[N, 512, 512, 3]` float32 | RGB in [0, 1]. ImageNet mean/std normalization and NHWC→NCHW are inside the graph |
| `head_0` | `[N, 5, 512, 512]` | logits: Background, Gallbladder, Cystic duct, Cystic artery, Common bile duct |
| `head_1` | `[N, 3, 512, 512]` | logits: Background, Liver, Rouvieres sulcus |
| `head_2` | `[N, 4, 512, 512]` | logits: Background, Hepatoduodenal ligament, Duodenum, Fat |
| `head_3` | `[N, 7, 512, 512]` | logits: Background, bipolar, clipper, grasper, hook, irrigator, scissors |
| `correlation_scores` | `[N, 7]` | per-frame scores |

Channel 0 of every head is Background. Channels 1…n map to the global classes in `head_groups`.

Assumptions, to be checked against the training code:
- Each head is single-label per pixel (softmax), so the class is the argmax of the logits. On real
  `video01` frames the argmax gives plausible anatomy and tools, with mean foreground confidence of 0.77–0.99.
- Frames were stretched, not letterboxed, to 512×512 during training.

## Phase / RSD: `preprocess_step.onnx` + `multimodal_rsd_step.onnx`

Surgical phase, remaining surgical duration (RSD) and progress run at **1 frame per second** of
source time (`RSD_InferFPS`), the rate the model was trained and evaluated at. For a video file
that is every 25th frame of a 25 fps video; for a camera it is wall-clock time. The on-screen
result changes only every `RSD_Period` seconds (default 15). Each frame runs through:

1. **Geometry** (`RsdPreprocessor::CropFrame`), same as `Eval_MultiModal_RSD.ipynb`: short side 256
   (`INTER_AREA`) → torchvision `Resize(255)` → `CenterCrop(224)`, RGB uint8.
2. Two ONNX models:

| Model | Inputs → outputs |
|---|---|
| `preprocess_step.onnx` | `frame` uint8 RGB `[1, 3, 224, 224]` + rolling colour statistics `hist_stats` `[10, 1, 3, 4]`, `hist_count` → normalized `image` + updated statistics |
| `multimodal_rsd_step.onnx` | `image` + recurrent `conv_state` `[4, 1, 1024, 3]`, `ssm_state` `[4, 1, 16, 64, 16]` → `phase_probs` `[1, 7]`, `pred_rsd` (seconds), `pred_progress`, updated states |

Both models are stateful: each step's state outputs are fed back as the next step's inputs.
`preprocess_step.onnx` always runs on CPU (it is tiny); the main model uses CUDA and falls back to
CPU if its first GPU run fails. They run on their own thread, so segmentation is not delayed. The
displayed result stays in a bottom-left panel until the next `RSD_Period` replaces it. The panel is
drawn in `--show` and in the recording, and each displayed update is also printed:

```
[rsd step 31 @ 00:30] phase: CalotTriangleDissection (0.98)  RSD: 25:13  progress: 31.2%
```

Checked on `video01` (121 steps, 0–120 s at 1 fps) against the notebook's torchvision geometry
followed by the same ONNX models:
- Crops match within ±2 grey levels; the difference is interpolation rounding.
- Phase agrees on 100% of steps.
- RSD differs by 2.2 s on average (9.4 s at most, on values of about 1300–1600 s).
- GPU and CPU agree to 0.01 s.

### How the model was built (from `RSD/src/MultiModal_RSD.py`, `Export_ONNX.py`, `Eval_MultiModal_RSD.ipynb`)

```
RGB frame, short side 256 (INTER_AREA) → Resize(255) → CenterCrop(224)      uint8 [1,3,224,224]
  │ preprocess_step.onnx       Surgical3StepPreprocess, 10-step colour history as state
image                                                                         f32  [1,3,224,224]
  │ ┌── multimodal_rsd_step.onnx ───────────────────────────────────────────────────────────┐
  │ │ Swin-B (swin_base_patch4_window7_224) → mean-pooled 1024-d → ProjectionHead → z_euc 512 │
  │ │ PoincareProjectionLayer: exp_o^c(clip(z_euc, 2.3)), c = 1     → z_t on the ball B^512  │
  │ │ SurgicalMamba (4 layers, d_state 16), conv_state / ssm_state as explicit state        │
  │ │   ├─ SPRHead → phase_logits / phase_probs [1,7] (Cholec80 phases)                     │
  │ │   └─ RSDHead → pred_rsd [1,1] seconds ≥ 0, pred_progress [1,1] in [0,1]               │
  │ └────────────────────────────────────────────────────────────────────────────────────────┘
```

* The Swin encoder, the Poincaré-ball projection and the temporal model are all **inside**
  `multimodal_rsd_step.onnx` (`MultiModalStep` in `Export_ONNX.py` = `MultiModal_RSD.stream_step`).
  It also outputs `z_t`, the 512-d Poincaré-ball embedding of the frame.
* Training and evaluation step the model at **1 fps** (`frame_stride = 25` on 25 fps Cholec80, 32-frame
  clips = 32 s), and the preprocessing history is 10 steps = 10 s.

## Config: `configs/config.yaml`

The file has one section per procedure. Its top-level key is the procedure name, and that section
holds the keys below:

```yaml
laparoscopic cholecystectomy:
  head_groups: ...
  label_names: ...
```

`--procedure` (or `iSurgicalConfig::procedure`) selects the section. Set it to `""` in code to read a
file whose keys are at the top level. Loaded with yaml-cpp at startup. The program stops before loading the model if a key is missing,
or if `head_groups` does not match the head sizes, or if `label_names`/`kclass_colours` do not have `NumClass` entries.

| Key | Use |
|---|---|
| `head_groups` | global class id of channels 1…n of each head |
| `label_names`, `kclass_colours`, `NumClass` | 16 global classes (0 = Background). Colours are RGBA in [0, 1]. Alpha is the per-class overlay opacity of the fill and outline: 0 hides the class, 1 is opaque. Background is never drawn |
| `MaskThreshold` | a pixel keeps its class only if its softmax probability ≥ this, otherwise Background |
| `ConfidenceThreshold` | a region is dropped if its mean pixel probability < this |
| `IoUThreshold` | YOLO NMS setting; semantic segmentation has no overlapping instances, so it is parsed but unused |
| `MorphKernelSize`, `MinAreaRatio` | optional, defaults 5, 0.001 |
| `RSD_Period` | optional, seconds between on-screen phase / RSD updates, default 15 |
| `RSD_InferFPS` | optional, phase / RSD inference rate, default 1 (the training rate; change only with a model trained at another rate) |
| `Phase` | optional, the 7 phase names in model output order; default Cholec80: Preparation, Calot triangle dissection, Clipping and cutting, Gallbladder dissection, Gallbladder packaging, Cleaning and coagulation, Gallbladder retraction |

Postprocessing per head:
1. Apply `MaskThreshold`.
2. For each class, apply morphological open and close, then split into connected components.
3. Drop components smaller than `MinAreaRatio` of the frame, or whose mean confidence is below `ConfidenceThreshold`.

The four heads are then fused into one 16-class label map. Where heads overlap, the most
confident one wins. This map is what `Draw` shows unless `--head` selects a single head.

## Pipeline

```
capture thread          inference thread                          main thread
CaptureImg  ──[1]──▶  SegPreprocessor → SegInferenceEngine  ──[4]──▶  Draw → RsdPostProcessor::Draw → VideoRecord
 (real-time pacing)      → SegPostProcessor                          → imshow / console
     │                                                                 ▲
     │ 1 fps                rsd thread                                 │ displayed result, refreshed
     └──────[8]──────────▶  RsdInferenceEngine → RsdPostProcessor ─────┘ every RSD_Period s (mutex)
```

`[n]` is the queue capacity. The frame queue keeps only the newest frame. The result queue blocks
when it is full.

| Path | Purpose |
|---|---|
| `src/main.cpp` | entry point: builds an `iSurgicalConfig` (defaults + command-line options) and calls `iSurgical::Run()` |
| `src/isurgical/` | `iSurgicalConfig` (all settings, `FromArgs`) and `iSurgical`: model loading, thread setup and shutdown |
| `src/cap/` | `CaptureImg`: camera / video / stream capture, real-time file replay, measured FPS |
| `src/preproc/preproc_seg.*` | `SegPreprocessor`: BGR→RGB, resize (stretch or letterbox), optional border crop, `RestoreMask` back to frame size |
| `src/infer/infer_seg.*` | `SegInferenceEngine`: ONNX Runtime session (CUDA, falls back to CPU if the first GPU run fails), per-pixel argmax |
| `src/preproc/preproc_rsd.*` | `RsdPreprocessor`: eval-notebook crop + `preprocess_step.onnx` with its 10-step colour history |
| `src/infer/infer_rsd.*` | `RsdInferenceEngine`: `multimodal_rsd_step.onnx` (Swin → Poincaré ball → SurgicalMamba → phase / RSD heads) with carried state |
| `src/postproc/postproc_rsd.*` | `RsdConfig` (YAML), `RsdPostProcessor`: `RSD_Period` display refresh, console line, phase / RSD / progress panel |
| `src/postproc/postproc_seg.*` | `SegPostConfig` (YAML loader) and `SegPostProcessor`: class mapping, thresholds, cleanup, regions, head fusion, overlay with class names |
| `src/record/` | `VideoRecord`: background-thread video writer with a bounded queue |
| `src/common/blocking_queue.hpp` | bounded thread-safe queue (block or drop-oldest) |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDNN_FE failure 11` / `CUDNN_STATUS_EXECUTION_FAILED_CUDART` on `/model/encoder/_conv_stem/Conv`, then `falling back to CPU` | System cuDNN without Pascal kernels was loaded. Reconfigure with `-DCUDNN_LIB_DIR` pointing to cuDNN 9.5.1 and rebuild |
| `CUDA provider unavailable, falling back to CPU` | ONNX Runtime CUDA libraries could not be loaded (wrong ORT build for the installed CUDA, or missing CUDA 12 libraries) |
| `error while loading shared libraries: libonnxruntime.so.1` | Binary built without the RPATH (old build dir). Re-run CMake as above |
| Video is processed slowly / many frames skipped | Check the inference time printed per frame. About 1.2 s means it is running on CPU |
