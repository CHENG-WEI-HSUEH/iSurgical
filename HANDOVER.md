# isurgical — Handover Guide

This guide is for the engineer taking over `isurgical`. After reading it you should be able to
change, debug or optimise any part of the program without having to reverse-engineer it first.

- **What it is:** a C++17 real-time inference app for laparoscopic cholecystectomy video. Each
  frame is segmented (anatomy + instruments). Once per second it also recognises the surgical
  phase and predicts the remaining surgical duration (RSD) and progress. Results are printed,
  drawn, optionally recorded or shown, and published to a parent process through shared memory.
- **Binary:** `build/iprovision` (the CMake project is named `iprovision`; the repo directory and
  the C++ class are `isurgical`).
- **Upstream / downstream:** `iprovision_client` launches isurgical as a child process and reads
  its results from shared memory, then relays them to `ior_client`, which publishes them over NATS
  to `iprovision_server`. isurgical itself does no networking.

Facts in this guide were checked against the source, the ONNX graphs in `data/models/` and the
training code in `../RSD/src/` on 2026-10-29. Measurements are from the development machine
(GTX 1080 Ti, `data/inputs/video01.mp4`, 854×480 @ 25 fps).

---

## Contents

1. [System context](#1-system-context)
2. [Build, run, environment](#2-build-run-environment)
3. [Source layout](#3-source-layout)
4. [Runtime architecture (threads and queues)](#4-runtime-architecture-threads-and-queues)
5. [Models](#5-models)
   - 5.1 [Segmentation: `lc_seg_model.onnx`](#51-segmentation-lc_seg_modelonnx)
   - 5.2 [Phase / RSD preprocessing: `preprocess_step.onnx`](#52-phase--rsd-preprocessing-preprocess_steponnx)
   - 5.3 [Phase / RSD: `multimodal_rsd_step.onnx`](#53-phase--rsd-multimodal_rsd_steponnx)
   - 5.4 [Unused: `swin_hyp_spatial_encoder.onnx`](#54-unused-swin_hyp_spatial_encoderonnx)
6. [Module reference](#6-module-reference)
7. [Configuration: `configs/config.yaml`](#7-configuration-configsconfigyaml)
8. [Shared-memory interface](#8-shared-memory-interface)
9. [Performance profile](#9-performance-profile)
10. [Known issues and pitfalls](#10-known-issues-and-pitfalls)
11. [Optimisation opportunities](#11-optimisation-opportunities)
12. [How-to recipes](#12-how-to-recipes)

---

## 1. System context

```
                        fork + execv, args: --show [--output ... --raw-output ...] --shm <name>
iprovision_client  ─────────────────────────────────────────────────────────────▶  isurgical
 (AppManager)      ◀──────── POSIX shared memory (src/common/isurgical_shm.hpp) ───  (this repo)
       │
       │ POSIX shared memory (iprovision_client/src/common/ior_shm.hpp)
       ▼
   ior_client  ──── NATS, subject "isurgical.status", JSON ────▶  iprovision_server (dashboard)
```

| Link | Mechanism | Files |
|---|---|---|
| iprovision_client → isurgical | command line | `iprovision_client/src/main.cpp:120-125`, `iprovision_client/src/app_manager/app_manager.cpp:16-25` |
| isurgical → iprovision_client | shared memory | `src/common/isurgical_shm.hpp` (both sides compile this header) |
| iprovision_client → ior_client | shared memory | `iprovision_client/src/common/ior_shm.hpp` |
| ior_client → iprovision_server | NATS | `ior_client/src/nats/nats.*`, `iprovision_server/iprovision_server.py` |

iprovision_client runs isurgical with its working directory set to this repo, so every relative
path below (models, config, outputs) resolves from the repo root.

---

## 2. Build, run, environment

### Dependencies

| Component | Version | Notes |
|---|---|---|
| GPU | GTX 1080 Ti (Pascal, sm_61) | Pascal limits the CUDA / cuDNN / ORT versions below |
| ONNX Runtime | `onnxruntime-linux-x64-gpu-1.20.1` | CUDA 12 build. 1.30 needs CUDA 13, which dropped Pascal |
| cuDNN | 9.5.1 | **Must be Pascal-capable.** The system cuDNN 9.26 fails with `CUDNN_STATUS_EXECUTION_FAILED_CUDART` |
| CUDA runtime | 12.x (system) | |
| OpenCV | 4.6 (system) | |
| yaml-cpp | 0.8 (system) | |
| TBB | system | linked in `CMakeLists.txt` |
| CMake | ≥ 3.20 | |

### Build

```bash
cd /media/andyhsueh/Storage2/smartsurgery/isurgical
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT_DIR=/media/andyhsueh/Storage2/sideproject/Onnx/onnxruntime-linux-x64-gpu-1.20.1 \
      -DCUDNN_LIB_DIR=/media/andyhsueh/Storage2/sideproject/Onnx/cudnn-9.5.1/lib
cmake --build build -j$(nproc)
```

- The ORT and cuDNN paths are embedded as **DT_RPATH** (not RUNPATH) on purpose. ORT `dlopen`s
  `libonnxruntime_providers_cuda.so`, and only the executable's DT_RPATH reaches that library's
  cuDNN dependency. That is how the Pascal-capable cuDNN wins over the system one. Do not
  "modernise" this to RUNPATH.
- If cuDNN is wrong, the first GPU run fails and each engine silently falls back to CPU
  (about 1.2 s per frame). Check the log for `falling back to CPU`.

### Command line

Run from the repo root.

| Option | Default | Effect |
|---|---|---|
| `--model <path>` | `data/models/lc_seg_model.onnx` | segmentation model |
| `--config <path>` | `configs/config.yaml` | classes, colours, thresholds, phases |
| `--procedure <name>` | `laparoscopic cholecystectomy` | top-level section of the config |
| `--input <src>` | `data/inputs/video01.mp4` | camera index (`0`), file, or URL with `://` |
| `--output <path>` | none | annotated video (stream) or image (single image) |
| `--raw-output <path>` | none | unannotated source, every frame |
| `--head <0-3>` | `-1` (all fused) | draw only one head |
| `--cpu` | off | disable CUDA |
| `--letterbox` | off | keep aspect ratio for the seg input (default: stretch) |
| `--crop-border` | off | crop the dark endoscope border before the seg resize |
| `--show` / `--windowed` | off | display full screen / in a window (q or Esc quits) |
| `--no-rsd` | off | disable phase / RSD |
| `--shm <name>` | none | attach to the parent's shared memory |

Output directories must already exist; `Init()` checks this before loading any model.

> **Warning:** the default run is currently broken for video files. See
> [Known issue #1](#10-known-issues-and-pitfalls).

Exit code: `0` on success, `-1` (255) on any failure (bad option, config, model, input, inference,
or recording).

---

## 3. Source layout

```
isurgical/
├── CMakeLists.txt
├── configs/config.yaml            procedure config (classes, colours, thresholds, phases)
├── data/
│   ├── inputs/video01.mp4         Cholec80 sample
│   ├── models/*.onnx              see §5
│   └── outputs/                   recordings (written by iprovision_client sessions)
└── src/
    ├── main.cpp                   CLI → iSurgicalConfig → iSurgical::Run()
    ├── isurgical/isurgical.*      orchestration: config, init, threads, display, shared memory
    ├── cap/cap.*                  CaptureImg: camera / file / stream, real-time file pacing
    ├── record/record.*            VideoRecord: background-thread cv::VideoWriter
    ├── common/
    │   ├── blocking_queue.hpp     bounded thread-safe queue (block or drop-oldest)
    │   └── isurgical_shm.hpp      shared-memory layout + seqlock helpers (shared with iprovision_client)
    ├── preproc/
    │   ├── preproc_seg.*          SegPreprocessor: frame → 512×512 RGB float; RestoreMask back
    │   └── preproc_rsd.*          RsdPreprocessor: crop to 224 + preprocess_step.onnx
    ├── infer/
    │   ├── infer_seg.*            SegInferenceEngine: ORT session, argmax + softmax confidence
    │   └── infer_rsd.*            RsdInferenceEngine: ORT session, carries the Mamba state
    └── postproc/
        ├── postproc_seg.*         SegPostConfig (YAML) + SegPostProcessor: regions, fusion, drawing
        └── postproc_rsd.*         RsdConfig (YAML) + RsdPostProcessor: display cadence, panel
```

The code follows a **pre → infer → post** split per model. Each stage is a class with a narrow API
and no knowledge of threads; `isurgical.cpp` is the only place that wires stages to threads.

---

## 4. Runtime architecture (threads and queues)

`iSurgical::Run()` calls `Init()`, then either `RunImage()` or `RunStream()` depending on
`config_.input_method` (see Known issue #1).

### Init order (`isurgical.cpp`, `Init()`)

Checks run cheapest first, so bad input fails before the slow model loads:

1. `--head` range.
2. Output directories exist.
3. Parse YAML → `SegPostProcessor`, `RsdPostProcessor` (throws on bad config).
4. If `--shm`: `OpenSharedMemory()` (checks `kMagic` / `kVersion`), then `PublishTables()` and set
   `tables_ready = 1`.
5. Load `SegInferenceEngine` (validates model I/O names, see §5.1).
6. If RSD is on: load `RsdInferenceEngine`.

### Stream pipeline (`RunStream()`)

```
capture thread                 inference thread                       main thread (DisplayLoop)
CaptureImg::Capture ──frame_queue[1, drop-oldest]──▶ SegInferenceEngine::Run ──result_queue[4, blocking]──▶ print
  │ raw recording (every frame)                       SegPostProcessor::Process                       publish shm status
  │                                                                                                    draw seg + RSD panel
  │ every 1/RSD_InferFPS s of source time                                                              record / imshow
  └──rsd_queue[8, drop-oldest]──▶ rsd thread: RsdInferenceEngine::Run → RsdPostProcessor::Update
                                   (writes latest_rsd and the displayed result under rsd_mutex)
```

| Queue | Capacity / policy | Why |
|---|---|---|
| `frame_queue` | 1, drop oldest | inference always takes the newest frame, so it never lags behind a live feed; skipped frames are counted (`Dropped()`) |
| `result_queue` | 4, blocking | no result is lost between inference and display |
| `rsd_queue` | 8, drop oldest | absorbs short RSD stalls; drops only if RSD falls 8 steps behind |

Key behaviours:

- **Source time:** for files it is `index / source_fps` (matches the video timeline, so RSD samples
  every 25th frame of a 25 fps file, as in training). For live sources it is wall-clock time.
- **Real-time replay:** `CaptureImg::PaceReplay()` sleeps so files play at their own frame rate,
  like a camera.
- **Recording keeps the source timeline:** `AnnotatedRecorder` repeats the last annotated frame for
  frames that inference skipped, so `--output` has the same length and fps as the input. Raw
  recording happens in the capture thread so it gets every frame.
- **Two RSD results exist:** `latest_rsd` (refreshed every second) goes to shared memory.
  `RsdPostProcessor::Displayed()` (refreshed every `RSD_Period` s, default 15) is drawn on screen.
- **Failure isolation:** if segmentation fails, the whole run stops. If RSD fails, only the RSD
  thread stops and segmentation continues.
- **Shutdown:** the main loop ends (input ends, q/Esc, `stop_requested` from the parent, or a
  recording error). `RunStream()` then sets `stop`, closes every queue (this wakes any blocked
  `Push` / `Pop`), and joins the threads in pipeline order.

### Single image (`RunImage()`)

Same stages, called inline once: seg → post → one RSD step → publish → `imwrite` → optional
`imshow` + `waitKey(0)`.

---

## 5. Models

| File | Size | Runs on | Rate | Used |
|---|---|---|---|---|
| `lc_seg_model.onnx` | 112 MB, 27.8 M params | CUDA (CPU fallback) | every processed frame | yes |
| `preprocess_step.onnx` | 70 KB | always CPU | 1 fps | yes |
| `multimodal_rsd_step.onnx` | 389 MB | CUDA (CPU fallback) | 1 fps | yes |
| `swin_hyp_spatial_encoder.onnx` | 246 MB | — | — | **no** (code removed 2026-10-29) |

Common ORT settings in every engine: `SetIntraOpNumThreads(1)`,
`ORT_ENABLE_EXTENDED`, CUDA EP on device 0. **CPU fallback rule:** if the *first* GPU run throws
(typical cause: a cuDNN without kernels for Pascal), the session is rebuilt on CPU and the same
input is retried. After one successful run, an error is treated as a real failure.

### 5.1 Segmentation: `lc_seg_model.onnx`

Exported from PyTorch 2.5.0, opset 17. The training code is **not** in this workspace; the
architecture below was read from the ONNX graph.

#### I/O

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `image` (in) | `[N, 512, 512, 3]` | float32 | **RGB**, NHWC, range [0, 1]. ImageNet mean/std and NHWC→NCHW are inside the graph |
| `head_0` | `[N, 5, 512, 512]` | float32 | logits: Background, Gallbladder, Cystic duct, Cystic artery, Common bile duct |
| `head_1` | `[N, 3, 512, 512]` | float32 | logits: Background, Liver, Rouvieres sulcus |
| `head_2` | `[N, 4, 512, 512]` | float32 | logits: Background, Hepatoduodenal ligament, Duodenum, Fat |
| `head_3` | `[N, 7, 512, 512]` | float32 | logits: Background, bipolar, clipper, grasper, hook, irrigator, scissors |
| `correlation_scores` | `[N, 7]` | float32 | Pearson correlations between class node features (see below) |

In the ONNX metadata the head channel counts are dynamic (`Resizehead_0_dim_1`; all four heads
reuse that name). The real values `{5, 3, 4, 7}` come from the weight shapes of
`/model/heads.N/Conv` (`[C, 256, 1, 1]`). That is why they are hard-coded in
`SegInferenceEngine::kHeadClasses`. In Netron, click a `heads.N/Conv` node and read `W`.

#### Architecture

```
image [N,512,512,3] → (Sub mean, Div std, Transpose to NCHW)
  │
  ├─ Encoder: EfficientNet (efficientnet_pytorch naming: _conv_stem, _blocks.0.._blocks.38)
  │    48-ch stem, 39 MBConv blocks with squeeze-excitation → matches EfficientNet-B5
  │    multi-scale features taken at 40 / 64 / 176 / 512 channels (strides 4 / 8 / 16 / 32)
  │
  ├─ Decoder: SegFormer-style all-MLP
  │    mlps.i.proj: Linear(C_i → 256) per scale → upsample to the stride-4 grid → concat (1024)
  │    linear_fuse: Conv1×1(1024 → 256) + ReLU             → fused feature F [N,256,128,128]
  │
  ├─ Heads: heads.k = Conv1×1(256 → C_k) on F → Resize to 512×512 → head_k logits
  │
  └─ GCN branch (does NOT feed the heads):
       GlobalAveragePool(F) [N,256] × const + learned class embeddings [1,16,256] → 16 nodes
       gcn_refiner: Linear(256,256) → A·X → ReLU → Linear(256,256) → A·X     (A = learned 16×16 adj)
       Pearson correlation over the 256-d node features for 7 fixed class pairs → correlation_scores
```

`correlation_scores` index → class pair (decoded from the graph):

| idx | pair (global class ids) | classes |
|---|---|---|
| 0 | 4, 1 | Liver – Gallbladder |
| 1 | 1, 2 | Gallbladder – Cystic duct |
| 2 | 1, 3 | Gallbladder – Cystic artery |
| 3 | 2, 5 | Cystic duct – Hepatoduodenal ligament |
| 4 | 3, 5 | Cystic artery – Hepatoduodenal ligament |
| 5 | 2, 13 | Cystic duct – Common bile duct |
| 6 | 4, 15 | Liver – Rouvieres sulcus |

These look like anatomical-relationship scores (Calot's-triangle structures), in [-1, 1]. They
are only printed and carried in `SegPostResult`. They are not used in drawing or in shared memory.
The comment in `infer_seg.hpp` ("correlation scores for the 7 global classes, in the order of the
config file") is inaccurate; the table above is correct.

#### Preprocessing (`SegPreprocessor::Process`)

1. Optional `--crop-border`: grey → threshold at `border_threshold` (15) → bounding rect of the
   bright area. It is ignored if the rect is smaller than `min_crop_ratio` (25%) of the frame.
2. Resize to 512×512, either **stretch** (default) or **letterbox** (centred, padded with
   `pad_value`), using `INTER_LINEAR`.
3. BGR → RGB, `/255` → float, written directly into the ORT input buffer (no extra copy).
4. `SegPreprocInfo` records the original size, the crop rect and where the content landed.
   `RestoreMask` uses it to invert the mapping.

The model was presumably trained with **stretch**; switching to letterbox changes the input
distribution.

#### Postprocessing

In `SegInferenceEngine::Run` (CPU, per head):
- argmax over channels → `masks[h]` (CV_8UC1, **head-local** ids, 0 = Background);
- softmax probability of the winning class → `confidences[h]` (CV_32FC1; 0 for Background, which
  skips the `exp` work);
- both are mapped back to frame size with `RestoreMask` (`INTER_NEAREST`, so ids never blend).

In `SegPostProcessor::Process` (per head, then fusion):
1. Pixels with confidence < `MaskThreshold` → Background.
2. For each local class: binary mask → morphological open + close (ellipse kernel,
   `MorphKernelSize`, default 5) → connected components (8-connectivity).
3. Drop components with area < `MinAreaRatio` × frame (default 0.001) or mean confidence <
   `ConfidenceThreshold`.
4. Map local → global id via `head_groups`, and build `SegRegion` (area, ratio, confidence, bbox,
   centroid, largest outer contour). Regions are sorted by area.
5. `FuseHeads`: one 16-class `label_map`; where heads overlap, the highest-confidence head wins.

`Draw`: per-class alpha blend via LUTs (alpha from `kclass_colours`), contour outlines with the
same alpha, and a fixed legend at the top right. The RSD panel is drawn at the bottom left by
`RsdPostProcessor::Draw`.

### 5.2 Phase / RSD preprocessing: `preprocess_step.onnx`

Source: `../RSD/src/Preprocess.py` (`Surgical3StepPreprocess`), exported as `PreprocessStep` in
`../RSD/src/Export_ONNX.py`. **Stateful**: the history is an explicit input and output.

#### Geometry before the model (`RsdPreprocessor::CropFrame`, C++)

This step matches `Eval_MultiModal_RSD.ipynb` exactly:

1. If short side > 256: resize so short side = 256 (`INTER_AREA`, sizes rounded).
2. torchvision `Resize(255)`: short side → 255, long side = `int(255 × long / short)`
   (`INTER_LINEAR`).
3. torchvision `CenterCrop(224)`: offsets use Python `round()` (round half to even), reproduced
   with `std::nearbyint`. **Do not replace it with `std::lround`**; that shifts the crop by one
   pixel.
4. BGR → RGB, then split to planar CHW uint8.

#### I/O

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `frame` (in) | `[1, 3, 224, 224]` | uint8 | RGB planar |
| `hist_stats` (in) | `[10, 1, 3, 4]` | float32 | ring of the last 10 frames' per-channel `[sum, sum², p25, p75]` |
| `hist_count` (in) | `[1]` | int64 | frames seen |
| `image` (out) | `[1, 3, 224, 224]` | float32 | ImageNet-normalised input for the RSD model |
| `hist_stats_out`, `hist_count_out` | as above | | fed back on the next call |

#### What it computes

1. Percentile clipping (p2.5 to p97.5 per channel) → rescale to [0, 1]. This removes specular
   highlights and the black border.
2. Soft global dehazing: a dark-channel-prior model with one scalar transmission per frame, which
   reduces cautery smoke without spatial flicker.
3. Causal history alignment: the frame is re-levelled with mean/std and p25/p75 from the last 10
   frames. This step is bypassed until 10 frames have been seen.
4. ImageNet normalisation.

The history spans **10 RSD steps = 10 s** at 1 fps. `RsdInferenceEngine::Reset()` clears it; call
that whenever the stream changes (new video, seek).

### 5.3 Phase / RSD: `multimodal_rsd_step.onnx`

Source: `../RSD/src/MultiModal_RSD.py` (`stream_step`), `SurgicalMamba.py`, `SPR_Head.py`,
`RSD_Head.py`. Exported as `MultiModalStep` (`Export_ONNX.py`). **Stateful**, and trained and
evaluated at **1 fps** (`RSD_InferFPS`). Calling it at a different rate changes its meaning.

#### I/O

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `image` (in) | `[1, 3, 224, 224]` | float32 | output of `preprocess_step` |
| `conv_state` (in) | `[4, 1, 1024, 3]` | float32 | per layer: causal conv history (layers, batch, d_inner, d_conv−1) |
| `ssm_state` (in) | `[4, 1, 16, 64, 16]` | float32 | per layer: SSM state (layers, batch, heads, headdim, d_state) |
| `phase_logits` | `[1, 7]` | float32 | not read by the C++ code |
| `phase_probs` | `[1, 7]` | float32 | softmax over the 7 Cholec80 phases |
| `pred_rsd` | `[1, 1]` | float32 | remaining duration, **seconds**, ≥ 0, capped at 7200 |
| `pred_progress` | `[1, 1]` | float32 | progress in [0, 1] |
| `z_t` | `[1, 512]` | float32 | Poincaré-ball frame embedding (copied to `RsdResult::z_t`, currently unused) |
| `conv_state_out`, `ssm_state_out` | as inputs | | fed back on the next step |

Both states start at zero (`Reset()`). They are committed only after a successful run, so the CPU
retry repeats the same step.

#### Architecture

```
image [1,3,224,224]
  → Swin-B (swin_base_patch4_window7_224) → mean-pooled 1024-d
  → ProjectionHead → z_euc [512]
  → PoincareProjectionLayer: exp_o^c(clip(z_euc, r=2.3)), curvature c = 1  → z_t [512] on B^512
  → SurgicalMamba (pure-PyTorch "ssm" mixer, Mamba-2 / SSD style):
        in_proj + norm, then 4 pre-norm residual blocks,
        d_model 512, expand 2 → d_inner 1024, headdim 64 → 16 heads, d_state 16, d_conv 4
        recurrence h_t = exp(dt·A)·h_{t-1} + dt·B_t·x_t,  y_t = C_t·h_t + D·x_t   (strictly causal)
  → h_t [512]
      ├─ SPRHead: per-frame MLP (LayerNorm → Linear → LayerNorm → GELU …) → 7 logits → softmax
      └─ RSDHead: shared trunk → rsd (in units of 60 s, softplus, ×60, cap 7200 s)
                               → progress (sigmoid)
```

The hyperbolic projection that SurgicalMamba consumes is **inside** this model. It needs no
external embedding.

#### Postprocessing (`RsdPostProcessor`, `PublishStatus`)

- `phase = argmax(phase_probs)`, confidence = max prob.
- Display cadence: `Update()` accepts a result as "displayed" only every `RSD_Period` seconds of
  video time. `Describe()` prints
  `[rsd step 31 @ 00:30] phase: CalotTriangleDissection (0.98)  RSD: 25:13  progress: 31.2%`.
- Shared memory: `progress` rounded to 2 decimals, `remaining_minutes = lround(rsd_s / 60)`,
  phase name from the config. Before the first result: phase `""`, progress −1, minutes −1.

### 5.4 Unused: `swin_hyp_spatial_encoder.onnx`

This is the encoder half of a separate Swin-B hyperbolic auto-encoder
(`../RSD/notebooks/Train_Swin_Hyperbolic_AE_Cholec80.ipynb`). Its outputs are `z_hyp [1,14,14,64]`
and `global_embedding [1,64]`. Nothing consumed its output (it was only printed), so its code was
removed on 2026-10-29. That saved about 540 MiB of GPU memory and removed a startup failure point.
The model file is still in `data/models/` and can be deleted. It is **not** an input to
SurgicalMamba (different dimension, separately trained). Feeding it in would require retraining
the RSD model.

---

## 6. Module reference

### `isurgical/` — `iSurgicalConfig`, `iSurgical`

- `iSurgicalConfig` holds every setting. `FromArgs` maps the CLI onto it, so the same settings can
  be set in code (see the README example).
- `iSurgical` owns the engines and post-processors (`unique_ptr`; `rsd_engine_` is null when RSD is
  off) and the mapped `SharedState*`.
- `StreamPipeline` (in the `.cpp`) is all cross-thread state: queues, atomics, `rsd_mutex`,
  `latest_rsd`.
- `rsd_mutex` guards `latest_rsd` **and** `RsdPostProcessor`'s displayed state. The RSD thread
  writes both; the main thread reads them.
- Shared-memory functions: `OpenSharedMemory`, `PublishTables`, `PublishStatus`, `StopRequested`
  (§8).

### `cap/` — `CaptureImg`

- A source made only of digits is treated as a camera index. A source containing `://` counts as a
  live stream. Anything else is a file.
- `IsLive()` drives the source-time computation and whether replay pacing is applied.
- `GetSourceFPS()` returns 30 if the container does not report a rate. `GetFPS()` is the measured
  capture rate (1 s windows), shown in the corner of `--show`.

### `record/` — `VideoRecord`

- `cv::VideoWriter` (`mp4v`) on its own thread. `WriteFrame` copies into a 64-frame queue and
  blocks only when the queue is full. `Release()` flushes and joins.
- The file is opened lazily on the first frame (the size is unknown before that).

### `common/blocking_queue.hpp` — `BlockingQueue<T>`

- `drop_oldest=false`: `Push` blocks when full. `true`: `Push` discards the oldest item and counts
  it.
- `Close()` wakes everyone; `Pop` drains the remaining items, then returns false.

### `preproc/`, `infer/`, `postproc/`

See §5 for what each class computes. API summary:

| Class | Key API | Notes |
|---|---|---|
| `SegPreprocessor` | `Process(frame, float* dst, info)`, `static RestoreMask(...)` | reuses member buffers; `dst` is the ORT input buffer |
| `SegInferenceEngine` | `Run(frame, SegResult&)`, constants `kNumHeads=4`, `kNumScores=7`, `kHeadClasses={5,3,4,7}` | `CheckModelIO()` enforces input `image` and outputs `head_0..3`, `correlation_scores` |
| `SegPostProcessor` | `Process`, `Draw(frame, result, out, head)`, `LabelName`, `Config` | stateless per call (`const`); safe to call from one thread at a time |
| `RsdPreprocessor` | `CropFrame` (static), `Process`, `Reset` | own CPU ORT session |
| `RsdInferenceEngine` | `Run(frame, time_s, RsdResult&)`, `Reset` | owns the preprocessor and both Mamba states |
| `RsdPostProcessor` | `Update`, `Displayed`, `Draw`, `Describe`, `PhaseName` | holds the display state; guarded by `rsd_mutex` in stream mode |

---

## 7. Configuration: `configs/config.yaml`

One top-level section per procedure (`--procedure` selects one; `""` reads the file root). It is
validated at startup and the program stops before loading models if anything is inconsistent.

| Key | Required | Meaning / constraint |
|---|---|---|
| `head_groups` | yes | 4 lists; list *k* gives the global class id of head *k* channels 1…C−1. Lengths must equal `kHeadClasses[k]−1` = 4, 2, 3, 6 |
| `label_names` | yes | `NumClass` names, index 0 = Background |
| `kclass_colours` | yes | `NumClass` × RGBA in [0, 1]; **alpha = overlay opacity** (0 hides a class) |
| `NumClass` | yes | 16 here, ≤ 256 |
| `MaskThreshold` | yes | per-pixel probability floor (0.5) |
| `ConfidenceThreshold` | yes | per-region mean-probability floor (0.25) |
| `IoUThreshold` | no | parsed, **unused** (a leftover from a YOLO pipeline) |
| `MorphKernelSize` | no | default 5; 0 disables morphology |
| `MinAreaRatio` | no | default 0.001 |
| `RSD_Period` | no | seconds between on-screen RSD updates, default 15 |
| `RSD_InferFPS` | no | default 1; change it only with a model trained at another rate |
| `Phase` | no | exactly 7 names, in model output order; default Cholec80 |

Global class ids: 0 Background, 1 Gallbladder, 2 Cystic duct, 3 Cystic artery, 4 Liver,
5 Hepatoduodenal ligament, 6 bipolar, 7 clipper, 8 grasper, 9 hook, 10 irrigator, 11 scissors,
12 Duodenum, 13 Common bile duct, 14 Fat, 15 Rouvieres sulcus.

---

## 8. Shared-memory interface

Defined in `src/common/isurgical_shm.hpp`, which is **compiled by both** isurgical and
`iprovision_client`.

```
SharedState
  uint32 magic          = kMagic (0x47555349, "ISUG" in memory) — set by parent, checked by isurgical
  uint32 version        = kVersion (1)                           — bump on ANY layout change
  atomic<u32> tables_ready    1 once Tables is written (release store)
  atomic<u32> stop_requested  parent sets it → isurgical finishes cleanly (recordings closed)
  atomic<u64> status_seq      seqlock: odd = write in progress, 0 = never written
  Tables  { num_phases, phase_names[16][64], num_classes, class_names[32][64], class_colours[32][3] RGB 0..1 }
  Status  { timestamp (time_t), frame_index, phase[64], detected[32] (region count per class),
            progress (−1 unknown), remaining_minutes (−1 unknown) }
```

Protocol:

1. The parent `shm_open`s and sizes the segment, zero-fills it, writes magic and version, then
   starts isurgical with `--shm <name>`.
2. isurgical attaches (`O_RDWR`, never creates), validates magic and version, and writes `Tables`
   during `Init()` (before the models load).
3. Every displayed frame: `WriteStatus()` (single writer, seqlock). The reader uses `ReadStatus()`,
   which retries until it gets a copy that was not torn.
4. The parent sets `stop_requested`; isurgical's main loop exits on the next frame.

Rules: only fixed-size, trivially copyable members; atomics must be lock-free (both enforced by
`static_assert`). After any change, bump `kVersion` and rebuild **both** projects.

---

## 9. Performance profile

Measured on 2026-10-29 (GTX 1080 Ti, `video01.mp4` 854×480 @ 25 fps, stream mode, Release):

| Item | Value |
|---|---|
| Segmentation, per processed frame (`engine_->Run`, includes CPU argmax/softmax and RestoreMask) | ~46–47 ms median |
| Of that, the CPU argmax + softmax loop (4 heads, 512×512) | ~10 ms (single thread) |
| Frames processed | ~1 in 2 of a 25 fps source (the rest are skipped by design) |
| Peak GPU memory, whole process | ~1.75 GB |
| CPU-only segmentation | ~1.2 s per frame |
| Phase / RSD | 1 step per second on its own thread; does not delay segmentation |

The per-frame console line reports `ms` (inference + seg-model postproc) and `latency`
(capture → display).

---

## 10. Known issues and pitfalls

| # | Issue | Where | Impact / fix |
|---|---|---|---|
| 1 | **`input_method` defaults to `"video"`, which calls `RunImage()`, which uses `cv::imread`. A video file therefore fails with "Could not read the image".** There is no CLI flag for `input_method`. | `isurgical.cpp` `Run()`, `isurgical.hpp` | The default run fails, and so does iprovision_client's launch (it passes no `--input`). Fix: choose `RunImage` vs `RunStream` by trying `cv::imread` / the file extension, or route `"video"` to `RunStream()` and add a CLI flag. Until then, test stream mode by setting `config.input_method = "rtsp"` in code. |
| 2 | Inaccurate comments in `infer_seg.hpp`: input described as BGR (it is RGB); `correlation_scores` described as "7 global classes" (they are 7 class *pairs*, §5.1); the `kNumHeads` / `kNumScores` comment lines are swapped. | `src/infer/infer_seg.hpp` | Misleading only. |
| 3 | `postproc_seg.hpp` / `postproc_rsd.hpp` refer to `configs/lc_config.yaml`; the real file is `configs/config.yaml`. | headers | Misleading only. |
| 4 | Not a git repository. | repo root | Run `git init` and make a first commit before making changes. |
| 5 | Stale object files in `build/CMakeFiles` (e.g. `postproc_step.cpp.o`). | `build/` | Harmless for CMake; `make clean` removes them. They break hand-written link commands that glob `*.o`. |
| 6 | `swin_hyp_spatial_encoder.onnx` (246 MB) is unused. | `data/models/` | Delete it if it is not needed elsewhere. |
| 7 | `RsdResult::z_t` (512 floats) is copied every step but not used. | `infer_rsd.*` | Negligible cost; it is kept as a ready-made embedding for future use. |
| 8 | Head channel counts and the `head_groups` lengths are coupled to the model, and the ONNX metadata does not expose the counts. | `infer_seg.hpp`, `config.yaml` | Swapping the model requires updating both (§12). |
| 9 | ORT sessions are single-threaded on purpose (`SetIntraOpNumThreads(1)`), so several processes can share the CPU. | all engines | Keep this in mind before adding CPU parallelism. |
| 10 | README performance figures (63 ms per frame, 52→63 ms with the encoder) are older than the measurements in §9. | `README.md` | Re-measure before quoting them. |

Pitfalls when editing:

- **Do not change `CenterCrop` rounding** (`std::nearbyint`) or the `INTER_AREA` → `INTER_LINEAR`
  order in `RsdPreprocessor::CropFrame`. They reproduce the training geometry to ±2 grey levels.
- **Use `INTER_NEAREST` for anything that resizes class-id maps.**
- **Keep RSD at 1 step per second of source time.** Both the Mamba state and the 10-step
  preprocessing history assume it.
- Frames are shared read-only between the seg and RSD threads (`cv::Mat` refcount). **Never modify
  a captured frame in place**; `CaptureLoop` creates a fresh `Mat` for each frame for this reason.
- Anything added to `SharedState` requires bumping `kVersion` and rebuilding iprovision_client.

---

## 11. Optimisation opportunities

These are ranked by expected value. Measure end to end before and after each one; the console
`ms` / `latency` lines and `frames skipped by inference` are enough.

1. **Move argmax and softmax into the ONNX graph.** Today 19 logit channels × 512×512 float32 (~20 MB)
   are copied from GPU to CPU every frame, and then reduced on the CPU (~10 ms). Appending
   `ArgMax` + `Softmax`/`ReduceMax` per head (or exporting them from PyTorch) returns 4 uint8 masks
   + 4 float confidence maps. That cuts the transfer by ~4× and removes the CPU loop. This needs a
   re-export or graph surgery with `onnx`/`onnx-graphsurgeon`; validate that the masks are bit-equal.
2. **Parallelise the CPU argmax loop** (`infer_seg.cpp`, Step 3) with `cv::parallel_for_` over
   rows. A benchmark gave identical output and 10.2 → 3.3 ms at 4 cores, 1.8 ms at 8. Cap the
   threads with `cv::setNumThreads` (issue #9). `std::execution::par` was similar but needs an
   index vector; restructuring the loop channel-major was *slower* single-threaded.
3. **Run heads at 128×128 and upsample the masks instead of the logits.** The model resizes the
   logits from the stride-4 grid to 512×512 inside the graph. Taking the argmax at 128×128 and
   upsampling ids with nearest neighbour is about 16× less argmax work. This needs an export change.
4. **ORT IO binding / pinned memory** for the seg session, to avoid reallocating output tensors
   every run.
5. **TensorRT EP**: possible on Pascal with FP32. **FP16 is not worth it on a GTX 1080 Ti** (Pascal
   GP102 FP16 throughput is a tiny fraction of FP32). Re-evaluate on newer GPUs.
6. **Post-processing**: `ProcessHead` runs morphology and connected components per class on full
   frame-resolution masks. Working at model resolution (before `RestoreMask`) and scaling the
   region stats afterwards would reduce the cost on large frames.
7. **Preprocessing**: `cv::resize` + `cvtColor` + `convertTo` could be fused (or moved into the
   graph as uint8 input) to save one full-frame pass.

---

## 12. How-to recipes

### Swap in a new segmentation model

1. Check the I/O in Netron or with `onnx`: input `image [N,512,512,3]` RGB [0, 1]; outputs `head_0..3`
   + `correlation_scores`. `CheckModelIO()` rejects other names.
2. Read each `heads.k/Conv` weight shape → update `kHeadClasses` (and `kNumHeads` / `kNumScores` if
   they changed) in `infer_seg.hpp`.
3. Update `head_groups` (lengths = C−1), `label_names`, `kclass_colours`, `NumClass` in the config.
4. If the input size or normalisation changed: `kInputWidth/Height` in `infer_seg.hpp` and
   `SegPreprocessor`.
5. If the training resize differs (letterbox vs stretch), change the default in `SegPreprocConfig`.
6. Rebuild and compare masks on `video01` against the training framework.

### Add a new procedure

Add a top-level section to `config.yaml` with every required key, and pass `--procedure "<name>"`
(or set it in iprovision_client's args). Head sizes are fixed by the model, so a new procedure with
different classes also needs a new model.

### Change what the parent receives

Edit `Status` / `Tables` in `isurgical_shm.hpp`, bump `kVersion`, fill the field in
`PublishStatus()` / `PublishTables()`, then update and rebuild `iprovision_client`
(`app_manager.*`), and, if it is forwarded, `ior_shm.hpp` / `ior_client` / the NATS JSON.

### Change the RSD display or sampling

- Screen refresh: `RSD_Period` in the config.
- Inference rate: `RSD_InferFPS`. Leave it at 1 unless the model was retrained at another rate.
- Panel look: `RsdPostProcessor::Draw`.

### Debug checklist

| Symptom | Look at |
|---|---|
| Every engine is slow (~1 s per frame) | log says `falling back to CPU` → wrong cuDNN on RPATH (§2) |
| `Unexpected model output ...` | model I/O names differ from `CheckModelIO()` |
| `head_groups[k] must have N classes` | config does not match `kHeadClasses` |
| `Shared memory ... unexpected layout` | parent and child built from different `isurgical_shm.hpp` (version mismatch) |
| Masks shifted or distorted | letterbox / crop-border flags vs. training; `RestoreMask` inputs |
| Phase / RSD drifts or is nonsense | RSD not sampled at 1 fps, or state not reset between videos |
| `Could not read the image <video>` | Known issue #1 |
