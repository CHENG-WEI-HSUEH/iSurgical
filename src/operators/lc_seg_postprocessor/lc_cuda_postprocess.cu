#include "lc_cuda_postprocess_api.h"
#include <cuda_runtime.h>
#include <math.h>

__constant__ float d_lc_class_colors[MAX_LC_CLASSES][4];

// Each head is a softmax classifier over (len(group) + 1) channels: channel 0 is the
// head's background and channels 1..N map to the head's global class ids. We decode by
// argmax over the channel axis (matching the reference inference_video.py), not by an
// independent per-channel sigmoid threshold. Heads are rendered sequentially by the host,
// so a later head's non-background pixel overwrites an earlier head's, also matching the
// reference's sequential pred_mask assignment.
__global__ void render_lc_heads_kernel(
    const float* head_logits,      // Shape: [C, Hf, Wf] channel-major (NCHW) at inference resolution
    const int* channel_class_map,  // [num_channels]; index = channel, value = global class id (0 = background)
    int num_channels,              // len(group) + 1
    float* overlay,                // Full RGBA target [Ho, Wo, 4] at system/overlay resolution
    int out_w,
    int out_h,
    int infer_w,
    int infer_h
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_w || y >= out_h) return;

    // Nearest-neighbour map from overlay space down to the head's inference grid. The
    // model output is interpolated to the (square) inference input size, and the format
    // converter resizes without preserving aspect ratio, so this is a direct axis scale
    // (same as the reference cv2.resize(..., INTER_NEAREST)).
    int sx = static_cast<int>((static_cast<long long>(x) * infer_w) / out_w);
    int sy = static_cast<int>((static_cast<long long>(y) * infer_h) / out_h);
    if (sx >= infer_w) sx = infer_w - 1;
    if (sy >= infer_h) sy = infer_h - 1;

    int spatial_stride = infer_w * infer_h;
    int src_idx = sy * infer_w + sx;

    // Argmax over channels (channel 0 = background).
    int best_c = 0;
    float best_v = head_logits[src_idx];
    for (int c = 1; c < num_channels; ++c)
    {
        float v = head_logits[c * spatial_stride + src_idx];
        if (v > best_v) { best_v = v; best_c = c; }
    }

    int global_cls = channel_class_map[best_c];
    if (global_cls <= 0) return; // background wins -> leave whatever an earlier head wrote

    int overlay_idx = (y * out_w + x) * 4;
    overlay[overlay_idx + 0] = d_lc_class_colors[global_cls][0];
    overlay[overlay_idx + 1] = d_lc_class_colors[global_cls][1];
    overlay[overlay_idx + 2] = d_lc_class_colors[global_cls][2];
    overlay[overlay_idx + 3] = 0.2f; // overlay blending alpha
}

void UploadLcClassColorsToGPU(const float* colors, int num_classes)
{
    cudaMemcpyToSymbol(
        d_lc_class_colors,
        colors,
        sizeof(float) * 4 * num_classes
    );
}

void RenderLcHeadsToOverlayGPU(
    const float* d_head_logits,
    const int* d_channel_class_map,
    int num_channels,
    float* d_overlay,
    int out_w,
    int out_h,
    int infer_w,
    int infer_h,
    cudaStream_t stream
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_lc_heads_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_head_logits,
        d_channel_class_map,
        num_channels,
        d_overlay,
        out_w,
        out_h,
        infer_w,
        infer_h
    );
}
