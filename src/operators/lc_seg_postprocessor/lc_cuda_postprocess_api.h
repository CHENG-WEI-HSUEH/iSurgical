#pragma once

#ifdef __cplusplus
extern "C" {
#endif

constexpr int MAX_LC_CLASSES = 32;

/**
 * @brief Transfers target RGB colors into device memory blocks.
 */
void UploadLcClassColorsToGPU(const float* colors, int num_classes);

/**
 * @brief Argmax-decodes one segmentation head onto the RGBA overlay.
 *
 * @param d_head_logits      Head logits [num_channels, infer_h, infer_w], channel-major (NCHW).
 * @param d_channel_class_map [num_channels] mapping channel index -> global class id (0 = background).
 * @param num_channels       len(group) + 1 (channel 0 is the head background).
 * @param d_overlay          RGBA overlay [out_h, out_w, 4] at system/overlay resolution.
 * @param out_w,out_h        Overlay (system) dimensions.
 * @param infer_w,infer_h    Head logit dimensions (model inference resize size).
 */
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
);

#ifdef __cplusplus
}
#endif
