#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

constexpr int MAX_CLASS_COLORS = 32;

// GPU / Host shared structure
struct GpuBox
{
    float x1, y1, x2, y2;   // box coords (pixel)
    float score;            // confidence
    int   class_id;         // class id
    float mask[32];         // segmentation coeffs
};

void UploadClassColorsToGPU(const float* colors, int num_classes);

void RenderMasksToOverlayGPU(
    const float* d_kept_masks,
    const GpuBox* d_kept_boxes,
    int kept_count,
    float* d_overlay,
    int overlay_w,
    int overlay_h,
    float mask_thresh
);

int ConfidenceFilterAndNmsGPU
(
    const float* d_preds,
    int num_preds,
    int feat_dim,
    float conf_thresh,
    float iou_thresh,
    GpuBox* d_out_boxes,
    int* d_out_count,
    uint8_t* d_keep_flags,
    GpuBox* d_kept_boxes,
    int* d_kept_count,
    const float* d_mask_proto,  
    float* d_kept_masks,        
    int max_boxes
);        

#ifdef __cplusplus
}
#endif