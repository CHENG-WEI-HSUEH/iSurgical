#include "CUDA_Postprocess_API.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

__constant__ float d_class_colors[12][4];

// -------------------------------
// Confidence filter kernel
// -------------------------------
__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x)
{
    float t = clampf((x - edge0) / (edge1 - edge0 + 1e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Bilinear sample from FULL decoded mask space (160x160)
__device__ __forceinline__ float sample_mask_bilinear_full(
    const float* mask,
    float mx,
    float my)
{
    mx = clampf(mx, 0.0f, 159.0f);
    my = clampf(my, 0.0f, 159.0f);

    int x0 = (int)floorf(mx);
    int y0 = (int)floorf(my);
    int x1 = min(x0 + 1, 159);
    int y1 = min(y0 + 1, 159);

    float tx = mx - x0;
    float ty = my - y0;

    float v00 = mask[y0 * 160 + x0];
    float v10 = mask[y0 * 160 + x1];
    float v01 = mask[y1 * 160 + x0];
    float v11 = mask[y1 * 160 + x1];

    float v0 = lerp(v00, v10, tx);
    float v1 = lerp(v01, v11, tx);
    return lerp(v0, v1, ty);
}

__device__ __forceinline__
bool dev_should_merge(
    const GpuBox& a,
    const GpuBox& b,
    float cover_threshold)
{
    if (a.class_id != b.class_id)
        return false;

    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.f, xx2 - xx1);
    float h = fmaxf(0.f, yy2 - yy1);
    float inter = w * h;
    if (inter <= 0.f)
        return false;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    float small_area = fminf(areaA, areaB);
    if (small_area <= 0.f)
        return false;

    float cover_ratio = inter / small_area;
    return cover_ratio >= cover_threshold;
}

__global__ void merge_boxes_kernel(
    GpuBox* boxes,
    uint8_t* keep_flags,
    int* box_count,
    float cover_threshold)
{
    int i = threadIdx.x;
    int n = *box_count;
    if (i >= n) return;
    if (!keep_flags[i]) return;

    GpuBox& curr = boxes[i];

    for (int j = 0; j < n; ++j)
    {
        if (j == i) continue;
        if (!keep_flags[j]) continue;

        GpuBox& other = boxes[j];

        // ensure "bigger eats smaller"
        float area_i = (curr.x2 - curr.x1) * (curr.y2 - curr.y1);
        float area_j = (other.x2 - other.x1) * (other.y2 - other.y1);

        int big   = area_i >= area_j ? i : j;
        int small = area_i >= area_j ? j : i;

        if (big != i) continue; // only let big box absorb

        if (dev_should_merge(boxes[big], boxes[small], cover_threshold))
        {
            // merge inplace
            boxes[big].x1 = fminf(boxes[big].x1, boxes[small].x1);
            boxes[big].y1 = fminf(boxes[big].y1, boxes[small].y1);
            boxes[big].x2 = fmaxf(boxes[big].x2, boxes[small].x2);
            boxes[big].y2 = fmaxf(boxes[big].y2, boxes[small].y2);

            keep_flags[small] = 0; // absorbed
        }
    }
}

__global__ void reset_count_kernel(int* count)
{
    if (threadIdx.x == 0)
        *count = 0;
}

__global__ void confidence_filter_kernel(
    const float* preds,
    int num_preds,
    int feat_dim,
    float conf_thresh,
    GpuBox* out_boxes,
    int* out_count,
    int max_boxes
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_preds) return;

    const float* row = preds + idx * feat_dim;
    float score = row[4];
    if (score < conf_thresh) return;

    int write_idx = atomicAdd(out_count, 1);

    // HARD CLAMP
    if (write_idx >= max_boxes)
        return;

    GpuBox& box = out_boxes[write_idx];
    box.x1 = row[0];
    box.y1 = row[1];
    box.x2 = row[2];
    box.y2 = row[3];
    box.score = score;
    box.class_id = (int)row[5] + 1;

    int mask_offset = feat_dim - 32;
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        box.mask[i] = row[mask_offset + i];
}

__device__ __forceinline__ float dev_iou(const GpuBox& a, const GpuBox& b)
{
    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.f, xx2 - xx1);
    float h = fmaxf(0.f, yy2 - yy1);
    float inter = w * h;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (areaA + areaB - inter + 1e-6f);
}

__global__ void nms_kernel(
    const GpuBox* boxes,
    const int* box_count,
    float iou_thresh,
    uint8_t* keep_flags
)
{
    int i = threadIdx.x;
    int n = *box_count;

    if (i >= n) return;

    const GpuBox& cur = boxes[i];
    uint8_t keep = 1;

    for (int j = 0; j < n; ++j)
    {
        if (j == i) continue;
        const GpuBox& other = boxes[j];

        if (cur.class_id != other.class_id) continue;

        if (other.score > cur.score)
        {
            if (dev_iou(cur, other) > iou_thresh)
            {
                keep = 0;
                break;
            }
        }
    }

    keep_flags[i] = keep;
}

__global__ void compact_kept_boxes_kernel(
    const GpuBox* boxes,        // [max_boxes]
    const uint8_t* keep_flags,  // [max_boxes]
    int max_boxes,
    GpuBox* kept_boxes,         // mapped pinned host
    int* kept_count             // mapped pinned host
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_boxes) return;
    if (!keep_flags[i]) return;

    int out = atomicAdd(kept_count, 1);
    if (out < max_boxes) {
        kept_boxes[out] = boxes[i]; // struct copy (x1,y1,x2,y2,obj,cls,mask[32])
    }
}

/*
__global__ void decode_yolo_masks_kernel(
    const float* proto,        // [32, 25600]
    const GpuBox* kept_boxes,  // [kept_count]
    const int* kept_count,
    float* out_masks           // [kept_count, 25600]
)
{
    int box_id = blockIdx.x;
    int pix    = blockIdx.y * blockDim.x + threadIdx.x;

    if (box_id >= *kept_count || pix >= 25600)
        return;

    float v = 0.f;

    #pragma unroll
    for (int c = 0; c < 32; ++c)
    {
        v += kept_boxes[box_id].mask[c] *
             proto[c * 25600 + pix];
    }

    out_masks[box_id * 25600 + pix] = 1.f / (1.f + expf(-v));
}
*/

__global__ void decode_yolo_masks_kernel(
    const float* proto,        // [32, 25600]
    const GpuBox* kept_boxes,  // [kept_count]
    int kept_count,
    float* out_masks           // [kept_count, 25600]
)
{
    int box_id = blockIdx.x;
    int pix    = blockIdx.y * blockDim.x + threadIdx.x;

    if (box_id >= kept_count || pix >= 25600)
        return;

    float v = 0.f;

    #pragma unroll
    for (int c = 0; c < 32; ++c)
        v += kept_boxes[box_id].mask[c] *
             proto[c * 25600 + pix];

    out_masks[box_id * 25600 + pix] = 1.f / (1.f + expf(-v));
}

__global__ void render_masks_to_overlay_kernel(
    const float* masks,        // [kept_count, 25600] (160x160)
    const GpuBox* boxes,       // boxes in 640x640 space
    int kept_count,
    float* overlay,            // [overlay_h, overlay_w, 4]
    int overlay_w,             // 1920
    int overlay_h,             // 1080
    float mask_thresh
)
{
    int box_id = blockIdx.x;
    int tid    = threadIdx.x;

    if (box_id >= kept_count) return;

    const GpuBox& box = boxes[box_id];
    int cls = box.class_id;
    if (cls <= 0 || cls >= 12) return;

    // ---- box: 640 -> overlay ----
    int x1 = max(0, min((int)(box.x1 * overlay_w / 640.f), overlay_w - 1));
    int y1 = max(0, min((int)(box.y1 * overlay_h / 640.f), overlay_h - 1));
    int x2 = max(0, min((int)(box.x2 * overlay_w / 640.f), overlay_w - 1));
    int y2 = max(0, min((int)(box.y2 * overlay_h / 640.f), overlay_h - 1));

    if (x2 <= x1 || y2 <= y1) return;

    const float* mask = masks + box_id * 25600;

    float r = d_class_colors[cls][0];
    float g = d_class_colors[cls][1];
    float b = d_class_colors[cls][2];
    float a = 0.3f;

    int bw = x2 - x1;
    int bh = y2 - y1;

    for (int i = tid; i < bw * bh; i += blockDim.x)
    {
        int dx = i % bw;
        int dy = i / bw;

        int ox = x1 + dx;
        int oy = y1 + dy;

        // IMPORTANT:
        // Keep FULL-IMAGE mask mapping, not box-relative mapping.
        float mx = ((float)ox + 0.5f) * 160.f / (float)overlay_w - 0.5f;
        float my = ((float)oy + 0.5f) * 160.f / (float)overlay_h - 0.5f;

        float mv = sample_mask_bilinear_full(mask, mx, my);

        // soft edge only
        float edge = smoothstep(mask_thresh - 0.08f, mask_thresh + 0.08f, mv);
        if (edge <= 0.001f) continue;

        int idx = (oy * overlay_w + ox) * 4;

        overlay[idx + 0] = r;
        overlay[idx + 1] = g;
        overlay[idx + 2] = b;
        overlay[idx + 3] = a * edge;
    }
}

// -------------------------------
// Public HOST API
// -------------------------------
void RenderMasksToOverlayGPU(
    const float* d_kept_masks,
    const GpuBox* d_kept_boxes,
    int kept_count,
    float* d_overlay,
    int overlay_w,
    int overlay_h,
    float mask_thresh
)
{
    if (kept_count <= 0) return;

    render_masks_to_overlay_kernel<<<kept_count, 256>>>(
        d_kept_masks,
        d_kept_boxes,
        kept_count,
        d_overlay,
        overlay_w,
        overlay_h,
        mask_thresh
    );
}

void UploadClassColorsToGPU(const float colors[12][4])
{
    cudaMemcpyToSymbol(
        d_class_colors,
        colors,
        sizeof(float) * 12 * 4
    );
}

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
)
{
    if (!d_preds || !d_out_boxes || !d_out_count ||
        !d_keep_flags || !d_kept_boxes || !d_kept_count)
        return -1;

    // 0) reset counters
    reset_count_kernel<<<1,1>>>(d_out_count);
    reset_count_kernel<<<1,1>>>(d_kept_count);

    // 1) confidence filter
    int threads = 256;
    int blocks  = (num_preds + threads - 1) / threads;

    confidence_filter_kernel<<<blocks, threads>>>(
        d_preds,
        num_preds,
        feat_dim,
        conf_thresh,
        d_out_boxes,
        d_out_count,
        max_boxes
    );

    // 2) NMS
    nms_kernel<<<1, max_boxes>>>(
        d_out_boxes,
        d_out_count,
        iou_thresh,
        d_keep_flags
    );

    // 3) merge
    merge_boxes_kernel<<<1, max_boxes>>>(
        d_out_boxes,
        d_keep_flags,
        d_out_count,
        0.6f
    );

    // 4) compact → mapped pinned host
    compact_kept_boxes_kernel<<<1, max_boxes>>>(
        d_out_boxes,
        d_keep_flags,
        max_boxes,
        d_kept_boxes,
        d_kept_count
    );

    // 5) YOLO mask decode (GPU only)
    int kept_count_host;
    cudaMemcpy(&kept_count_host, d_kept_count,
               sizeof(int), cudaMemcpyDeviceToHost);

    if (kept_count_host > 0)
    {
        dim3 mask_threads(256);
        dim3 mask_blocks(
            kept_count_host,
            (25600 + mask_threads.x - 1) / mask_threads.x
        );

        decode_yolo_masks_kernel<<<mask_blocks, mask_threads>>>(
            d_mask_proto,
            d_kept_boxes,
            kept_count_host,
            d_kept_masks
        );
    }
    return 0;
}