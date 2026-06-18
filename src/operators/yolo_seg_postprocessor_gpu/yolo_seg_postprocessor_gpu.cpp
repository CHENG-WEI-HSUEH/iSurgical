#include "yolo_seg_postprocessor_gpu.hpp"


// Includes the Holoviz operator for GPU-accelerated visualization.
#include <holoscan/operators/holoviz/holoviz.hpp>

// Includes the segmentation postprocessor for converting inference output into bounding boxes or masks.
#include <array>

// Includes standard C++ library for string operations.
#include <string>

// Includes standard C++ library for unordered map operations.
#include <unordered_map>

// Includes standard C++ library for vector operations.
#include <vector>
#include <queue>
#include <holoscan/core/execution_context.hpp>
#include <gxf/cuda/cuda_stream.hpp>

#include <holoscan/core/gxf/gxf_component.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/core/condition.hpp>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <tuple>
#include <nvtx3/nvToolsExt.h>

#include "CUDA_Postprocess_API.h"

constexpr int BOX_ROI_W = 640;
constexpr int BOX_ROI_H = 640;
constexpr int MAX_BOXES = 64;

static inline bool ComputeUnionROI(
    const int* kept_indices, int kept,
    const GpuBox* boxes,
    int img_w, int img_h,
    int pad,
    int& roi_x, int& roi_y, int& roi_w, int& roi_h)
{
    int minx = img_w, miny = img_h, maxx = 0, maxy = 0;
    bool any = false;

    for (int k = 0; k < kept; ++k) {
        int idx = kept_indices[k];
        if (idx < 0) continue;

        const GpuBox& b = boxes[idx];
        int x1 = std::clamp((int)std::floor(b.x1), 0, img_w - 1);
        int y1 = std::clamp((int)std::floor(b.y1), 0, img_h - 1);
        int x2 = std::clamp((int)std::ceil (b.x2), 0, img_w - 1);
        int y2 = std::clamp((int)std::ceil (b.y2), 0, img_h - 1);

        minx = std::min(minx, x1);
        miny = std::min(miny, y1);
        maxx = std::max(maxx, x2);
        maxy = std::max(maxy, y2);
        any = true;
    }

    if (!any) return false;

    minx = std::max(0, minx - pad);
    miny = std::max(0, miny - pad);
    maxx = std::min(img_w - 1, maxx + pad);
    maxy = std::min(img_h - 1, maxy + pad);

    roi_x = minx;
    roi_y = miny;
    roi_w = (maxx - minx + 1);
    roi_h = (maxy - miny + 1);

    return (roi_w > 0 && roi_h > 0);
}


void YoloSegPostprocesserGPUOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<nvidia::gxf::Entity>("in");
    spec.output<nvidia::gxf::Entity>("out");

    spec.param(this->Mallocator, "MemoryAllocator", "Memory pool allocator");
    spec.param(this->colour_labels_, "colour_labels", "Colour labels for classes");
}

void YoloSegPostprocesserGPUOp::start()
{
    const YAML::Node& colour_labels = this->colour_labels_.get();
    this->label_names_ = colour_labels["label_names"].as<std::vector<std::string>>();
    this->kclass_colours_ = colour_labels["kclass_colours"].as<std::vector<std::vector<float>>>();
    this->confidence_threshold_ = colour_labels["ConfidenceThreshold"].as<float>();
    this->iou_threshold_ = colour_labels["IoUThreshold"].as<float>();
    this->mask_threshold_ = colour_labels["MaskThreshold"].as<float>();
    this->hight_ = colour_labels["Sys_Height"].as<int>();
    this->width_ = colour_labels["Sys_Width"].as<int>();
    this->inference_resize_height_ = colour_labels["InferenceResizeHeight"].as<int>();
    this->inference_resize_width_ = colour_labels["InferenceResizeWidth"].as<int>();
    this->num_blocks_ = colour_labels["NumBlocks"].as<int>();

    // Dedicated non-default stream: all async GPU work in this operator (ClearOverlay,
    // SetText, DrawKeptBoxesOnOverlay) must run off the legacy default stream (0), which
    // implicitly synchronizes with every other stream in the process and can stall/deadlock
    // teardown with the other operators' cuda_stream_pool-backed streams.
    cudaStreamCreate(&this->cuda_stream);

    // -------------------------
    // GPU output buffers
    // -------------------------
    cudaMalloc(&this->d_out_boxes_,  MAX_BOXES * sizeof(GpuBox));
    cudaMalloc(&this->d_out_count_,  sizeof(int));
    cudaMalloc(&this->d_keep_flags_, MAX_BOXES * sizeof(uint8_t));

    // -------------------------
    // MAPPED PINNED OUTPUT (GPU → CPU zero copy)
    // -------------------------

    // -------------------------
    // MAPPED PINNED (GPU → CPU zero-copy)
    // -------------------------
    cudaHostAlloc(
        (void**)&this->h_kept_count_,
        sizeof(int),
        cudaHostAllocMapped | cudaHostAllocPortable
    );
    cudaHostGetDevicePointer(
        (void**)&this->d_kept_count_,
        (void*)this->h_kept_count_,
        0
    );

    *this->h_kept_count_ = 0;

    cudaHostAlloc(
        (void**)&this->h_kept_boxes_,
        MAX_BOXES * sizeof(GpuBox),
        cudaHostAllocMapped | cudaHostAllocPortable
    );
    cudaHostGetDevicePointer(
        (void**)&this->d_kept_boxes_,
        (void*)this->h_kept_boxes_,
        0
    );

    cudaMalloc(
        &this->d_kept_masks_,
        MAX_BOXES * 25600 * sizeof(float)
    );

    {
        int num_classes = static_cast<int>(this->kclass_colours_.size());
        std::vector<float> flat_colours(num_classes * 4, 1.0f);
        for (int c = 0; c < num_classes; ++c) {
            const auto& colour = this->kclass_colours_[c];
            for (size_t ch = 0; ch < colour.size() && ch < 4; ++ch) {
                flat_colours[c * 4 + ch] = colour[ch];
            }
        }
        UploadClassColorsToGPU(flat_colours.data(), num_classes);
    }

    // -------------------------
    // CPU drawing buffers
    // -------------------------
    cudaMallocHost(&this->h_box_roi_, this->width_ * this->hight_ * this->num_blocks_ * sizeof(float));
    this->BoxRoiMat = cv::Mat(this->hight_, this->width_, CV_32FC4, this->h_box_roi_);
    this->BoxRoiMat.setTo(cv::Scalar(0,0,0,0));

    InitTxtROI();
    InitClearRoi();

    this->count = 0;
}

void YoloSegPostprocesserGPUOp::stop()
{
    if (this->cuda_stream) {
        cudaStreamSynchronize(this->cuda_stream);
    }

    // Release the pool-backed overlay tensor now (decrements the GXF entity refcount and
    // frees it back to the BlockMemoryPool) instead of leaving it for this object's
    // destructor, which would run after gxf_executor starts destroying the context and
    // produce "BlockMemoryPool still has unreleased memory" / hangs on teardown.
    this->OverlayEntity = nvidia::gxf::Entity();
    this->OverlayInitialised = false;
    this->d_overlay = nullptr;

    if (this->d_out_boxes_)  { cudaFree(this->d_out_boxes_);  this->d_out_boxes_ = nullptr; }
    if (this->d_out_count_)  { cudaFree(this->d_out_count_);  this->d_out_count_ = nullptr; }
    if (this->d_keep_flags_) { cudaFree(this->d_keep_flags_); this->d_keep_flags_ = nullptr; }
    if (this->d_kept_masks_) { cudaFree(this->d_kept_masks_); this->d_kept_masks_ = nullptr; }

    // d_kept_boxes_ / d_kept_count_ are device pointers mapped from pinned host memory
    // (h_kept_boxes_ / h_kept_count_) via cudaHostGetDevicePointer, not separate
    // allocations — must not be cudaFree'd. The host allocations are released below.
    this->d_kept_boxes_ = nullptr;
    this->d_kept_count_ = nullptr;

    if (this->h_box_roi_)    { cudaFreeHost(this->h_box_roi_);    this->h_box_roi_ = nullptr; }
    if (this->h_text_roi)    { cudaFreeHost(this->h_text_roi);    this->h_text_roi = nullptr; }
    if (this->h_rubber_roi)  { cudaFreeHost(this->h_rubber_roi);  this->h_rubber_roi = nullptr; }
    if (this->h_kept_boxes_) { cudaFreeHost(this->h_kept_boxes_); this->h_kept_boxes_ = nullptr; }
    if (this->h_kept_count_) { cudaFreeHost(this->h_kept_count_); this->h_kept_count_ = nullptr; }

    if (this->cuda_stream) {
        cudaStreamDestroy(this->cuda_stream);
        this->cuda_stream = nullptr;
    }
}

void YoloSegPostprocesserGPUOp::compute(holoscan::InputContext& op_input,holoscan::OutputContext& op_output,holoscan::ExecutionContext& context)
{
    // Trace for NVIDIA Nsight Systems
    nvtxRangePush("YoloSegPostprocesserGPUOp::compute");



    // OverlayInitialised
    if (!this->OverlayInitialised)
    {
        // Initialise overlay tensor (Don't move to setup/constructor, as it requires ExecutionContext)
        this->OverlayEntity = InitOverlayTensor(context,this->Mallocator,"OverlayTensor");

        this->OverlayInitialised = true;
    }

    // Step 0. Count the frames
    // this->count++;
    // HOLOSCAN_LOG_INFO("YoloSegPostprocesserOp compute count: {}", this->count);
    

    // Step 1. Clear overlay
    ClearOverlay();


    // Step 2. Receive inputs
    std::shared_ptr<holoscan::Tensor> Predictions, MasksSeg;
    ReceiveInputs(op_input, Predictions, MasksSeg);

    // Step 3. Confidence filtering -> IOU -> NMS -> Fillcountours -> Gaussian smoothing
    const float* gpu_pred_ptr = static_cast<const float*>(Predictions->data());
    const float* gpu_mask_ptr = static_cast<const float*>(MasksSeg->data());


    ConfidenceFilterAndNmsGPU(
    gpu_pred_ptr,
    8400, // 8400 = 20*20 + 40*40 + 80*80
    38, // feat_dim 32 + 5 (for x,y,w,h,score) + 1 (class_logit)
    this->confidence_threshold_, // confidence threshold
    this->iou_threshold_, // IOU threshold
    this->d_out_boxes_,       // caller-allocated, size max_boxes (32)
    this->d_out_count_,      // caller-allocated, size 1
    this->d_keep_flags_, // caller-allocated, size max_boxes (32)
    this->d_kept_boxes_,
    this->d_kept_count_,
    gpu_mask_ptr,          
    this->d_kept_masks_,   
    MAX_BOXES);         // 64

    // Step 4. Draw kept boxes on overlay
    DrawKeptBoxesOnOverlay();
    
    // Step 5. Render masks to overlay
    RenderMasksToOverlayGPU(
        this->d_kept_masks_,
        this->d_kept_boxes_,
        *this->h_kept_count_,
        this->d_overlay,
        this->width_,
        this->hight_,
        this->mask_threshold_
    );

    /*
    // Step 6. (Optional) Show the frame count
    this->TxtBuf = "After: " + std::to_string(this->count);
    SetText(
        this->ROITxtMat,
        this->h_text_roi,
        this->TxtBuf,
        cv::Point(this->width_ - this->TEXT_W, this->hight_ - this->TEXT_H - 250),  
        cv::Scalar(1.0, 1.0, 1.0, 1.0) // RGBA
    );*/

    // --- emit result to Holoviz ---
    op_output.emit(this->OverlayEntity, "out");

    // End trace for NVIDIA Nsight Systems
    nvtxRangePop();
}


void YoloSegPostprocesserGPUOp::ReceiveInputs(holoscan::InputContext& op_input,
                                           std::shared_ptr<holoscan::Tensor>& predictions,
                                           std::shared_ptr<holoscan::Tensor>& masks_seg) 
{
    auto InMsg = op_input.receive<holoscan::TensorMap>("in");
    auto& InTensors = InMsg.value();
    
    // Try to get tensors by name - use operator[] instead of get()
    if (InTensors.find("input_0") != InTensors.end()) 
        predictions = InTensors["input_0"];
    else if (InTensors.find("outputs") != InTensors.end()) 
        predictions = InTensors["outputs"];
    else 
        throw std::runtime_error("Missing 'input_0' or 'outputs' tensor for predictions");
    
    
    if (InTensors.find("input_1") != InTensors.end()) 
        masks_seg = InTensors["input_1"];
    else if (InTensors.find("proto") != InTensors.end()) 
        masks_seg = InTensors["proto"];
    else 
        throw std::runtime_error("Missing 'input_1' or 'proto' tensor for masks");
    
    
    if (!predictions || !masks_seg) 
        throw std::runtime_error("Failed to receive input tensors");
    

}

nvidia::gxf::Entity YoloSegPostprocesserGPUOp::InitOverlayTensor(
                                            holoscan::ExecutionContext& context,
                                            std::shared_ptr<holoscan::Allocator> allocator,
                                            const std::string& name)
{
    // 1. Create new GXF Entity from Holoscan ExecutionContext
    auto entity = nvidia::gxf::Entity::New(context.context());

    // 2. Add tensor component 
    auto tensor = entity.value().add<nvidia::gxf::Tensor>(name.c_str());

    // 3. Convert holoscan::Allocator → gxf::Allocator handle
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator->gxf_cid());

    // 4. Reshape: NHWC = (H, W, 4)
    nvidia::gxf::Shape shape{this->hight_, this->width_, this->num_blocks_}; // 1080, 1920, 4 (RGBA)

    // 5. Reshape tensor
    tensor.value()->reshape<float>(
        shape,
        nvidia::gxf::MemoryStorageType::kDevice,
        gxf_allocator.value()
    );

    // 6. Get GPU pointer and store in d_overlay
    this->d_overlay = tensor.value()->data<float>().value();

    // 7. Initialise overlay to zeros
    cudaMemset(this->d_overlay, 0, this->width_ * this->hight_ * this->num_blocks_ * sizeof(float));

    // 8. Return entity
    return holoscan::gxf::Entity(std::move(entity.value()));
}

void YoloSegPostprocesserGPUOp::InitTxtROI(void)
{
    // 1. Calculate total bytes
    size_t total_bytes = TEXT_W * TEXT_H * this->num_blocks_ * sizeof(float);

    // 2. pinned host allocation
    cudaHostAlloc(
        &this->h_text_roi,
        total_bytes,
        cudaHostAllocDefault
    );



    // 3. Create OpenCV Mat for ROI text drawing
    this->ROITxtMat = cv::Mat(this->TEXT_H, this->TEXT_W, CV_32FC4, this->h_text_roi);


    // 4. Initialise to zeros
    memset(this->h_text_roi, 0, total_bytes);

}

void YoloSegPostprocesserGPUOp::InitClearRoi()
{
    // allocate pinned host
    cudaMallocHost(&this->h_rubber_roi, TEXT_W * TEXT_H * this->num_blocks_ * sizeof(float));

    // wrap as CV_32FC4 (RGBA float)
    this->Rubber_Mat = cv::Mat(TEXT_H, TEXT_W, CV_32FC4, this->h_rubber_roi);

    // fill transparent once
    this->Rubber_Mat.setTo(cv::Scalar(0,0,0,0));
}


void YoloSegPostprocesserGPUOp::ClearOverlay()
{
    // 1. Reset device and host zero-copy counters (Your current code)
    cudaMemsetAsync(this->d_out_count_,   0, sizeof(int), this->cuda_stream);
    cudaMemsetAsync(this->d_kept_count_,  0, sizeof(int), this->cuda_stream);
    cudaMemsetAsync(this->d_keep_flags_,  0, MAX_BOXES * sizeof(uint8_t), this->cuda_stream);

    // 2. FIX: Completely blank out the actual GPU overlay mask tensor (H * W * 4 channels)
    if (this->d_overlay != nullptr) {
        size_t total_bytes = this->width_ * this->hight_ * this->num_blocks_ * sizeof(float);
        cudaMemsetAsync(this->d_overlay, 0, total_bytes, this->cuda_stream);
    }
}

void YoloSegPostprocesserGPUOp::SetText(
                                cv::Mat& mat,
                                float* h_roi_ptr,
                                const std::string& text,
                                const cv::Point& position,  
                                const cv::Scalar& color)
{
    // Clear previous text
    mat.setTo(cv::Scalar(0,0,0,0));

    // CPU OpenCV draw (ROI only)
    cv::putText(
        mat,
        text,
        cv::Point(2, TEXT_H - 2),   // ROI Position
        cv::FONT_HERSHEY_SIMPLEX,
        1.0f,       // Font scale
        color,      // RGBA
        2.0f,       // Thickness
        cv::LINE_AA // Line type
    );

    // Copy ROI → GPU overlay (async, pinned) Full screen position
    int roi_x = std::clamp(position.x, 0, this->width_ - this->TEXT_W);
    int roi_y = std::clamp(position.y, 0, this->hight_ - this->TEXT_H);

    size_t dst_pitch = this->width_ * this->num_blocks_ * sizeof(float);
    size_t src_pitch = this->TEXT_W * this->num_blocks_ * sizeof(float);

    float* d_dst = this->d_overlay + (roi_y * this->width_ + roi_x) * this->num_blocks_;

    cudaMemcpy2DAsync(
        d_dst,      //Destination pointer
        dst_pitch,  //Destination pitch (in bytes)
        h_roi_ptr, //Source pointer
        src_pitch,  //Source pitch (in bytes)
        src_pitch,  //Width (in bytes)
        this->TEXT_H,     //Height
        cudaMemcpyHostToDevice,//Type of transfer
        this->cuda_stream //CUDA stream
    );


}


void YoloSegPostprocesserGPUOp::DrawKeptBoxesOnOverlay()
{
    this->BoxRoiMat.setTo(cv::Scalar(0, 0, 0, 0));

    // Bounding boxes and per-detection text labels are intentionally not drawn;
    // only the segmentation masks (rendered separately on the GPU) and the
    // static class legend are shown on the overlay.

    // draw the legend menu of all configured classes
    try {
        DrawLegendOnOverlay();
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("DrawLegendOnOverlay threw: {}", e.what());
    }

    // copy ROI → GPU overlay
    cudaMemcpy2DAsync(
        this->d_overlay,
        this->width_ * this->num_blocks_ * sizeof(float),
        this->h_box_roi_,
        this->width_ * this->num_blocks_ * sizeof(float),
        this->width_ * this->num_blocks_ * sizeof(float),
        this->hight_,
        cudaMemcpyHostToDevice,
        this->cuda_stream
    );
    
}

void YoloSegPostprocesserGPUOp::DrawLegendOnOverlay() {
    // Show every configured class (skipping "Background") regardless of what's
    // currently detected, so the legend is static rather than flickering per-frame.
    std::vector<int> active_classes;
    for (int cls_id = 1; cls_id < static_cast<int>(this->label_names_.size()); ++cls_id) {
        active_classes.push_back(cls_id);
    }

    if (active_classes.empty()) return;

    // 2. Geometry configuration for the legend menu panel (2 columns)
    int item_height = 30;
    int padding = 15;
    int color_box_size = 20;
    int columns = 2;
    int col_width = 290;
    int col_gap = 20;

    int num_items = static_cast<int>(active_classes.size());
    int rows = (num_items + columns - 1) / columns;

    int legend_w = (padding * 2) + (col_width * columns) + (col_gap * (columns - 1));
    int legend_h = (rows * item_height) + (padding * 2);

    // Position legend box at the top right corner
    int margin = 20;
    int start_x = this->width_ - legend_w - margin;
    int start_y = margin;

    // 3. Draw a semi-transparent dark background card panel for readability
    cv::Rect panel_rect(start_x, start_y, legend_w, legend_h);
    // Mix overlay background with a 60% opacity dark tone
    cv::Mat panel_roi = this->BoxRoiMat(panel_rect);
    cv::Mat color_card(panel_roi.size(), panel_roi.type(), cv::Scalar(0.0f, 0.0f, 0.0f, 0.6f));
    cv::addWeighted(color_card, 0.6, panel_roi, 0.4, 0.0, panel_roi);

    // 4. Populate rows with corresponding labels and color swatches, filling left-to-right then top-to-bottom
    for (int current_item = 0; current_item < num_items; ++current_item) {
        int cls_id = active_classes[current_item];
        int row = current_item / columns;
        int col = current_item % columns;

        int col_x = start_x + padding + col * (col_width + col_gap);
        int row_y = start_y + padding + (row * item_height);

        // Map colors (Ensure values scale into OpenCV's floating-point system)
        cv::Scalar swatch_color(
            this->kclass_colours_[cls_id][0],
            this->kclass_colours_[cls_id][1],
            this->kclass_colours_[cls_id][2],
            1.0f
        );

        // Draw color rectangle block indicator
        cv::Rect color_box(col_x, row_y + 4, color_box_size, color_box_size);
        cv::rectangle(this->BoxRoiMat, color_box, swatch_color, cv::FILLED);
        cv::rectangle(this->BoxRoiMat, color_box, cv::Scalar(1.0f, 1.0f, 1.0f, 1.0f), 1); // white border

        // Fix missing character labels inside display strings manually if necessary
        std::string display_name = this->label_names_[cls_id];
        if (display_name == "Hepatoduodenal ligamen") {
            display_name = "Hepatoduodenal ligament";
        }

        // Render matching label text alongside the box
        cv::Point text_pos(col_x + color_box_size + 12, row_y + 20);
        cv::putText(
            this->BoxRoiMat,
            display_name,
            text_pos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,                             // Font scale
            cv::Scalar(1.0f, 1.0f, 1.0f, 1.0f), // White text
            1,                               // Thickness
            cv::LINE_AA
        );
    }
}

YoloSegPostprocesserGPUOp::~YoloSegPostprocesserGPUOp()
{
    // Normal teardown releases everything in stop(), which Holoscan guarantees runs
    // before the destructor. These are a fallback only for the case where the operator
    // is destroyed without having been started (e.g. construction failure).
    if (this->h_text_roi) cudaFreeHost(this->h_text_roi);
    if (this->h_rubber_roi) cudaFreeHost(this->h_rubber_roi);
    if (this->h_kept_boxes_) cudaFreeHost(this->h_kept_boxes_);
    if (this->h_kept_count_) cudaFreeHost(this->h_kept_count_);
}