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

static const std::string LabelNames[12] = 
{
    "Background",
    "Spleen",
    "Left_Kidney",
    "Left_Renal_Artery",
    "Left_Renal_Vein",
    "Left_Ureter",
    "Left_Lumbar_Vein",
    "Left_Adrenal_Vein",
    "Left_Gonadal_Vein",
    "Psoas_Muscle",
    "Colon",
    "Abdominal_Aorta",
};

static constexpr float kClassColors[12][4] = {
    {0.0f,    0.0f,    0.0f,    0.0f},   // Background
    {0.1451f, 0.9412f, 0.6157f, 0.2f},   // Spleen
    {0.0098f, 0.3176f, 0.9410f, 0.2f},   // Left_Kidney
    {1.0f,    0.8039f, 0.1529f, 0.2f},   // Left_Renal_Artery
    {0.0039f, 0.9373f, 1.0f,    0.2f},   // Left_Renal_Vein
    {0.9569f, 0.9019f, 0.1569f, 0.2f},   // Left_Ureter
    {0.0157f, 0.4549f, 0.4509f, 0.0f},   // Left_Lumbar_Vein
    {0.8941f, 0.5647f, 0.0706f, 0.0f},   // Left_Adrenal_Vein
    {0.5019f, 0.1059f, 0.4471f, 0.0f},   // Left_Gonadal_Vein
    {1.0f,    1.0f,    1.0f,    0.2f},   // Psoas_Muscle
    {0.4314f, 0.4863f, 1.0f,    0.0f},   // Colon
    {0.6784f, 0.4941f, 0.2745f, 0.0f}    // Abdominal_Aorta
};

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

    spec.param(this->ScoreThredshold, "ScoreThredshold", "Scores threshold");
    spec.param(this->NumClass, "NumClass", "Number of classes");
    spec.param(this->Mallocator, "MemoryAllocator", "Memory pool allocator");

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

    UploadClassColorsToGPU(kClassColors);

    // -------------------------
    // CPU drawing buffers
    // -------------------------
    cudaMallocHost(&this->h_box_roi_, 1920 * 1080 * 4 * sizeof(float));
    this->BoxRoiMat = cv::Mat(1080, 1920, CV_32FC4, this->h_box_roi_);
    this->BoxRoiMat.setTo(cv::Scalar(0,0,0,0));

    InitTxtROI();
    InitClearRoi();

    this->count = 0;
}

void YoloSegPostprocesserGPUOp::compute(holoscan::InputContext& op_input,holoscan::OutputContext& op_output,holoscan::ExecutionContext& context)
{
    // Trace for NVIDIA Nsight Systems
    nvtxRangePush("YoloSegPostprocesserGPUOp::compute");



    // OverlayInitialised
    if (!this->OverlayInitialised)
    {
        // Initialise overlay tensor (Don't move to setup/constructor, as it requires ExecutionContext)
        this->OverlayEntity = InitOverlayTensor(context,1920,1080,this->Mallocator,"OverlayTensor");

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
    0.25f, // confidence threshold
    0.5f, // IOU threshold
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
        1920,
        1080,
        0.2f
    );

    /*
    // Step 6. (Optional) Show the frame count
    this->TxtBuf = "After: " + std::to_string(this->count);
    SetText(
        this->ROITxtMat,
        this->h_text_roi,
        this->TxtBuf,
        cv::Point(1920 - this->TEXT_W, 1080 - this->TEXT_H - 250),  
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
                                            int width,
                                            int height,
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
    nvidia::gxf::Shape shape{height, width, 4};

    // 5. Reshape tensor
    tensor.value()->reshape<float>(
        shape,
        nvidia::gxf::MemoryStorageType::kDevice,
        gxf_allocator.value()
    );

    // 6. Get GPU pointer and store in d_overlay
    this->d_overlay = tensor.value()->data<float>().value();

    // 7. Initialise overlay to zeros
    cudaMemset(this->d_overlay, 0, width * height * 4 * sizeof(float));

    // 8. Return entity
    return holoscan::gxf::Entity(std::move(entity.value()));
}

void YoloSegPostprocesserGPUOp::InitTxtROI(void)
{
    // 1. Calculate total bytes
    size_t total_bytes = TEXT_W * TEXT_H * 4 * sizeof(float);

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
    cudaMallocHost(&this->h_rubber_roi, TEXT_W * TEXT_H * 4 * sizeof(float));

    // wrap as CV_32FC4 (RGBA float)
    this->Rubber_Mat = cv::Mat(TEXT_H, TEXT_W, CV_32FC4, this->h_rubber_roi);

    // fill transparent once
    this->Rubber_Mat.setTo(cv::Scalar(0,0,0,0));
}


void YoloSegPostprocesserGPUOp::ClearOverlay()
{
    cudaMemset(this->d_out_count_,   0, sizeof(int));
    cudaMemset(this->d_kept_count_,  0, sizeof(int));
    cudaMemset(this->d_keep_flags_,  0, MAX_BOXES * sizeof(uint8_t));
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
        2.0f,          // Thickness
        cv::LINE_AA // Line type
    );

    // Copy ROI → GPU overlay (async, pinned) Full screen position
    int roi_x = std::clamp(position.x, 0, 1920 - this->TEXT_W);
    int roi_y = std::clamp(position.y, 0, 1080 - this->TEXT_H);

    size_t dst_pitch = 1920 * 4 * sizeof(float);
    size_t src_pitch = TEXT_W * 4 * sizeof(float);

    float* d_dst = this->d_overlay + (roi_y * 1920 + roi_x) * 4;

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

    cv::RNG rng(cv::getTickCount());
    this->BoxRoiMat.setTo(cv::Scalar(0, 0, 0, 0));
    
    int kept = *this->h_kept_count_;

    for (int i = 0; i < kept; ++i)
    {
        const GpuBox& b = this->h_kept_boxes_[i];

        int x1 = std::clamp((int)(b.x1 / 640 * 1920), 0, 1920 - 1);
        int y1 = std::clamp((int)(b.y1 / 640 * 1080), 0, 1080 - 1);
        int x2 = std::clamp((int)(b.x2 / 640 * 1920), 0, 1920 - 1);
        int y2 = std::clamp((int)(b.y2 / 640 * 1080), 0, 1080 - 1);
        float score = b.score;
        int   cls_id   = b.class_id;

        cv::Scalar color(
            kClassColors[cls_id][0],
            kClassColors[cls_id][1],
            kClassColors[cls_id][2],
            1.0f
        );

        

            // 1) draw bounding box
            cv::rectangle(
                this->BoxRoiMat,
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                color,
                2,
                cv::LINE_AA
            );
        


            // 2) draw label (left-top of box)
            const std::string& label = LabelNames[cls_id] + "," + std::to_string(score * 100.0f).substr(0, 5) + "%";

            int text_x = x1;
            int text_y = std::max(12, y2 + 20); // Ensure text is within image bounds

            cv::putText(
                this->BoxRoiMat,
                label,
                cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,        // font scale
                cv::Scalar(1.0, 1.0, 1.0, 1.0), // white color
                2,          // thickness
                cv::LINE_AA
            );
        }
    

    // copy ROI → GPU overlay
    cudaMemcpy2DAsync(
        this->d_overlay,
        1920 * 4 * sizeof(float),
        this->h_box_roi_,
        1920 * 4 * sizeof(float),
        1920 * 4 * sizeof(float),
        1080,
        cudaMemcpyHostToDevice,
        this->cuda_stream
    );
    
}

YoloSegPostprocesserGPUOp::~YoloSegPostprocesserGPUOp()
{
    if (this->h_text_roi) cudaFreeHost(this->h_text_roi);
    if (this->h_rubber_roi) cudaFreeHost(this->h_rubber_roi);
    if (this->h_kept_boxes_) cudaFreeHost(this->h_kept_boxes_);
    if (this->h_kept_count_)        cudaFreeHost(this->h_kept_count_);
}