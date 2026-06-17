#ifndef YOLO_SEG_POSTPROCESSOR_GPU_HPP
#define YOLO_SEG_POSTPROCESSOR_GPU_HPP


#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
// Includes the main Holoscan framework API.
#include <holoscan/holoscan.hpp>

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
class YoloSegPostprocesserGPUOp : public holoscan::Operator 
{
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(YoloSegPostprocesserGPUOp)
        YoloSegPostprocesserGPUOp() = default;
        void setup(holoscan::OperatorSpec& spec) override;
        void start() override;

        void compute(
        holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;
    
        ~YoloSegPostprocesserGPUOp();

    private:

        bool OverlayInitialised{false};
        static constexpr int TEXT_W = 200;
        static constexpr int TEXT_H = 250;

        // Initialize overlay tensor
        void InitTxtROI(void);

        // GPU overlay pointer (The ownership is GXF / Holoscan allocator)
        float* d_overlay{nullptr};


        // Frame count 
        long long count;
        std::string TxtBuf;
        

        holoscan::Parameter<YAML::Node> colour_labels_;
        std::vector<std::string> label_names_;
        std::vector<std::vector<float>> kclass_colours_;    
        float confidence_threshold_;
        float iou_threshold_;
        float mask_threshold_;
        int hight_;
        int width_;
        int inference_resize_height_;
        int inference_resize_width_;
        int num_blocks_;
        
        holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> Mallocator;
        nvidia::gxf::Entity OverlayEntity;
        
        cv::Mat ROITxtMat;
        cv::Mat Rubber_Mat;
        // Pinned host ROI (The ownership is CPU, but mapping to GXF / Holoscan allocator)
        float* h_text_roi{nullptr};
        float* h_rubber_roi{nullptr};


        cudaStream_t cuda_stream;
        nvidia::gxf::Entity in_entity;

        void ReceiveInputs(holoscan::InputContext& op_input,
                            std::shared_ptr<holoscan::Tensor>& predictions,
                            std::shared_ptr<holoscan::Tensor>& masks_seg);

        
        void ClearOverlay();
        void InitClearRoi();
        void SetText(
            cv::Mat& mat,
            float* h_roi_ptr,
            const std::string& text,
            const cv::Point& position,  
            const cv::Scalar& color);

        nvidia::gxf::Entity InitOverlayTensor(
                            holoscan::ExecutionContext& context,
                            std::shared_ptr<holoscan::Allocator> allocator,
                            const std::string& name);


        // =======================
        // Host (mapped pinned)   GPU → CPU zero-copy
        // =======================
        GpuBox* h_kept_boxes_  = nullptr;   // [MAX_BOXES] kept boxes
        int*    h_kept_count_  = nullptr;   // [1] kept count

        // =======================
        // Device
        // =======================
        GpuBox*  d_out_boxes_   = nullptr;
        int*     d_out_count_   = nullptr;
        uint8_t* d_keep_flags_  = nullptr;

        // mapped device pointers (point to host memory)
        GpuBox*  d_kept_boxes_  = nullptr;
        int*     d_kept_count_  = nullptr;
        float*   d_kept_masks_ = nullptr;
        
        // =======================
        // ROI (CPU drawing)
        // =======================
        float*  h_box_roi_ = nullptr;
        cv::Mat BoxRoiMat;



        void DrawKeptBoxesOnOverlay();
        void DrawLegendOnOverlay();

};

#endif // YOLO_SEG_POSTPROCESSOR_GPU_HPP