#ifndef LC_SEG_POSTPROCESSOR_GPU_HPP
#define LC_SEG_POSTPROCESSOR_GPU_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/core/gxf/gxf_operator.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/core/execution_context.hpp>
#include <gxf/cuda/cuda_stream.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include "lc_cuda_postprocess_api.h"

class LcSegPostprocesserGPUOp : public holoscan::Operator 
{
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(LcSegPostprocesserGPUOp)
        LcSegPostprocesserGPUOp() = default;
        void setup(holoscan::OperatorSpec& spec) override;
        void start() override;
        void stop() override;

        void compute(
            holoscan::InputContext& op_input,
            holoscan::OutputContext& op_output,
            holoscan::ExecutionContext& context) override;
    
        ~LcSegPostprocesserGPUOp();

    private:
        bool OverlayInitialised{false};

        // GPU overlay pointer 
        float* d_overlay{nullptr};

        holoscan::Parameter<YAML::Node> colour_labels_;
        std::vector<std::string> label_names_;
        std::vector<std::vector<float>> kclass_colours_;    
        
        float mask_threshold_;
        int height_;
        int width_;
        int inference_resize_width_;  // head logit width  (model inference input size)
        int inference_resize_height_; // head logit height (model inference input size)
        int num_blocks_; // typically 4 for RGBA
        
        // Multi-head tracking configurations
        std::vector<std::vector<int>> head_groups_;
        int num_heads_;

        holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> Mallocator;
        nvidia::gxf::Entity OverlayEntity;
        
        cv::Mat BoxRoiMat;
        float* h_box_roi_{nullptr}; // Pinned host memory for background canvas/legend

        cudaStream_t cuda_stream{nullptr};

        void ReceiveInputs(holoscan::InputContext& op_input,
                           std::vector<std::shared_ptr<holoscan::Tensor>>& head_tensors);

        void ClearOverlay();
        
        nvidia::gxf::Entity InitOverlayTensor(
                            holoscan::ExecutionContext& context,
                            std::shared_ptr<holoscan::Allocator> allocator,
                            const std::string& name);

        // Draws the legend card into BoxRoiMat and returns the panel rectangle (clamped to
        // the frame) so the caller can blit just that region on top of the masks.
        cv::Rect DrawLegendOnOverlay();
};

#endif // LC_SEG_POSTPROCESSOR_GPU_HPP
