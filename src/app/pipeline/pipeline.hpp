
#ifndef  _INFERENCE_PIPELINE_HPP_
#define  _INFERENCE_PIPELINE_HPP_

// Includes the main Holoscan framework API.
#include <holoscan/holoscan.hpp>  
#include <yaml-cpp/yaml.h>
class InferencePipeline : public holoscan::Application {
  public:
    InferencePipeline(const YAML::Node& config);
    void compose() override;

  private:
    std::shared_ptr<holoscan::Operator> video_source_op_;
    std::shared_ptr<holoscan::Operator> pre_processor_op_;
    std::shared_ptr<holoscan::Operator> segmentation_inference_op_;
    std::shared_ptr<holoscan::Operator> post_processor_op_;
    std::shared_ptr<holoscan::Operator> result_visualizer_op_;

    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool_;

    YAML::Node config_;
};

#endif //  "_INFERENCE_PIPELINE_HPP_"