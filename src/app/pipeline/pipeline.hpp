
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
    void ConfigureSourceOperator();
    void ConfigurePreProcessorOperator();
    void ConfigureInferenceOperator();
    void ConfigurePostProcessorOperator();
    void ConfigureVisualizerOperator(); 

    std::shared_ptr<holoscan::Operator> source_op_;
    std::shared_ptr<holoscan::Operator> pre_processor_op_;
    std::shared_ptr<holoscan::Operator> inference_op_;  
    std::shared_ptr<holoscan::Operator> post_processor_op_;
    std::shared_ptr<holoscan::Operator> visualizer_op_;
    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool_;


    YAML::Node config_;
};

#endif //  "_INFERENCE_PIPELINE_HPP_"