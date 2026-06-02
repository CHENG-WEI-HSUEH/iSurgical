
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
    void ConfigureInferencePreProcessorOperator();
    void ConfigureRawPreProcessorOperator();
    void ConfigureInferenceOperator();
    void ConfigurePostProcessorOperator();
    void ConfigureVisualizerOperator();
    void ConfigurePeriodicCounterOperator();
    void ConfigureRecorderOperator();
    void ConfigureSurgicalPhaseOperator();

    // Operators
    std::shared_ptr<holoscan::Operator> source_op_;
    std::shared_ptr<holoscan::Operator> pre_infer_processor_op_;
    std::shared_ptr<holoscan::Operator> pre_raw_processor_op_;
    std::shared_ptr<holoscan::Operator> inference_op_;
    std::shared_ptr<holoscan::Operator> post_processor_op_;
    std::shared_ptr<holoscan::Operator> visualizer_op_;
    std::shared_ptr<holoscan::Operator> periodic_counter_op_;
    std::shared_ptr<holoscan::Operator> recorder_op_;
    std::shared_ptr<holoscan::Operator> surgical_phase_op_;
    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool_;


    YAML::Node config_;
};

#endif //  "_INFERENCE_PIPELINE_HPP_"