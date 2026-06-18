
#ifndef  _INFERENCE_PIPELINE_HPP_
#define  _INFERENCE_PIPELINE_HPP_

// Includes the main Holoscan framework API.
#include <holoscan/holoscan.hpp>
#include <yaml-cpp/yaml.h>

#include "./operators/phase_grpc_client/phase_grpc_client.hpp"

class InferencePipeline : public holoscan::Application {
  public:
    InferencePipeline(const YAML::Node& config);
    void compose() override;

  private:
    void ConfigureSourceOperator();
    void ConfigureInferencePreProcessorOperator();
    void ConfigureRawPreProcessorOperator();
    void ConfigureRecordePreProcessorOperator();
    void ConfigureInferenceOperator();
    void ConfigurePostProcessorOperator();
    void ConfigureVisualizerOperator();
    void ConfigurePeriodicCounterOperator();
    void ConfigureRecorderOperator();
    void ConfigurePhaseFrameSamplerOperator();
    void ConfigurePhasePreProcessorOperator();
    void ConfigurePhaseInferenceOperator();
    void ConfigureSurgicalPhaseOperator();
    void ConfigurePhaseGrpcClient();

    // Operators
    std::shared_ptr<holoscan::Operator> source_op_;
    std::shared_ptr<holoscan::Operator> pre_infer_processor_op_;
    std::shared_ptr<holoscan::Operator> pre_raw_processor_op_;
    std::shared_ptr<holoscan::Operator> pre_record_processor_op_;
    std::shared_ptr<holoscan::Operator> inference_op_;
    std::shared_ptr<holoscan::Operator> post_processor_op_;
    std::shared_ptr<holoscan::Operator> visualizer_op_;
    std::shared_ptr<holoscan::Operator> periodic_counter_op_;
    std::shared_ptr<holoscan::Operator> recorder_op_;
    std::shared_ptr<holoscan::Operator> phase_frame_sampler_op_;
    std::shared_ptr<holoscan::Operator> phase_pre_processor_op_;
    std::shared_ptr<holoscan::Operator> phase_inference_op_;
    std::shared_ptr<holoscan::Operator> surgical_phase_op_;
    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool_;

    // Separate CUDA stream pool for the phase branch, kept independent of cuda_stream_pool_
    // purely for resource isolation (so a burst of phase-branch allocations can't starve
    // the segmentation path of streams).
    std::shared_ptr<holoscan::CudaStreamPool> phase_cuda_stream_pool_;

    // Plain gRPC client (not a Holoscan operator)
    std::shared_ptr<PhaseGrpcClient> phase_grpc_client_;

    YAML::Node config_;
    YAML::Node procedure_config_;
    bool is_recorded_;
};

#endif //  "_INFERENCE_PIPELINE_HPP_"