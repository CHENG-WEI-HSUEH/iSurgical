#ifndef SURGICAL_PHASE_HPP
#define SURGICAL_PHASE_HPP

#include <holoscan/holoscan.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <string>
#include <memory>

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

class SurgicalPhaseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SurgicalPhaseOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> model_path_;
  holoscan::Parameter<int64_t> interval_ms_;
  holoscan::Parameter<int> seq_len_;

  std::thread worker_thread_;
  std::atomic<bool> is_running_{false};
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;

  // TensorRT engine
  TrtLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
  void* d_input_{nullptr};
  void* d_output_{nullptr};
  cudaStream_t infer_stream_{nullptr};

  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  bool has_frame_{false};

  static constexpr int kInputSize = 224;
  static constexpr int kNumClasses = 7;
  static const std::vector<std::string> kPhaseNames;

  bool waitFor(std::chrono::milliseconds duration);
  bool buildEngine();
  void workerLoop();
  void runInference();
};

#endif // SURGICAL_PHASE_HPP
