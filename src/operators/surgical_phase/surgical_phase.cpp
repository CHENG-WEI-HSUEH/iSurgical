#include "surgical_phase.hpp"

#include <NvOnnxParser.h>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <future>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    HOLOSCAN_LOG_WARN("TensorRT: {}", msg);
  }
}

const std::vector<std::string> SurgicalPhaseOp::kPhaseNames = {
    "Preparation",
    "CalotTriangleDissection",
    "ClippingAndCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningAndCoagulation",
    "GallbladderRetraction"
};

bool SurgicalPhaseOp::waitFor(std::chrono::milliseconds duration) {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  return !stop_cv_.wait_for(lock, duration, [this]() { return !is_running_.load(); });
}

void SurgicalPhaseOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("frame_input")
      .condition(holoscan::ConditionType::kNone);

  spec.param(model_path_, "model_path", "Model Path",
             "Path to surgical_phase_model.onnx");
  spec.param(interval_ms_, "interval_ms", "Interval (ms)",
             "Inference trigger interval in milliseconds", int64_t(60000));
  spec.param(seq_len_, "seq_len", "Sequence Length",
             "Number of frames to sample (1 per second)", 5);
}

bool SurgicalPhaseOp::buildEngine() {
  std::string onnx_path = model_path_.get();
  int seq = seq_len_.get();

  std::string engine_path = onnx_path + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file.good()) {
    HOLOSCAN_LOG_INFO("SurgicalPhaseOp: loading cached TensorRT engine from {}", engine_path);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  } else {
    HOLOSCAN_LOG_INFO("SurgicalPhaseOp: building TensorRT engine from {} (this may take a few minutes)...", onnx_path);

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger_));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, trt_logger_));

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      HOLOSCAN_LOG_ERROR("SurgicalPhaseOp: failed to parse ONNX model");
      return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 28);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto input_tensor = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims input_dims{};
    input_dims.nbDims = 5;
    input_dims.d[0] = 1;
    input_dims.d[1] = seq;
    input_dims.d[2] = 3;
    input_dims.d[3] = kInputSize;
    input_dims.d[4] = kInputSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    engine_.reset(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) {
      HOLOSCAN_LOG_ERROR("SurgicalPhaseOp: failed to build TensorRT engine");
      return false;
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    std::ofstream out(engine_path, std::ios::binary);
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    out.close();
    HOLOSCAN_LOG_INFO("SurgicalPhaseOp: cached engine to {}", engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  }

  if (!engine_) return false;

  trt_context_.reset(engine_->createExecutionContext());
  if (!trt_context_) return false;

  nvinfer1::Dims input_dims{};
  input_dims.nbDims = 5;
  input_dims.d[0] = 1;
  input_dims.d[1] = seq;
  input_dims.d[2] = 3;
  input_dims.d[3] = kInputSize;
  input_dims.d[4] = kInputSize;
  trt_context_->setInputShape(engine_->getIOTensorName(0), input_dims);

  size_t input_size = 1 * seq * 3 * kInputSize * kInputSize * sizeof(float);
  size_t output_size = 1 * seq * kNumClasses * sizeof(float);
  cudaMalloc(&d_input_, input_size);
  cudaMalloc(&d_output_, output_size);
  cudaStreamCreate(&infer_stream_);

  return true;
}

void SurgicalPhaseOp::start() {
  is_running_ = true;
  worker_thread_ = std::thread(&SurgicalPhaseOp::workerLoop, this);

  HOLOSCAN_LOG_INFO("SurgicalPhaseOp started (interval={}ms, seq_len={})",
                    interval_ms_.get(), seq_len_.get());
}

void SurgicalPhaseOp::stop() {
  {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    is_running_ = false;
  }
  stop_cv_.notify_all();

  if (worker_thread_.joinable()) {
    std::future<void> fut = std::async(std::launch::async, [this]() {
      worker_thread_.join();
    });
    if (fut.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
      HOLOSCAN_LOG_WARN("SurgicalPhaseOp: worker thread did not exit in time, detaching");
      worker_thread_.detach();
    }
  }

  if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
  if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
  if (infer_stream_) { cudaStreamDestroy(infer_stream_); infer_stream_ = nullptr; }

  HOLOSCAN_LOG_INFO("SurgicalPhaseOp stopped");
}

void SurgicalPhaseOp::compute(holoscan::InputContext& input,
                              holoscan::OutputContext&,
                              holoscan::ExecutionContext&) {
  auto msg = input.receive<holoscan::TensorMap>("frame_input");
  if (!msg) return;

  auto& tmap = msg.value();
  if (tmap.find("video") == tmap.end()) return;

  auto tensor = tmap["video"];
  auto shape = tensor->shape();
  int h = shape[0];
  int w = shape[1];
  int c = shape[2];

  cv::Mat frame(h, w, CV_8UC3);
  size_t data_size = static_cast<size_t>(h) * w * c;
  cudaMemcpy(frame.data, tensor->data(), data_size, cudaMemcpyDeviceToHost);

  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_ = std::move(frame);
  has_frame_ = true;
}

void SurgicalPhaseOp::workerLoop() {
  HOLOSCAN_LOG_INFO("SurgicalPhaseOp: initializing TensorRT engine in background...");
  if (!buildEngine()) {
    HOLOSCAN_LOG_ERROR("SurgicalPhaseOp: engine init failed, worker exiting");
    return;
  }
  HOLOSCAN_LOG_INFO("SurgicalPhaseOp: TensorRT engine ready");

  // Wait for first frame
  while (is_running_) {
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (has_frame_) break;
    }
    if (!waitFor(std::chrono::milliseconds(100))) return;
  }

  auto interval = std::chrono::milliseconds(interval_ms_.get());

  while (is_running_) {
    if (!waitFor(interval)) return;
    if (!is_running_) return;

    runInference();
  }
}

void SurgicalPhaseOp::runInference() {
  int seq = seq_len_.get();
  std::vector<cv::Mat> sampled_frames;
  sampled_frames.reserve(seq);

  for (int i = 0; i < seq && is_running_; ++i) {
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (has_frame_) {
        cv::Mat resized;
        cv::resize(latest_frame_, resized, cv::Size(kInputSize, kInputSize));
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        sampled_frames.push_back(resized);
      }
    }
    if (i < seq - 1) {
      if (!waitFor(std::chrono::milliseconds(1000))) return;
    }
  }

  if (static_cast<int>(sampled_frames.size()) < seq) {
    HOLOSCAN_LOG_WARN("SurgicalPhaseOp: only got {}/{} frames, skipping inference",
                      sampled_frames.size(), seq);
    return;
  }

  size_t frame_chw_size = 3 * kInputSize * kInputSize;
  std::vector<float> input_buffer(seq * frame_chw_size);

  for (int t = 0; t < seq; ++t) {
    std::vector<cv::Mat> channels;
    cv::split(sampled_frames[t], channels);
    for (int ch = 0; ch < 3; ++ch) {
      float* dst = input_buffer.data() + t * frame_chw_size + ch * kInputSize * kInputSize;
      memcpy(dst, channels[ch].data, kInputSize * kInputSize * sizeof(float));
    }
  }

  size_t input_bytes = input_buffer.size() * sizeof(float);
  cudaMemcpyAsync(d_input_, input_buffer.data(), input_bytes, cudaMemcpyHostToDevice, infer_stream_);

  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);
  trt_context_->setTensorAddress(input_name, d_input_);
  trt_context_->setTensorAddress(output_name, d_output_);
  trt_context_->enqueueV3(infer_stream_);

  size_t output_bytes = seq * kNumClasses * sizeof(float);
  std::vector<float> output_buffer(seq * kNumClasses);
  cudaMemcpyAsync(output_buffer.data(), d_output_, output_bytes, cudaMemcpyDeviceToHost, infer_stream_);

  while (cudaStreamQuery(infer_stream_) == cudaErrorNotReady) {
    if (!is_running_) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  float* logits = output_buffer.data() + (seq - 1) * kNumClasses;
  int max_idx = std::distance(logits, std::max_element(logits, logits + kNumClasses));

  HOLOSCAN_LOG_INFO("[SurgicalPhase] Current phase: {}", kPhaseNames[max_idx]);
}
