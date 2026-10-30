#include "infer_rsd.hpp"

#include <algorithm>
#include <iostream>

namespace {

// The RSD model has 4 inputs 1 batch of 3 channels 224x224 images
constexpr std::array<int64_t, 4> kImageShape = {1, 3, RsdPreprocessor::kInputSize, RsdPreprocessor::kInputSize};
// 4x1x1024x3 conv state 
constexpr std::array<int64_t, 4> kConvShape = {4, 1, 1024, 3};
// 4x1x16x64x16 ssm state - 
// ssm is a state space model, which is a type of recurrent neural network that can model long-term dependencies in sequential data. 
// The ssm state is used to capture the temporal dynamics of the surgical procedure and is updated at each time step based on the current input and the previous state.
constexpr std::array<int64_t, 5> kSsmShape = {4, 1, 16, 64, 16};

template <size_t N>
size_t NumElements(const std::array<int64_t, N>& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
}

}  // namespace

RsdInferenceEngine::RsdInferenceEngine(const std::string& preprocess_model_path, const std::string& model_path,
                                       bool use_cuda, int device_id)
    : model_path_(model_path),
      device_id_(device_id),
      env_(ORT_LOGGING_LEVEL_WARNING, "RsdInference"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      preprocessor_(env_, preprocess_model_path),
      image_(RsdPreprocessor::kImageSize),
      conv_state_(NumElements(kConvShape)),
      ssm_state_(NumElements(kSsmShape)) {
    CreateSession(use_cuda);
    Reset();
}

void RsdInferenceEngine::CreateSession(bool use_cuda) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    using_cuda_ = false;
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = device_id_;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            using_cuda_ = true;
        } catch (const Ort::Exception& e) {
            std::cerr << "CUDA provider unavailable for the RSD model, falling back to CPU: " << e.what() << std::endl;
        }
    }
    session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);
}

void RsdInferenceEngine::Reset() {
    preprocessor_.Reset();
    std::fill(conv_state_.begin(), conv_state_.end(), 0.0f);
    std::fill(ssm_state_.begin(), ssm_state_.end(), 0.0f);
    step_index_ = 0;
}

bool RsdInferenceEngine::Run(const cv::Mat& frame, double video_time_s, RsdResult& result) {
    if (!preprocessor_.Process(frame, image_.data())) {
        return false;
    }

    // The temporal state is only committed after a successful run, so a CPU retry repeats the same step
    try {
        RunModel(result);
    } catch (const Ort::Exception& e) {
        if (!using_cuda_ || first_run_done_) {
            std::cerr << "RSD inference failed: " << e.what() << std::endl;
            return false;
        }
        std::cerr << "GPU RSD inference failed on the first run, falling back to CPU: " << e.what() << std::endl;
        try {
            CreateSession(false);
            RunModel(result);
        } catch (const Ort::Exception& e2) {
            std::cerr << "RSD inference failed: " << e2.what() << std::endl;
            return false;
        }
    }
    first_run_done_ = true;
    result.video_time_s = video_time_s;
    result.step_index = ++step_index_;
    return true;
}

void RsdInferenceEngine::RunModel(RsdResult& result) {
    std::array<Ort::Value, 3> inputs = {
        Ort::Value::CreateTensor<float>(memory_info_, image_.data(), image_.size(),
                                        kImageShape.data(), kImageShape.size()),
        Ort::Value::CreateTensor<float>(memory_info_, conv_state_.data(), conv_state_.size(),
                                        kConvShape.data(), kConvShape.size()),
        Ort::Value::CreateTensor<float>(memory_info_, ssm_state_.data(), ssm_state_.size(),
                                        kSsmShape.data(), kSsmShape.size())};
    const std::array<const char*, 3> input_names = {"image", "conv_state", "ssm_state"};
    const std::array<const char*, 6> output_names = {"phase_probs", "pred_rsd", "pred_progress", "z_t",
                                                     "conv_state_out", "ssm_state_out"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                 output_names.data(), output_names.size());

    const float* probs = outputs[0].GetTensorData<float>();
    std::copy(probs, probs + RsdResult::kNumPhases, result.phase_probs.begin());
    const auto best = std::max_element(result.phase_probs.begin(), result.phase_probs.end());
    result.phase = static_cast<int>(best - result.phase_probs.begin());
    result.phase_confidence = *best;
    result.rsd_s = outputs[1].GetTensorData<float>()[0];
    result.progress = outputs[2].GetTensorData<float>()[0];
    const float* z_t = outputs[3].GetTensorData<float>();
    std::copy(z_t, z_t + RsdResult::kEmbedDim, result.z_t.begin());

    const float* conv = outputs[4].GetTensorData<float>();
    std::copy(conv, conv + conv_state_.size(), conv_state_.begin());
    const float* ssm = outputs[5].GetTensorData<float>();
    std::copy(ssm, ssm + ssm_state_.size(), ssm_state_.begin());
}
