#include "infer_seg.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

SegInferenceEngine::SegInferenceEngine(const std::string& model_path, bool use_cuda, int device_id,
                                       const SegPreprocConfig& preproc_config)
    : model_path_(model_path),
      device_id_(device_id),
      env_(ORT_LOGGING_LEVEL_WARNING, "SegInference"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      preprocessor_(kInputWidth, kInputHeight, preproc_config),
      input_buffer_(static_cast<size_t>(kInputHeight) * kInputWidth * kInputChannels),
      input_shape_{1, kInputHeight, kInputWidth, kInputChannels} {

    // Create the session for inference, and get the input/output names.
    CreateSession(use_cuda);

    Ort::AllocatorWithDefaultOptions allocator;
    // Get the input and output names from the model, and store them in vectors of strings and C-style string pointers.
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
    }

    // Convert the input and output names to C-style string pointers for use with the ONNX Runtime API.
    for (const auto& name : input_names_) input_names_ptr_.push_back(name.c_str());
    for (const auto& name : output_names_) output_names_ptr_.push_back(name.c_str());

    // Check that the model has the expected input and output names and shapes.
    CheckModelIO();

    // Input tensor wraps input_buffer_, so it is created once and refilled on every Run
    input_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        memory_info_, input_buffer_.data(), input_buffer_.size(), input_shape_.data(), input_shape_.size()));
}

// Set up the ONNX Runtime session with the specified device (CPU or CUDA)..
void SegInferenceEngine::CreateSession(bool use_cuda) {
    // Options must be set before the session is created
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
            std::cerr << "CUDA provider unavailable, falling back to CPU: " << e.what() << std::endl;
        }
    }
    session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);
}

// Check that the model has the expected input and output names and shapes.
void SegInferenceEngine::CheckModelIO() {
    if (input_names_.size() != 1 || input_names_[0] != "image") {
        throw std::runtime_error("Unexpected model input, expected a single input named 'image'");
    }
    // + 1 for the correlation_scores output
    if (output_names_.size() != kNumHeads + 1) {
        throw std::runtime_error("Unexpected model output count, expected " + std::to_string(kNumHeads + 1));
    }
    for (int i = 0; i < kNumHeads; ++i) {
        if (output_names_[i] != "head_" + std::to_string(i)) {
            throw std::runtime_error("Unexpected model output name: " + output_names_[i]);
        }
    }
    if (output_names_[kNumHeads] != "correlation_scores") {
        throw std::runtime_error("Unexpected model output name: " + output_names_[kNumHeads]);
    }
}

// Runs the segmentation model on one frame and fills `result`. Steps:
//   1. Preprocess the frame into the 512x512 model input
//   2. Run the ONNX model (falls back to CPU if the very first GPU run fails)
//   3. For each of the 4 heads, turn logits into a class-id mask and a confidence map,
//      then map both back to the original frame size
//   4. Copy the 7 correlation scores
// Returns false if preprocessing or inference fails.
bool SegInferenceEngine::Run(const cv::Mat& input_image, SegResult& result) {

    // Step 1: resize / crop / convert BGR->RGB [0, 1] directly into input_buffer_, which input_tensor_ wraps.
    // preproc_info_ records how the frame was mapped, so the outputs can be mapped back later.
    if (!preprocessor_.Process(input_image, input_buffer_.data(), preproc_info_)) {
        // Invalid frame (e.g. empty); nothing to infer.
        return false;
    }

    // Step 2: wrap the inference call in a lambda so it can be retried after rebuilding the session.
    auto run_session = [this] {
        // Feed the single input "image" and request all outputs (head_0..head_3, correlation_scores).
        return session_->Run(Ort::RunOptions{nullptr},
                             input_names_ptr_.data(), input_tensor_.get(), input_names_ptr_.size(),
                             output_names_ptr_.data(), output_names_ptr_.size());
    };

    // Model outputs, in the same order as output_names_.
    std::vector<Ort::Value> outputs;
    try {
        // Normal path: run on the current device (GPU or CPU).
        outputs = run_session();
    } catch (const Ort::Exception& e) {
        // Already on CPU, or GPU has worked before: this is a real error, give up on this frame.
        if (!using_cuda_ || first_run_done_) {
            std::cerr << "Segmentation inference failed: " << e.what() << std::endl;
            return false;
        }
        // First GPU run failed (e.g. cuDNN has no kernels for this GPU): switch to CPU permanently.
        std::cerr << "GPU inference failed on the first run, falling back to CPU: " << e.what() << std::endl;
        try {
            // Rebuild the session on CPU and retry the same input.
            CreateSession(false);
            outputs = run_session();
        } catch (const Ort::Exception& e2) {
            // CPU failed too; give up on this frame.
            std::cerr << "Segmentation inference failed: " << e2.what() << std::endl;
            return false;
        }
    }
    // From now on a failure is never treated as a GPU compatibility problem.
    first_run_done_ = true;

    // Step 3: segmentation heads: [1, C, H, W] logits -> per-pixel argmax class id
    for (int head = 0; head < kNumHeads; ++head) {
        // Read this head's output shape; C differs per head (5 / 3 / 4 / 7).
        const auto shape = outputs[head].GetTensorTypeAndShapeInfo().GetShape();
        // C: number of channels = number of classes in this head, channel 0 is Background.
        const int num_classes = static_cast<int>(shape[1]);
        // Output height and width.
        const int out_h = static_cast<int>(shape[2]);
        const int out_w = static_cast<int>(shape[3]);
        // Number of pixels in one channel; the data is laid out channel by channel (NCHW).
        const size_t plane = static_cast<size_t>(out_h) * out_w;
        // Raw logits: the score of class c at pixel p is logits[c * plane + p].
        const float* logits = outputs[head].GetTensorData<float>();

        // Allocate (or reuse) the model-resolution class-id mask and confidence map.
        model_mask_.create(out_h, out_w, CV_8UC1);
        model_conf_.create(out_h, out_w, CV_32FC1);
        // Raw pointers for fast per-pixel writes.
        uint8_t* mask_data = model_mask_.ptr<uint8_t>();
        float* conf_data = model_conf_.ptr<float>();
        // Visit every pixel.
        for (size_t p = 0; p < plane; ++p) {
            // Start with class 0 (Background) as the best so far.
            int best_class = 0;
            float best_logit = logits[p];
            // Compare against the remaining classes to find the argmax.
            for (int c = 1; c < num_classes; ++c) {
                const float v = logits[c * plane + p];
                if (v > best_logit) {
                    best_logit = v;
                    best_class = c;
                }
            }
            // Store the winning head-local class id (0 = Background, 1..C-1 = this head's classes).
            mask_data[p] = static_cast<uint8_t>(best_class);
            // Softmax probability of the winning class: 1 / sum_c exp(l_c - l_max).
            // Only foreground pixels use it, so background skips the exp work.
            if (best_class == 0) {
                conf_data[p] = 0.0f;
                continue;
            }
            // Sum exp(l_c - l_max) over all classes; subtracting l_max avoids overflow.
            float denom = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                denom += std::exp(logits[c * plane + p] - best_logit);
            }
            // The winner's term is exp(0) = 1, so its probability is 1 / denom.
            conf_data[p] = 1.0f / denom;
        }
        // Undo the preprocessing (crop / letterbox / resize) so the mask lines up with the original frame.
        SegPreprocessor::RestoreMask(model_mask_, preproc_info_, result.masks[head]);
        // Same for the confidence map.
        SegPreprocessor::RestoreMask(model_conf_, preproc_info_, result.confidences[head]);
    }

    // Step 4: correlation scores: [1, 7], the output right after the heads.
    const float* scores = outputs[kNumHeads].GetTensorData<float>();
    // Copy the 7 values into the result as-is.
    std::copy(scores, scores + kNumScores, result.correlation_scores.begin());

    // Success.
    return true;
}
