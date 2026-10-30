#pragma once

#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "preproc/preproc_seg.hpp"

// N is the batch size, which is always 1 in this application
// C is the number of classes for each head, which varies by head
//   input : "image"              [N, 512, 512, 3] float32, RGB, range [0, 1]
//                                (mean/std normalization and NHWC->NCHW are done inside the graph)
//   output: "head_0".."head_3"   [N, C, 512, 512] float32 logits, C = 5 / 3 / 4 / 7
//           "correlation_scores" [N, 7] float32
struct SegResult {
    static constexpr int kNumHeads = 4;
    static constexpr int kNumScores = 7;
    std::array<cv::Mat, kNumHeads> masks;       // CV_8UC1 per-pixel head-local class id (argmax), resized to the input frame size
    std::array<cv::Mat, kNumHeads> confidences; // CV_32FC1 softmax probability of that class (0 for background), same size
    std::array<float, kNumScores> correlation_scores; // correlation scores for the 7 global classes, in the order of the config file
};

class SegInferenceEngine {
public:
    // The input size is fixed at 512x512, and the number of channels is 3 (BGR).
    static constexpr int kInputWidth = 512;
    static constexpr int kInputHeight = 512;
    static constexpr int kInputChannels = 3;

    // Unet GCN model has 4 heads, each with a different number of classes:
    static constexpr int kNumHeads = SegResult::kNumHeads;
    // The number of classes for each head, in the order of the model outputs
    static constexpr int kNumScores = SegResult::kNumScores;
    // Head's number of channel (classes) in the order of the model outputs
    static constexpr std::array<int, kNumHeads> kHeadClasses = {5, 3, 4, 7};

    explicit SegInferenceEngine(const std::string& model_path, bool use_cuda = true, int device_id = 0,
                                const SegPreprocConfig& preproc_config = SegPreprocConfig());
    // input_image: BGR CV_8UC3 frame of any size
    // If the first GPU run fails (e.g. cuDNN without kernels for this GPU), the session is rebuilt on CPU
    bool Run(const cv::Mat& input_image, SegResult& result);
    bool UsingCuda() const { return using_cuda_; }
private:
    void CreateSession(bool use_cuda);
    void CheckModelIO();

    std::string model_path_;
    int device_id_;
    bool using_cuda_ = false;
    bool first_run_done_ = false;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    SegPreprocessor preprocessor_;
    SegPreprocInfo preproc_info_;
    cv::Mat model_mask_;
    cv::Mat model_conf_;
    std::vector<float> input_buffer_;
    std::array<int64_t, 4> input_shape_;
    std::unique_ptr<Ort::Value> input_tensor_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;
};
