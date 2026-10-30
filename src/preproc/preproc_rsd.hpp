#pragma once

#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Preprocessing for phase recognition / remaining surgical duration, matching
// RSD/notebooks/Eval_MultiModal_RSD.ipynb:
//   1. geometry: BGR frame -> RGB, short side 256 (INTER_AREA, only when larger)
//                -> Resize(255) (torchvision: short side 255, long side truncated) -> CenterCrop(224)
//   2. preprocess_step.onnx (Surgical3StepPreprocess, one causal step)
//        input : "frame" uint8 [1, 3, 224, 224] RGB, "hist_stats" f32 [10, 1, 3, 4], "hist_count" i64 [1]
//        output: "image" f32 [1, 3, 224, 224] ImageNet-normalized, updated hist_stats / hist_count
// The 10-step colour history is carried between calls; Reset() clears it for a new video.
class RsdPreprocessor {
public:
    static constexpr int kInputSize = 224;
    static constexpr int kDecodeShortSide = 256;
    static constexpr int kResizeShortSide = 255;    // int(224 * 1.14)
    static constexpr int kHistLen = 10;
    static constexpr size_t kImageSize = 3 * kInputSize * kInputSize;

    // Runs on CPU: the graph is tiny and keeping it off the GPU avoids a device round trip
    RsdPreprocessor(Ort::Env& env, const std::string& model_path);
    // Step 1 only: BGR frame of any size -> RGB CV_8UC3 224x224
    static void CropFrame(const cv::Mat& bgr, cv::Mat& rgb);
    // Steps 1 + 2: writes the normalized NCHW image (kImageSize floats) into dst and advances the history
    bool Process(const cv::Mat& bgr, float* dst);
    void Reset();
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<uint8_t> frame_buffer_;             // [1, 3, 224, 224] RGB planar
    std::vector<float> hist_stats_;                 // [10, 1, 3, 4]
    std::array<int64_t, 1> hist_count_{};
    cv::Mat rgb_;
};
