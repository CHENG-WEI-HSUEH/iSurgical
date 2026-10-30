#include "preproc_rsd.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

constexpr std::array<int64_t, 4> kFrameShape = {1, 3, RsdPreprocessor::kInputSize, RsdPreprocessor::kInputSize};
constexpr std::array<int64_t, 4> kHistShape = {RsdPreprocessor::kHistLen, 1, 3, 4};
constexpr std::array<int64_t, 1> kCountShape = {1};

}  // namespace

RsdPreprocessor::RsdPreprocessor(Ort::Env& env, const std::string& model_path)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      frame_buffer_(kImageSize),
      hist_stats_(static_cast<size_t>(kHistLen) * 3 * 4) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    Reset();
}

void RsdPreprocessor::Reset() {
    std::fill(hist_stats_.begin(), hist_stats_.end(), 0.0f);
    hist_count_[0] = 0;
}

void RsdPreprocessor::CropFrame(const cv::Mat& bgr, cv::Mat& rgb) {
    // Decode-time downscale: short side 256, sizes rounded (Eval notebook _decode)
    cv::Mat decoded = bgr;
    const int short_side = std::min(bgr.cols, bgr.rows);
    if (short_side > kDecodeShortSide) {
        const double s = static_cast<double>(kDecodeShortSide) / short_side;
        cv::resize(bgr, decoded, cv::Size(static_cast<int>(std::lround(bgr.cols * s)),
                                          static_cast<int>(std::lround(bgr.rows * s))),
                   0, 0, cv::INTER_AREA);
    }

    // torchvision Resize(255): short side -> 255, long side = int(255 * long / short)
    const int w = decoded.cols;
    const int h = decoded.rows;
    cv::Size resized_size;
    if (w <= h) {
        resized_size = cv::Size(kResizeShortSide, static_cast<int>(static_cast<double>(kResizeShortSide) * h / w));
    } else {
        resized_size = cv::Size(static_cast<int>(static_cast<double>(kResizeShortSide) * w / h), kResizeShortSide);
    }
    cv::Mat resized;
    cv::resize(decoded, resized, resized_size, 0, 0, cv::INTER_LINEAR);

    // torchvision CenterCrop(224): offsets use Python round(), i.e. halves to even (114.5 -> 114),
    // which std::nearbyint matches under the default rounding mode; std::lround would give 115
    const int left = static_cast<int>(std::nearbyint((resized.cols - kInputSize) / 2.0));
    const int top = static_cast<int>(std::nearbyint((resized.rows - kInputSize) / 2.0));
    cv::cvtColor(resized(cv::Rect(left, top, kInputSize, kInputSize)), rgb, cv::COLOR_BGR2RGB);
}

bool RsdPreprocessor::Process(const cv::Mat& bgr, float* dst) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        std::cerr << "RSD input frame must be a non-empty CV_8UC3 BGR image." << std::endl;
        return false;
    }

    // HWC RGB -> planar CHW uint8
    CropFrame(bgr, rgb_);
    const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    std::vector<cv::Mat> channels = {
        cv::Mat(kInputSize, kInputSize, CV_8UC1, frame_buffer_.data()),
        cv::Mat(kInputSize, kInputSize, CV_8UC1, frame_buffer_.data() + plane),
        cv::Mat(kInputSize, kInputSize, CV_8UC1, frame_buffer_.data() + 2 * plane)};
    cv::split(rgb_, channels);

    std::array<Ort::Value, 3> inputs = {
        Ort::Value::CreateTensor<uint8_t>(memory_info_, frame_buffer_.data(), frame_buffer_.size(),
                                          kFrameShape.data(), kFrameShape.size()),
        Ort::Value::CreateTensor<float>(memory_info_, hist_stats_.data(), hist_stats_.size(),
                                        kHistShape.data(), kHistShape.size()),
        Ort::Value::CreateTensor<int64_t>(memory_info_, hist_count_.data(), hist_count_.size(),
                                          kCountShape.data(), kCountShape.size())};
    const std::array<const char*, 3> input_names = {"frame", "hist_stats", "hist_count"};
    const std::array<const char*, 3> output_names = {"image", "hist_stats_out", "hist_count_out"};
    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "RSD preprocessing failed: " << e.what() << std::endl;
        return false;
    }

    const float* image = outputs[0].GetTensorData<float>();
    std::copy(image, image + kImageSize, dst);
    const float* stats = outputs[1].GetTensorData<float>();
    std::copy(stats, stats + hist_stats_.size(), hist_stats_.begin());
    hist_count_[0] = outputs[2].GetTensorData<int64_t>()[0];
    return true;
}
