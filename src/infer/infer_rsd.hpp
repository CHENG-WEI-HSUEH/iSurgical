#pragma once

#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "preproc/preproc_rsd.hpp"

// Phase recognition + remaining surgical duration with multimodal_rsd_step.onnx
// (MultiModal_RSD.stream_step, see RSD/src/MultiModal_RSD.py and Export_ONNX.py):
//   image [1, 3, 224, 224] (from RsdPreprocessor)
//     -> Swin-B -> 512-d z_euc -> Poincare ball exp_o(clip(z_euc)) -> z_t
//     -> SurgicalMamba (conv_state [4, 1, 1024, 3], ssm_state [4, 1, 16, 64, 16] carried between steps)
//     -> SPRHead: phase_probs [1, 7]      RSDHead: pred_rsd [1, 1] seconds, pred_progress [1, 1]
// The model was trained and evaluated at 1 frame per second; call Run at that rate.
struct RsdResult {
    static constexpr int kNumPhases = 7;
    static constexpr int kEmbedDim = 512;
    int phase = -1;                         // argmax of phase_probs, -1 before the first step
    float phase_confidence = 0.0f;
    std::array<float, kNumPhases> phase_probs{};
    float rsd_s = 0.0f;                     // predicted remaining surgical duration, seconds
    float progress = 0.0f;                  // predicted progress in [0, 1]
    std::array<float, kEmbedDim> z_t{};     // Poincare-ball frame embedding
    double video_time_s = 0.0;              // source time of the frame
    int64_t step_index = 0;                 // steps since the last Reset
};

class RsdInferenceEngine {
public:
    RsdInferenceEngine(const std::string& preprocess_model_path, const std::string& model_path,
                       bool use_cuda = true, int device_id = 0);
    // frame: BGR CV_8UC3 of any size. Advances the preprocessing history and the temporal state by one step.
    bool Run(const cv::Mat& frame, double video_time_s, RsdResult& result);
    // New video: clear the preprocessing history and the temporal state
    void Reset();
    bool UsingCuda() const { return using_cuda_; }
private:
    void CreateSession(bool use_cuda);
    void RunModel(RsdResult& result);

    std::string model_path_;
    int device_id_;
    bool using_cuda_ = false;
    bool first_run_done_ = false;

    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    RsdPreprocessor preprocessor_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<float> image_;              // [1, 3, 224, 224]
    std::vector<float> conv_state_;         // [4, 1, 1024, 3]
    std::vector<float> ssm_state_;          // [4, 1, 16, 64, 16]
    int64_t step_index_ = 0;
};
