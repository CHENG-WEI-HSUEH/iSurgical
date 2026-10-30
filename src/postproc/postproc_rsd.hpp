#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "infer/infer_rsd.hpp"

// Settings from configs/lc_config.yaml (all optional)
//   RSD_Period   : seconds between on-screen updates of phase / RSD, default 15
//   RSD_InferFPS : inference rate in frames per second of source time, default 1 (the training rate)
//   Phase        : the 7 phase names in model output order, default Cholec80
struct RsdConfig {
    double display_period_s = 15.0;
    double infer_fps = 1.0;
    std::vector<std::string> phase_names = {
        "Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
        "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"};

    // Reads the `procedure` section of the file, or the whole file if `procedure` is empty.
    // Throws std::runtime_error on bad values
    static RsdConfig FromYaml(const std::string& path, const std::string& procedure = "");
};

// Inference runs every 1 / infer_fps seconds; the displayed result only changes every display_period_s
// and is drawn unchanged on every frame in between.
class RsdPostProcessor {
public:
    explicit RsdPostProcessor(const RsdConfig& config);
    // Feed every inference result; returns true when it became the displayed result
    bool Update(const RsdResult& result);
    const RsdResult& Displayed() const { return displayed_; }
    void Reset();
    // Bottom-left panel with phase, RSD and progress of `result`
    void Draw(cv::Mat& output, const RsdResult& result) const;
    std::string Describe(const RsdResult& result) const;
    const std::string& PhaseName(int phase) const;
    const RsdConfig& Config() const { return config_; }
private:
    RsdConfig config_;
    RsdResult displayed_;
    double next_display_time_s_ = 0.0;
};

// "mm:ss", or "h:mm:ss" from one hour
std::string FormatDuration(double seconds);
