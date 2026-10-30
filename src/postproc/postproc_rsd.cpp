#include "postproc_rsd.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

RsdConfig RsdConfig::FromYaml(const std::string& path, const std::string& procedure) {
    RsdConfig config;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (!procedure.empty()) {
            if (!root[procedure]) throw std::runtime_error(path + ": missing procedure '" + procedure + "'");
            root = root[procedure];
        }
        if (root["RSD_Period"]) config.display_period_s = root["RSD_Period"].as<double>();
        if (root["RSD_InferFPS"]) config.infer_fps = root["RSD_InferFPS"].as<double>();
        if (root["Phase"]) config.phase_names = root["Phase"].as<std::vector<std::string>>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    if (config.display_period_s <= 0.0) {
        throw std::runtime_error(path + ": RSD_Period must be > 0");
    }
    if (config.infer_fps <= 0.0) {
        throw std::runtime_error(path + ": RSD_InferFPS must be > 0");
    }
    if (static_cast<int>(config.phase_names.size()) != RsdResult::kNumPhases) {
        throw std::runtime_error(path + ": Phase must have " + std::to_string(RsdResult::kNumPhases) + " entries");
    }
    return config;
}

std::string FormatDuration(double seconds) {
    const int total = static_cast<int>(std::lround(std::max(0.0, seconds)));
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;
    return h > 0 ? cv::format("%d:%02d:%02d", h, m, s) : cv::format("%02d:%02d", m, s);
}

RsdPostProcessor::RsdPostProcessor(const RsdConfig& config)
    : config_(config) {}

void RsdPostProcessor::Reset() {
    displayed_ = RsdResult();
    next_display_time_s_ = 0.0;
}

bool RsdPostProcessor::Update(const RsdResult& result) {
    if (result.video_time_s < next_display_time_s_) {
        return false;
    }
    displayed_ = result;
    while (next_display_time_s_ <= result.video_time_s) {
        next_display_time_s_ += config_.display_period_s;
    }
    return true;
}

const std::string& RsdPostProcessor::PhaseName(int phase) const {
    static const std::string kUnknown = "-";
    return (phase >= 0 && phase < static_cast<int>(config_.phase_names.size())) ? config_.phase_names[phase] : kUnknown;
}

std::string RsdPostProcessor::Describe(const RsdResult& result) const {
    std::ostringstream out;
    out << "[rsd step " << result.step_index << " @ " << FormatDuration(result.video_time_s) << "] phase: "
        << PhaseName(result.phase) << " (" << cv::format("%.2f", result.phase_confidence) << ")  RSD: "
        << FormatDuration(result.rsd_s) << "  progress: " << cv::format("%.1f", result.progress * 100.0f) << "%";
    return out.str();
}

void RsdPostProcessor::Draw(cv::Mat& output, const RsdResult& result) const {
    constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double kFontScale = 0.45;
    constexpr int kPad = 6;
    constexpr int kMargin = 8;
    constexpr int kBarHeight = 6;

    std::vector<std::string> lines;
    if (result.phase < 0) {
        lines = {"Phase: waiting for first step", "RSD: --:--", "Progress: --"};
    } else {
        lines = {cv::format("Phase: %s (%.2f)", PhaseName(result.phase).c_str(), result.phase_confidence),
                 "RSD: " + FormatDuration(result.rsd_s),
                 cv::format("Progress: %.0f%%", std::clamp(result.progress, 0.0f, 1.0f) * 100.0f),
                 "Updated at " + FormatDuration(result.video_time_s)};
    }

    int text_width = 0;
    int text_height = 0;
    for (const auto& line : lines) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(line, kFont, kFontScale, 1, &baseline);
        text_width = std::max(text_width, size.width);
        text_height = std::max(text_height, size.height);
    }
    const int row_height = text_height + kPad;
    const int bar_width = std::max(text_width, 160);
    const cv::Size box_size(kPad + bar_width + kPad,
                            kPad + row_height * static_cast<int>(lines.size()) + kBarHeight + kPad);
    cv::Rect box(kMargin, output.rows - box_size.height - kMargin, box_size.width, box_size.height);
    box &= cv::Rect(0, 0, output.cols, output.rows);
    if (box.empty()) return;

    // Darkened translucent background, same style as the class legend
    cv::Mat roi = output(box);
    roi.convertTo(roi, -1, 0.4, 0.0);

    int y = box.y + kPad;
    for (const auto& line : lines) {
        cv::putText(output, line, cv::Point(box.x + kPad, y + text_height), kFont, kFontScale,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += row_height;
    }

    // Progress bar
    const cv::Rect bar(box.x + kPad, y, bar_width, kBarHeight);
    cv::rectangle(output, bar, cv::Scalar(255, 255, 255), 1);
    if (result.phase >= 0) {
        const int filled = static_cast<int>(std::lround(bar.width * std::clamp(result.progress, 0.0f, 1.0f)));
        if (filled > 0) {
            cv::rectangle(output, cv::Rect(bar.x, bar.y, filled, bar.height), cv::Scalar(80, 200, 255), cv::FILLED);
        }
    }
}
