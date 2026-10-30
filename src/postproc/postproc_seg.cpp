#include "postproc_seg.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

SegPostConfig SegPostConfig::FromYaml(const std::string& path, const std::string& procedure) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load " + path + ": " + e.what());
    }
    if (!procedure.empty()) {
        if (!root[procedure]) throw std::runtime_error(path + ": missing procedure '" + procedure + "'");
        root = root[procedure];
    }

    auto require = [&](const char* key) {
        if (!root[key]) throw std::runtime_error(path + ": missing key '" + key + "'");
        return root[key];
    };

    SegPostConfig config;
    try {
        config.head_groups = require("head_groups").as<std::vector<std::vector<int>>>();
        config.label_names = require("label_names").as<std::vector<std::string>>();
        for (const auto& rgba : require("kclass_colours").as<std::vector<std::vector<double>>>()) {
            if (rgba.size() < 3) throw std::runtime_error(path + ": kclass_colours entries need at least 3 values");
            config.class_colours.emplace_back(rgba[2] * 255.0, rgba[1] * 255.0, rgba[0] * 255.0);   // RGB -> BGR
            config.class_alphas.push_back(rgba.size() >= 4 ? static_cast<float>(std::clamp(rgba[3], 0.0, 1.0)) : 1.0f);
        }
        config.num_classes = require("NumClass").as<int>();
        config.mask_threshold = require("MaskThreshold").as<float>();
        config.confidence_threshold = require("ConfidenceThreshold").as<float>();
        if (root["IoUThreshold"]) config.iou_threshold = root["IoUThreshold"].as<float>();
        if (root["MorphKernelSize"]) config.morph_kernel_size = root["MorphKernelSize"].as<int>();
        if (root["MinAreaRatio"]) config.min_area_ratio = root["MinAreaRatio"].as<float>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    config.Validate();
    return config;
}

void SegPostConfig::Validate() const {
    if (num_classes <= 0 || num_classes > 256) {
        throw std::runtime_error("NumClass must be in [1, 256]");
    }
    if (static_cast<int>(label_names.size()) != num_classes || static_cast<int>(class_colours.size()) != num_classes ||
        static_cast<int>(class_alphas.size()) != num_classes) {
        throw std::runtime_error("label_names and kclass_colours must both have NumClass entries");
    }
    if (static_cast<int>(head_groups.size()) != SegResult::kNumHeads) {
        throw std::runtime_error("head_groups must have " + std::to_string(SegResult::kNumHeads) + " groups");
    }
    for (int head = 0; head < SegResult::kNumHeads; ++head) {
        // Channel 0 of each head is Background, so a head with C channels maps C - 1 classes
        const int expected = SegInferenceEngine::kHeadClasses[head] - 1;
        if (static_cast<int>(head_groups[head].size()) != expected) {
            throw std::runtime_error("head_groups[" + std::to_string(head) + "] must have " +
                                     std::to_string(expected) + " classes to match the model");
        }
        for (int class_id : head_groups[head]) {
            if (class_id <= 0 || class_id >= num_classes) {
                throw std::runtime_error("head_groups class id " + std::to_string(class_id) + " out of range");
            }
        }
    }
}

SegPostProcessor::SegPostProcessor(const SegPostConfig& config)
    : config_(config) {
    config_.Validate();
    // Per-class colour and opacity lookup tables; Background (and unused ids) stay fully transparent
    colour_lut_ = cv::Mat(1, 256, CV_8UC3, cv::Scalar::all(0));
    alpha_lut_ = cv::Mat(1, 256, CV_8UC1, cv::Scalar(0));
    for (int class_id = 1; class_id < config_.num_classes; ++class_id) {
        const cv::Scalar& c = config_.class_colours[class_id];
        colour_lut_.at<cv::Vec3b>(0, class_id) = cv::Vec3b(cv::saturate_cast<uint8_t>(c[0]),
                                                           cv::saturate_cast<uint8_t>(c[1]),
                                                           cv::saturate_cast<uint8_t>(c[2]));
        alpha_lut_.at<uint8_t>(0, class_id) = cv::saturate_cast<uint8_t>(config_.class_alphas[class_id] * 255.0f);
    }
    if (config_.morph_kernel_size > 0) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(config_.morph_kernel_size, config_.morph_kernel_size));
    }
}

const std::string& SegPostProcessor::LabelName(int class_id) const {
    return config_.label_names[class_id];
}

cv::Scalar SegPostProcessor::ClassColour(int class_id) const {
    return config_.class_colours[class_id];
}

void SegPostProcessor::Process(const SegResult& seg_result, SegPostResult& post_result) const {
    for (int head = 0; head < SegResult::kNumHeads; ++head) {
        ProcessHead(head, seg_result.masks[head], seg_result.confidences[head], post_result.heads[head]);
    }
    FuseHeads(seg_result, post_result);
    post_result.correlation_scores = seg_result.correlation_scores;
}

void SegPostProcessor::ProcessHead(int head, const cv::Mat& mask, const cv::Mat& confidence,
                                   SegHeadOutput& head_output) const {
    head_output.regions.clear();
    if (mask.empty()) {
        head_output.mask.release();
        return;
    }

    const double frame_area = static_cast<double>(mask.total());
    const int min_area = static_cast<int>(config_.min_area_ratio * frame_area);
    const auto& group = config_.head_groups[head];

    // Output holds global class ids; everything not kept below stays Background
    head_output.mask = cv::Mat::zeros(mask.size(), CV_8UC1);

    // Pixels below MaskThreshold are treated as Background
    cv::Mat confident;
    cv::compare(confidence, config_.mask_threshold, confident, cv::CMP_GE);

    cv::Mat binary, labels, stats, centroids;
    for (int local_class = 1; local_class <= static_cast<int>(group.size()); ++local_class) {
        cv::compare(mask, local_class, binary, cv::CMP_EQ);
        binary &= confident;
        if (cv::countNonZero(binary) == 0) continue;

        if (!morph_kernel_.empty()) {
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, morph_kernel_);
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, morph_kernel_);
        }

        const int class_id = group[local_class - 1];
        const int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
        for (int label = 1; label < num_labels; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < min_area) continue;

            const cv::Rect bbox(stats.at<int>(label, cv::CC_STAT_LEFT),
                                stats.at<int>(label, cv::CC_STAT_TOP),
                                stats.at<int>(label, cv::CC_STAT_WIDTH),
                                stats.at<int>(label, cv::CC_STAT_HEIGHT));
            const cv::Mat component = (labels(bbox) == label);
            const float region_confidence = static_cast<float>(cv::mean(confidence(bbox), component)[0]);
            if (region_confidence < config_.confidence_threshold) continue;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(component, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, bbox.tl());
            if (contours.empty()) continue;
            auto largest = std::max_element(contours.begin(), contours.end(),
                [](const auto& a, const auto& b) { return a.size() < b.size(); });

            head_output.mask(bbox).setTo(class_id, component);

            SegRegion region;
            region.class_id = class_id;
            region.head = head;
            region.area = area;
            region.area_ratio = static_cast<float>(area / frame_area);
            region.confidence = region_confidence;
            region.bbox = bbox;
            region.centroid = cv::Point2f(static_cast<float>(centroids.at<double>(label, 0)),
                                          static_cast<float>(centroids.at<double>(label, 1)));
            region.contour = std::move(*largest);
            head_output.regions.push_back(std::move(region));
        }
    }

    std::sort(head_output.regions.begin(), head_output.regions.end(),
              [](const SegRegion& a, const SegRegion& b) { return a.area > b.area; });
}

void SegPostProcessor::FuseHeads(const SegResult& seg_result, SegPostResult& post_result) const {
    const cv::Mat& first = post_result.heads[0].mask;
    if (first.empty()) {
        post_result.label_map.release();
        return;
    }
    post_result.label_map = cv::Mat::zeros(first.size(), CV_8UC1);
    cv::Mat best_confidence = cv::Mat::zeros(first.size(), CV_32FC1);
    cv::Mat select;
    for (int head = 0; head < SegResult::kNumHeads; ++head) {
        const cv::Mat& head_mask = post_result.heads[head].mask;
        const cv::Mat& confidence = seg_result.confidences[head];
        if (head_mask.empty()) continue;
        cv::compare(confidence, best_confidence, select, cv::CMP_GT);
        select &= (head_mask > 0);
        head_mask.copyTo(post_result.label_map, select);
        confidence.copyTo(best_confidence, select);
    }
}

void SegPostProcessor::Draw(const cv::Mat& frame, const SegPostResult& result, cv::Mat& output, int head) const {
    frame.copyTo(output);
    const cv::Mat& label_map = (head < 0) ? result.label_map : result.heads[head].mask;
    if (label_map.empty() || label_map.size() != frame.size()) return;

    // Fill layer: every labelled pixel, blended with its class colour and alpha from kclass_colours
    BlendLabels(label_map, output);

    // Outline layer: region contours drawn as class ids, blended with the same per-class alpha on top
    // of the fill. Names go in the legend instead of on every instance.
    cv::Mat outline = cv::Mat::zeros(label_map.size(), CV_8UC1);
    for (int h = 0; h < SegResult::kNumHeads; ++h) {
        if (head >= 0 && h != head) continue;
        for (const auto& region : result.heads[h].regions) {
            cv::polylines(outline, region.contour, true, cv::Scalar(region.class_id), 2);
        }
    }
    BlendLabels(outline, output);

    // Constant legend: every class from the config (or from the selected head), whether visible or not
    std::set<int> legend_classes;
    if (head < 0) {
        for (int class_id = 1; class_id < config_.num_classes; ++class_id) legend_classes.insert(class_id);
    } else {
        legend_classes.insert(config_.head_groups[head].begin(), config_.head_groups[head].end());
    }
    DrawLegend(output, legend_classes);
}

void SegPostProcessor::BlendLabels(const cv::Mat& labels, cv::Mat& output) const {
    // output = (1 - a) * output + a * colour, with colour and a looked up per class id; Background has a = 0
    cv::Mat labels_bgr, colour, alpha8, alpha, inv_alpha;
    cv::cvtColor(labels, labels_bgr, cv::COLOR_GRAY2BGR);
    cv::LUT(labels_bgr, colour_lut_, colour);
    cv::LUT(labels, alpha_lut_, alpha8);
    alpha8.convertTo(alpha, CV_32FC1, 1.0 / 255.0);
    cv::subtract(cv::Scalar::all(1.0), alpha, inv_alpha);
    cv::Mat blended;
    cv::blendLinear(colour, output, alpha, inv_alpha, blended);
    blended.copyTo(output, alpha8 > 0);
}

void SegPostProcessor::DrawLegend(cv::Mat& output, const std::set<int>& class_ids) const {
    if (class_ids.empty()) return;

    constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double kFontScale = 0.45;
    constexpr int kSwatch = 12;
    constexpr int kPad = 6;
    constexpr int kMargin = 8;

    int text_width = 0;
    int text_height = 0;
    for (int class_id : class_ids) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(LabelName(class_id), kFont, kFontScale, 1, &baseline);
        text_width = std::max(text_width, size.width);
        text_height = std::max(text_height, size.height);
    }
    const int row_height = std::max(kSwatch, text_height) + kPad;
    const cv::Size box_size(kPad + kSwatch + kPad + text_width + kPad,
                            kPad + row_height * static_cast<int>(class_ids.size()));
    cv::Rect box(output.cols - box_size.width - kMargin, kMargin, box_size.width, box_size.height);
    box &= cv::Rect(0, 0, output.cols, output.rows);
    if (box.empty()) return;

    // Darkened translucent background keeps the legend readable on any frame
    cv::Mat roi = output(box);
    roi.convertTo(roi, -1, 0.4, 0.0);

    int y = box.y + kPad;
    for (int class_id : class_ids) {
        const cv::Rect swatch(box.x + kPad, y + (row_height - kPad - kSwatch) / 2, kSwatch, kSwatch);
        cv::rectangle(output, swatch, ClassColour(class_id), cv::FILLED);
        cv::rectangle(output, swatch, cv::Scalar(255, 255, 255), 1);
        const cv::Point origin(swatch.br().x + kPad, y + (row_height - kPad + text_height) / 2);
        cv::putText(output, LabelName(class_id), origin, kFont, kFontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += row_height;
    }
}
