#pragma once

#include <array>
#include <set>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "infer/infer_seg.hpp"

// Settings from configs/lc_config.yaml
//   head_groups         : global class id of each head channel 1..n (channel 0 of every head is Background)
//   label_names         : name of each global class, index 0 = Background
//   kclass_colours      : RGBA in [0, 1] per global class (stored here as BGR 0-255 plus alpha);
//                         alpha is the overlay opacity of that class, 0 hides it, 1 is opaque
//   MaskThreshold       : a pixel keeps its class only if its softmax probability >= this
//   ConfidenceThreshold : a region is kept only if its mean pixel probability >= this
//   IoUThreshold        : YOLO NMS setting, not applicable to semantic segmentation (parsed, unused)
struct SegPostConfig {
    std::vector<std::vector<int>> head_groups;
    std::vector<std::string> label_names;
    std::vector<cv::Scalar> class_colours;
    std::vector<float> class_alphas;        // 1.0 when an entry has only RGB
    int num_classes = 0;
    float mask_threshold = 0.5f;
    float confidence_threshold = 0.25f;
    float iou_threshold = 0.5f;

    // Not in lc_config.yaml; optional keys MorphKernelSize / MinAreaRatio override them
    int morph_kernel_size = 5;          // open + close kernel, 0 disables morphology
    float min_area_ratio = 0.001f;      // components smaller than this fraction of the frame are removed

    // Reads the `procedure` section of the file, or the whole file if `procedure` is empty.
    // Throws std::runtime_error if the file is missing, malformed or inconsistent with the model heads
    static SegPostConfig FromYaml(const std::string& path, const std::string& procedure = "");
    void Validate() const;
};

// One connected component of a single class
struct SegRegion {
    int class_id;                       // global class id (index into label_names)
    int head;                           // head that produced it
    int area;                           // pixels
    float area_ratio;                   // area / frame area
    float confidence;                   // mean softmax probability over the region
    cv::Rect bbox;
    cv::Point2f centroid;
    std::vector<cv::Point> contour;     // outer contour
};

// Segmentation masks and regions for a single head, after thresholding and cleanup
struct SegHeadOutput {
    cv::Mat mask;                       // CV_8UC1 global class id after thresholding and cleanup, 0 = Background
    std::vector<SegRegion> regions;     // sorted by area, largest first
};

struct SegPostResult {
    std::array<SegHeadOutput, SegResult::kNumHeads> heads;
    cv::Mat label_map;                  // CV_8UC1 all heads fused, overlaps go to the most confident head
    std::array<float, SegResult::kNumScores> correlation_scores;
};

class SegPostProcessor {
public:
    explicit SegPostProcessor(const SegPostConfig& config);
    void Process(const SegResult& seg_result, SegPostResult& post_result) const;
    // Blend classes onto a BGR frame, outline each region and draw a fixed top-right legend of all
    // classes (or of the selected head's classes); head < 0 draws the fused map
    void Draw(const cv::Mat& frame, const SegPostResult& result, cv::Mat& output, int head = -1) const;
    const std::string& LabelName(int class_id) const;
    cv::Scalar ClassColour(int class_id) const;
    const SegPostConfig& Config() const { return config_; }
private:
    void ProcessHead(int head, const cv::Mat& mask, const cv::Mat& confidence, SegHeadOutput& head_output) const;
    void FuseHeads(const SegResult& seg_result, SegPostResult& post_result) const;
    void BlendLabels(const cv::Mat& labels, cv::Mat& output) const;
    void DrawLegend(cv::Mat& output, const std::set<int>& class_ids) const;

    SegPostConfig config_;
    cv::Mat morph_kernel_;
    cv::Mat colour_lut_;                    // 1x256 CV_8UC3, class id -> BGR
    cv::Mat alpha_lut_;                     // 1x256 CV_8UC1, class id -> alpha * 255
};
