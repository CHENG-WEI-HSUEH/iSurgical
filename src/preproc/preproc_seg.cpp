// Segmentation preprocessing: turns a BGR frame of any size into the model's 512x512 RGB float input,
// and maps model outputs back onto the original frame (RestoreMask).
#include "preproc_seg.hpp"

// std::min / std::max
#include <algorithm>
// std::round
#include <cmath>
// std::cerr for error messages 
#include <iostream>

// Store the model input size and preprocessing options; no allocation happens here.
SegPreprocessor::SegPreprocessor(int input_width, int input_height, const SegPreprocConfig& config)
    : input_width_(input_width),
      input_height_(input_height),
      config_(config) {}

// Find the region of the frame that actually contains the endoscope image (excluding the black border).
cv::Rect SegPreprocessor::FindContentRect(const cv::Mat& frame) {
    // The whole frame, used when cropping is off or fails.
    const cv::Rect full(0, 0, frame.cols, frame.rows);
    // Border cropping disabled: use the whole frame.
    if (!config_.crop_black_border) return full;

    // Convert to grayscale so brightness can be thresholded (gray_ is a reused member buffer).
    cv::cvtColor(frame, gray_, cv::COLOR_BGR2GRAY);
    // Pixels brighter than border_threshold become 255 (content), darker ones 0 (border).
    cv::threshold(gray_, gray_, config_.border_threshold, 255, cv::THRESH_BINARY);

    // Smallest rectangle that contains all non-zero (content) pixels.
    const cv::Rect content = cv::boundingRect(gray_);
    // If the found area is too small (e.g. a very dark frame), cropping is unreliable: use the whole frame.
    if (content.area() < config_.min_crop_ratio * full.area()) return full;
    // Use the detected content rectangle.
    return content;
}

// Preprocess one frame into dst (NHWC float32 RGB in [0, 1]) and record the geometry in info.
bool SegPreprocessor::Process(const cv::Mat& frame, float* dst, SegPreprocInfo& info) {
    // Only non-empty 8-bit 3-channel (BGR) images are supported.
    if (frame.empty() || frame.type() != CV_8UC3) {
        std::cerr << "Input frame must be a non-empty CV_8UC3 BGR image." << std::endl;
        return false;
    }

    // Remember the original frame size, so outputs can be restored to it later.
    info.frame_size = frame.size();
    // Remember the model input size (512x512).
    info.input_size = cv::Size(input_width_, input_height_);
    // Region of the frame that will be used (whole frame, or the area without the black border).
    info.crop = FindContentRect(frame);

    // A view of that region; no pixels are copied.
    const cv::Mat roi = frame(info.crop);

    // Decide where the region will be placed inside the 512x512 model input.
    if (config_.resize_mode == SegResizeMode::kLetterbox) {
        // Letterbox: one scale factor for both axes, so the aspect ratio is kept and the region fits entirely.
        const double scale = std::min(static_cast<double>(input_width_) / roi.cols,
                                      static_cast<double>(input_height_) / roi.rows);
        // Scaled width and height, at least 1 pixel.
        const int w = std::max(1, static_cast<int>(std::round(roi.cols * scale)));
        const int h = std::max(1, static_cast<int>(std::round(roi.rows * scale)));
        // Center the scaled image; the remaining area becomes padding.
        info.content = cv::Rect((input_width_ - w) / 2, (input_height_ - h) / 2, w, h);
    } else {
        // Stretch: fill the whole input, ignoring the aspect ratio.
        info.content = cv::Rect(0, 0, input_width_, input_height_);
    }

    // Allocate the 512x512 BGR canvas once; later calls reuse it.
    canvas_.create(input_height_, input_width_, CV_8UC3);
    // With letterbox there is padding: fill the canvas with pad_value first.
    if (info.content.size() != canvas_.size()) canvas_.setTo(cv::Scalar::all(config_.pad_value));
    // Resize the region to the target size with bilinear interpolation.
    cv::resize(roi, resized_, info.content.size(), 0, 0, cv::INTER_LINEAR);
    // Paste the resized image into its place on the canvas.
    resized_.copyTo(canvas_(info.content));
    // OpenCV images are BGR, the model expects RGB.
    cv::cvtColor(canvas_, canvas_, cv::COLOR_BGR2RGB);

    // Write straight into the caller's NHWC buffer
    // Wrap dst as a 512x512 3-channel float image (no copy), so convertTo writes directly into it.
    cv::Mat dst_mat(input_height_, input_width_, CV_32FC3, dst);
    // Convert uint8 [0, 255] to float [0, 1]; mean/std normalization is done inside the model.
    canvas_.convertTo(dst_mat, CV_32FC3, 1.0 / 255.0);
    return true;
}

// Map a model-resolution map (class ids or confidences) back to the original frame: the reverse of Process().
void SegPreprocessor::RestoreMask(const cv::Mat& model_mask, const SegPreprocInfo& info, cv::Mat& frame_mask,
                                  uint8_t fill_value) {
    // Model outputs may differ in resolution from the input; scale the content rect accordingly
    const double sx = static_cast<double>(model_mask.cols) / info.input_size.width;
    const double sy = static_cast<double>(model_mask.rows) / info.input_size.height;
    // Where the real image (not padding) lies in model_mask.
    cv::Rect content(static_cast<int>(std::round(info.content.x * sx)),
                     static_cast<int>(std::round(info.content.y * sy)),
                     static_cast<int>(std::round(info.content.width * sx)),
                     static_cast<int>(std::round(info.content.height * sy)));
    // Clip to the mask bounds in case rounding pushed it one pixel outside.
    content &= cv::Rect(0, 0, model_mask.cols, model_mask.rows);

    // Allocate the output at the original frame size, same type as the input (uint8 ids or float confidences).
    frame_mask.create(info.frame_size, model_mask.type());
    // If only part of the frame was used (border cropped), fill the rest with fill_value (0 = Background).
    if (info.crop.size() != info.frame_size) frame_mask.setTo(cv::Scalar::all(fill_value));
    // View of the output area that corresponds to the cropped region.
    cv::Mat crop_mask = frame_mask(info.crop);
    // Resize the content part back to the crop size and write it into place.
    // Nearest neighbour keeps class ids intact (no blending of ids like 2 and 4 into 3).
    cv::resize(model_mask(content), crop_mask, info.crop.size(), 0, 0, cv::INTER_NEAREST);
}
