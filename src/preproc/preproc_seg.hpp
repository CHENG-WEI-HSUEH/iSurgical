#pragma once

#include <opencv2/opencv.hpp>

// Preprocessing for seg_model.onnx: BGR CV_8UC3 frame -> NHWC float32 RGB [0, 1] at the model input size.
// Mean/std normalization is done inside the model graph, so it is not applied here.
enum class SegResizeMode {
    kStretch,       // resize to the input size, ignoring aspect ratio
    kLetterbox,     // keep aspect ratio, pad the remainder with pad_value
};

struct SegPreprocConfig {
    SegResizeMode resize_mode = SegResizeMode::kStretch;
    uint8_t pad_value = 0;
    bool crop_black_border = false;     // crop the dark border around the endoscope view before resizing
    int border_threshold = 15;          // gray level below which a pixel counts as border
    float min_crop_ratio = 0.25f;       // ignore crops smaller than this fraction of the frame
};

// How a frame was mapped onto the model input, used to map model outputs back
struct SegPreprocInfo {
    cv::Size frame_size;                // original frame size
    cv::Size input_size;                // model input size
    cv::Rect crop;                      // region of the frame that was used
    cv::Rect content;                   // where that region landed in the model input
};

class SegPreprocessor {
public:
    SegPreprocessor(int input_width, int input_height, const SegPreprocConfig& config = SegPreprocConfig());
    // dst must hold input_height * input_width * 3 floats
    bool Process(const cv::Mat& frame, float* dst, SegPreprocInfo& info);
    // Map a model-resolution map (class ids or confidences) back to frame size with nearest-neighbour
    // resampling; pixels outside the crop get fill_value
    static void RestoreMask(const cv::Mat& model_mask, const SegPreprocInfo& info, cv::Mat& frame_mask,
                            uint8_t fill_value = 0);
private:
    cv::Rect FindContentRect(const cv::Mat& frame);

    int input_width_;
    int input_height_;
    SegPreprocConfig config_;
    cv::Mat gray_;
    cv::Mat resized_;
    cv::Mat canvas_;
};
