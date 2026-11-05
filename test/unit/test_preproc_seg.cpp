// Unit tests for src/preproc/preproc_seg.cpp: SegPreprocessor (frame → 512x512 RGB float) and RestoreMask (model → frame).

// Code under test
#include "preproc/preproc_seg.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::min_element / std::max_element
#include <algorithm>
// std::vector
#include <vector>

// Anonymous namespace
namespace {

// Model input size
constexpr int kW = 512;
// Model input size
constexpr int kH = 512;

// Fixture: an output buffer and helpers to read a pixel from it
class SegPreprocTest : public ::testing::Test {
protected:
    // NHWC float buffer for one 512x512 RGB image
    std::vector<float> dst_ = std::vector<float>(static_cast<size_t>(kW) * kH * 3, -1.0f);
    // Geometry recorded by Process
    SegPreprocInfo info_;
    // R, G or B (0, 1, 2) of model pixel (x, y)
    float At(int x, int y, int c) const { return dst_[(static_cast<size_t>(y) * kW + x) * 3 + c]; }
};

// An empty frame is rejected
TEST_F(SegPreprocTest, EmptyFrameIsRejected) {
    // Default config
    SegPreprocessor pre(kW, kH);
    // Empty Mat
    EXPECT_FALSE(pre.Process(cv::Mat(), dst_.data(), info_));
}

// Only 8-bit 3-channel frames are accepted
class WrongTypeTest : public SegPreprocTest, public ::testing::WithParamInterface<int> {};

// Each wrong type
TEST_P(WrongTypeTest, IsRejected) {
    // Default config
    SegPreprocessor pre(kW, kH);
    // 64x64 frame of the given type
    const cv::Mat frame(64, 64, GetParam(), cv::Scalar::all(0));
    // Rejected
    EXPECT_FALSE(pre.Process(frame, dst_.data(), info_));
    // Buffer untouched
    EXPECT_EQ(dst_[0], -1.0f);
}

// Gray, BGRA, float BGR, 16-bit BGR
INSTANTIATE_TEST_SUITE_P(Types, WrongTypeTest, ::testing::Values(CV_8UC1, CV_8UC4, CV_32FC3, CV_16UC3));

// Stretch (default): the whole frame fills the whole input, and info records the geometry
TEST_F(SegPreprocTest, StretchFillsWholeInput) {
    // Default = stretch
    SegPreprocessor pre(kW, kH);
    // 854x480 like video01.mp4
    const cv::Mat frame(480, 854, CV_8UC3, cv::Scalar(10, 20, 30));
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Original size
    EXPECT_EQ(info_.frame_size, cv::Size(854, 480));
    // Model size
    EXPECT_EQ(info_.input_size, cv::Size(kW, kH));
    // Whole frame used
    EXPECT_EQ(info_.crop, cv::Rect(0, 0, 854, 480));
    // Whole input covered
    EXPECT_EQ(info_.content, cv::Rect(0, 0, kW, kH));
}

// BGR is converted to RGB and scaled to [0, 1]
TEST_F(SegPreprocTest, ConvertsBgrToRgbInUnitRange) {
    // Default config
    SegPreprocessor pre(kW, kH);
    // BGR order: B = 255, G = 128, R = 0
    const cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(255, 128, 0));
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // R = 0
    EXPECT_FLOAT_EQ(At(256, 256, 0), 0.0f);
    // G = 128 / 255
    EXPECT_NEAR(At(256, 256, 1), 128.0f / 255.0f, 1e-6);
    // B = 1
    EXPECT_FLOAT_EQ(At(256, 256, 2), 1.0f);
}

// Every output value is inside [0, 1]
TEST_F(SegPreprocTest, AllValuesInUnitRange) {
    // Default config
    SegPreprocessor pre(kW, kH);
    // Random noise
    cv::Mat frame(300, 400, CV_8UC3);
    // Fixed seed
    cv::theRNG().state = 42;
    // Uniform 0..255
    cv::randu(frame, cv::Scalar::all(0), cv::Scalar::all(256));
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Smallest
    const float lo = *std::min_element(dst_.begin(), dst_.end());
    // Largest
    const float hi = *std::max_element(dst_.begin(), dst_.end());
    // In range
    EXPECT_GE(lo, 0.0f);
    // In range
    EXPECT_LE(hi, 1.0f);
}

// Letterbox keeps the aspect ratio and centres the image; the rest is pad_value
TEST_F(SegPreprocTest, LetterboxCentresAndPads) {
    // Letterbox, pad grey 100
    SegPreprocConfig config;
    // Mode
    config.resize_mode = SegResizeMode::kLetterbox;
    // Pad colour
    config.pad_value = 100;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // 854x480 white frame
    const cv::Mat frame(480, 854, CV_8UC3, cv::Scalar::all(255));
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Scale 512/854 → 512x288, centred vertically at y = 112
    EXPECT_EQ(info_.content, cv::Rect(0, 112, 512, 288));
    // Top padding has pad_value
    EXPECT_NEAR(At(256, 10, 0), 100.0f / 255.0f, 1e-6);
    // Image area is white
    EXPECT_FLOAT_EQ(At(256, 256, 0), 1.0f);
    // Bottom padding
    EXPECT_NEAR(At(256, 505, 2), 100.0f / 255.0f, 1e-6);
}

// A portrait frame is letterboxed horizontally
TEST_F(SegPreprocTest, LetterboxPortraitPadsSides) {
    // Letterbox
    SegPreprocConfig config;
    // Mode
    config.resize_mode = SegResizeMode::kLetterbox;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // 200 wide, 400 high
    const cv::Mat frame(400, 200, CV_8UC3, cv::Scalar::all(255));
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // 256x512 centred horizontally
    EXPECT_EQ(info_.content, cv::Rect(128, 0, 256, 512));
    // Left padding is 0 (default pad_value)
    EXPECT_FLOAT_EQ(At(10, 256, 0), 0.0f);
}

// A square frame letterboxed has no padding at all
TEST_F(SegPreprocTest, LetterboxSquareHasNoPadding) {
    // Letterbox
    SegPreprocConfig config;
    // Mode
    config.resize_mode = SegResizeMode::kLetterbox;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // Square
    ASSERT_TRUE(pre.Process(cv::Mat(300, 300, CV_8UC3, cv::Scalar::all(9)), dst_.data(), info_));
    // Full input
    EXPECT_EQ(info_.content, cv::Rect(0, 0, kW, kH));
}

// crop_black_border finds the bright endoscope area inside a black frame
TEST_F(SegPreprocTest, CropBlackBorderFindsContent) {
    // Cropping on
    SegPreprocConfig config;
    // Enabled
    config.crop_black_border = true;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // Black 640x480 frame
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
    // Bright content area (more than 25% of the frame)
    cv::rectangle(frame, cv::Rect(100, 50, 400, 380), cv::Scalar(200, 150, 100), cv::FILLED);
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Exactly the bright rectangle
    EXPECT_EQ(info_.crop, cv::Rect(100, 50, 400, 380));
    // It now fills the model input (stretch), so a corner of the input is bright, not black
    EXPECT_GT(At(1, 1, 0), 0.3f);
}

// Content smaller than min_crop_ratio falls back to the whole frame
TEST_F(SegPreprocTest, CropTooSmallFallsBackToWholeFrame) {
    // Cropping on, default min ratio 0.25
    SegPreprocConfig config;
    // Enabled
    config.crop_black_border = true;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // Black frame
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
    // Tiny bright spot (much less than 25%)
    cv::rectangle(frame, cv::Rect(10, 10, 50, 50), cv::Scalar::all(255), cv::FILLED);
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Whole frame
    EXPECT_EQ(info_.crop, cv::Rect(0, 0, 640, 480));
}

// Pixels at or below border_threshold count as border
TEST_F(SegPreprocTest, BorderThresholdIsRespected) {
    // Cropping on, threshold 50
    SegPreprocConfig config;
    // Enabled
    config.crop_black_border = true;
    // Raised threshold
    config.border_threshold = 50;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // Dark grey 40 everywhere (would be content with the default threshold 15)
    cv::Mat frame(200, 200, CV_8UC3, cv::Scalar::all(40));
    // Bright centre
    cv::rectangle(frame, cv::Rect(20, 20, 160, 160), cv::Scalar::all(255), cv::FILLED);
    // Process
    ASSERT_TRUE(pre.Process(frame, dst_.data(), info_));
    // Only the bright centre
    EXPECT_EQ(info_.crop, cv::Rect(20, 20, 160, 160));
}

// The preprocessor can be reused for frames of different sizes
TEST_F(SegPreprocTest, ReusedForDifferentFrameSizes) {
    // Letterbox (reuses the canvas)
    SegPreprocConfig config;
    // Mode
    config.resize_mode = SegResizeMode::kLetterbox;
    // Preprocessor
    SegPreprocessor pre(kW, kH, config);
    // Wide frame
    ASSERT_TRUE(pre.Process(cv::Mat(100, 400, CV_8UC3, cv::Scalar::all(255)), dst_.data(), info_));
    // Wide content
    EXPECT_EQ(info_.content.height, 128);
    // Then a square one
    ASSERT_TRUE(pre.Process(cv::Mat(50, 50, CV_8UC3, cv::Scalar::all(255)), dst_.data(), info_));
    // Full content, no stale padding
    EXPECT_EQ(info_.content, cv::Rect(0, 0, kW, kH));
    // Top-left now white
    EXPECT_FLOAT_EQ(At(0, 0, 0), 1.0f);
}

// RestoreMask after stretch maps the model mask back to the frame size with nearest-neighbour
TEST(RestoreMaskTest, StretchRoundTripKeepsClassIds) {
    // Geometry of a stretched 854x480 frame
    SegPreprocInfo info{cv::Size(854, 480), cv::Size(512, 512), cv::Rect(0, 0, 854, 480), cv::Rect(0, 0, 512, 512)};
    // Model mask: left half class 2, right half class 4
    cv::Mat model(512, 512, CV_8UC1, cv::Scalar(2));
    // Right half
    model(cv::Rect(256, 0, 256, 512)).setTo(4);
    // Output
    cv::Mat frame_mask;
    // Restore
    SegPreprocessor::RestoreMask(model, info, frame_mask);
    // Frame size
    EXPECT_EQ(frame_mask.size(), cv::Size(854, 480));
    // Same type
    EXPECT_EQ(frame_mask.type(), CV_8UC1);
    // Left side
    EXPECT_EQ(frame_mask.at<uint8_t>(240, 100), 2);
    // Right side
    EXPECT_EQ(frame_mask.at<uint8_t>(240, 800), 4);
    // Only 2 and 4 (no interpolated 3)
    cv::Mat threes = (frame_mask == 3);
    // None
    EXPECT_EQ(cv::countNonZero(threes), 0);
}

// RestoreMask after letterbox drops the padding rows
TEST(RestoreMaskTest, LetterboxIgnoresPadding) {
    // Content occupies rows 112..399 of the model input
    SegPreprocInfo info{cv::Size(854, 480), cv::Size(512, 512), cv::Rect(0, 0, 854, 480), cv::Rect(0, 112, 512, 288)};
    // Padding rows are class 9, content is class 1
    cv::Mat model(512, 512, CV_8UC1, cv::Scalar(9));
    // Content
    model(info.content).setTo(1);
    // Output
    cv::Mat frame_mask;
    // Restore
    SegPreprocessor::RestoreMask(model, info, frame_mask);
    // Every frame pixel is class 1 (no padding leaked in)
    EXPECT_EQ(cv::countNonZero(frame_mask == 1), 854 * 480);
}

// Pixels outside the crop get fill_value
TEST(RestoreMaskTest, OutsideCropGetsFillValue) {
    // Crop 100..499 x 50..429 of a 640x480 frame
    SegPreprocInfo info{cv::Size(640, 480), cv::Size(512, 512), cv::Rect(100, 50, 400, 380), cv::Rect(0, 0, 512, 512)};
    // Model says class 3 everywhere
    cv::Mat model(512, 512, CV_8UC1, cv::Scalar(3));
    // Output
    cv::Mat frame_mask;
    // Restore with fill 7
    SegPreprocessor::RestoreMask(model, info, frame_mask, 7);
    // Outside the crop
    EXPECT_EQ(frame_mask.at<uint8_t>(5, 5), 7);
    // Inside the crop
    EXPECT_EQ(frame_mask.at<uint8_t>(200, 300), 3);
    // Count of class 3 = crop area
    EXPECT_EQ(cv::countNonZero(frame_mask == 3), 400 * 380);
}

// A model output at a different resolution than the input (e.g. 128x128) is scaled correctly
TEST(RestoreMaskTest, LowerResolutionModelOutput) {
    // Letterboxed geometry
    SegPreprocInfo info{cv::Size(854, 480), cv::Size(512, 512), cv::Rect(0, 0, 854, 480), cv::Rect(0, 112, 512, 288)};
    // 128x128 output: content is rows 28..99
    cv::Mat model(128, 128, CV_8UC1, cv::Scalar(9));
    // Content rows at 1/4 scale
    model(cv::Rect(0, 28, 128, 72)).setTo(1);
    // Output
    cv::Mat frame_mask;
    // Restore
    SegPreprocessor::RestoreMask(model, info, frame_mask);
    // All content
    EXPECT_EQ(cv::countNonZero(frame_mask == 1), 854 * 480);
}

// Float confidence maps keep their type and values
TEST(RestoreMaskTest, FloatMapsAreSupported) {
    // Stretched geometry
    SegPreprocInfo info{cv::Size(100, 50), cv::Size(512, 512), cv::Rect(0, 0, 100, 50), cv::Rect(0, 0, 512, 512)};
    // Confidence 0.75 everywhere
    cv::Mat model(512, 512, CV_32FC1, cv::Scalar(0.75));
    // Output
    cv::Mat frame_mask;
    // Restore
    SegPreprocessor::RestoreMask(model, info, frame_mask);
    // Float
    EXPECT_EQ(frame_mask.type(), CV_32FC1);
    // Value kept
    EXPECT_FLOAT_EQ(frame_mask.at<float>(25, 50), 0.75f);
}

// Process followed by RestoreMask is an identity on the class layout (end-to-end geometry check)
TEST(RestoreMaskTest, ProcessThenRestoreMatchesFrameLayout) {
    // Letterbox + crop to exercise every transform
    SegPreprocConfig config;
    // Letterbox
    config.resize_mode = SegResizeMode::kLetterbox;
    // Crop
    config.crop_black_border = true;
    // Preprocessor
    SegPreprocessor pre(512, 512, config);
    // Black frame with a bright content area
    cv::Mat frame(480, 854, CV_8UC3, cv::Scalar::all(0));
    // Content
    cv::rectangle(frame, cv::Rect(100, 40, 600, 400), cv::Scalar::all(255), cv::FILLED);
    // Buffer
    std::vector<float> dst(512 * 512 * 3);
    // Geometry
    SegPreprocInfo info;
    // Process
    ASSERT_TRUE(pre.Process(frame, dst.data(), info));
    // Pretend the model labels exactly the content area as class 1
    cv::Mat model(512, 512, CV_8UC1, cv::Scalar(0));
    // Content
    model(info.content).setTo(1);
    // Back to the frame
    cv::Mat frame_mask;
    // Restore
    SegPreprocessor::RestoreMask(model, info, frame_mask);
    // Class 1 exactly where the frame was bright
    cv::Mat bright;
    // Gray
    cv::cvtColor(frame, bright, cv::COLOR_BGR2GRAY);
    // Same pixel set
    EXPECT_EQ(cv::countNonZero((frame_mask == 1) != (bright > 0)), 0);
}

}  // namespace
