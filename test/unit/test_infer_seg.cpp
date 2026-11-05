// Unit tests for src/infer/infer_seg.cpp: SegInferenceEngine with the fake seg.onnx (see make_fake_models.py).
// Fake model: red → head 0 class 1, green → head 0 class 2, blue → head 0 class 3, yellow → head 1 class 1.

// Code under test
#include "infer/infer_seg.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::exp
#include <cmath>
// std::runtime_error
#include <stdexcept>
// std::string
#include <string>

// FakeModel, ColourFrame
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// Softmax probability of the winner for logits {0.5 (Background), 3 (winner), others}
float Softmax(std::initializer_list<float> others) {
    // exp(l - 3) for every logit, the winner contributes 1
    float denom = std::exp(0.5f - 3.0f) + 1.0f;
    // The others
    for (float l : others) denom += std::exp(l - 3.0f);
    // Winner probability
    return 1.0f / denom;
}

// Head-0 confidence of a pure red / green / blue pixel: other logits -1, -1, -10
const float kPureColourConf = Softmax({-1.0f, -1.0f, -10.0f});
// Head-0 confidence of yellow: red and green tie at 3 (class 1 wins the tie), blue -1, unused -10
const float kYellowHead0Conf = Softmax({3.0f, -1.0f, -10.0f});
// Head-1 confidence of yellow: 4 + 4 - 5 = 3 vs Background 0.5 and -10
const float kYellowHead1Conf = Softmax({-10.0f});

// The model layout constants
TEST(SegEngineConstantsTest, ModelLayout) {
    // 512x512x3 input
    EXPECT_EQ(SegInferenceEngine::kInputWidth, 512);
    // Height
    EXPECT_EQ(SegInferenceEngine::kInputHeight, 512);
    // Channels
    EXPECT_EQ(SegInferenceEngine::kInputChannels, 3);
    // 4 heads with 5 / 3 / 4 / 7 channels
    EXPECT_EQ(SegInferenceEngine::kHeadClasses, (std::array<int, 4>{5, 3, 4, 7}));
    // 7 correlation scores
    EXPECT_EQ(SegInferenceEngine::kNumScores, 7);
}

// A missing model file throws from the constructor
TEST(SegEngineTest, MissingModelThrows) {
    // Not on disk
    EXPECT_THROW(SegInferenceEngine("/no/such/model.onnx", false), Ort::Exception);
}

// Models with the wrong inputs / outputs are rejected by CheckModelIO
class BadModelTest : public ::testing::TestWithParam<std::pair<const char*, const char*>> {};

// Each broken model
TEST_P(BadModelTest, IsRejected) {
    // Must throw runtime_error with the expected text
    try {
        // Construct (CPU)
        SegInferenceEngine engine(FakeModel(GetParam().first), false);
        // Did not throw
        FAIL() << "no exception";
    // Expected
    } catch (const std::runtime_error& e) {
        // Message
        EXPECT_NE(std::string(e.what()).find(GetParam().second), std::string::npos) << e.what();
    }
}

// The broken variants and the error each one triggers
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Models,
    // Fixture
    BadModelTest,
    // Rows: file, message
    ::testing::Values(std::make_pair("seg_bad_input.onnx", "expected a single input named 'image'"),
                      std::make_pair("seg_missing_scores.onnx", "Unexpected model output count, expected 5"),
                      std::make_pair("seg_bad_head_name.onnx", "Unexpected model output name: mask_0"),
                      std::make_pair("seg_bad_scores_name.onnx", "Unexpected model output name: scores")),
    // Row name from the file name
    [](const ::testing::TestParamInfo<std::pair<const char*, const char*>>& info) {
        // File name without ".onnx"
        std::string name = info.param.first;
        // Strip the extension
        name = name.substr(0, name.find('.'));
        // Remove characters gtest does not allow
        for (char& c : name) if (c == '_') c = 'X';
        // Use it
        return name;
    });

// Fixture: the fake model on CPU
class SegEngineCpuTest : public ::testing::Test {
protected:
    // CPU engine
    SegInferenceEngine engine_{FakeModel("seg.onnx"), false};
    // Output
    SegResult result_;
};

// CPU was requested, so CUDA is not used
TEST_F(SegEngineCpuTest, UsesCpuWhenAsked) {
    // Not CUDA
    EXPECT_FALSE(engine_.UsingCuda());
}

// An empty frame fails before inference
TEST_F(SegEngineCpuTest, EmptyFrameFails) {
    // Empty
    EXPECT_FALSE(engine_.Run(cv::Mat(), result_));
}

// Argmax gives the right head-local class for every colour, Background elsewhere
TEST_F(SegEngineCpuTest, ArgmaxPerHead) {
    // Red square top-left, yellow square bottom-right, black elsewhere
    ASSERT_TRUE(engine_.Run(ColourFrame(), result_));
    // Head 0: red → class 1
    EXPECT_EQ(result_.masks[0].at<uint8_t>(128, 128), 1);
    // Head 0: yellow → class 1 (tie with class 2, first wins)
    EXPECT_EQ(result_.masks[0].at<uint8_t>(384, 384), 1);
    // Head 0: black → Background
    EXPECT_EQ(result_.masks[0].at<uint8_t>(10, 500), 0);
    // Head 1: only yellow
    EXPECT_EQ(result_.masks[1].at<uint8_t>(384, 384), 1);
    // Head 1: red is Background
    EXPECT_EQ(result_.masks[1].at<uint8_t>(128, 128), 0);
    // Heads 2 and 3 are all Background
    EXPECT_EQ(cv::countNonZero(result_.masks[2]) + cv::countNonZero(result_.masks[3]), 0);
}

// Green and blue select head-0 classes 2 and 3
TEST_F(SegEngineCpuTest, GreenAndBlueClasses) {
    // Left half green, right half blue (BGR)
    cv::Mat frame(512, 512, CV_8UC3, cv::Scalar(0, 255, 0));
    // Right half blue
    frame(cv::Rect(256, 0, 256, 512)).setTo(cv::Scalar(255, 0, 0));
    // Run
    ASSERT_TRUE(engine_.Run(frame, result_));
    // Green
    EXPECT_EQ(result_.masks[0].at<uint8_t>(256, 100), 2);
    // Blue
    EXPECT_EQ(result_.masks[0].at<uint8_t>(256, 400), 3);
}

// Confidence is the softmax probability of the winning class; 0 for Background
TEST_F(SegEngineCpuTest, ConfidenceIsSoftmax) {
    // Run
    ASSERT_TRUE(engine_.Run(ColourFrame(), result_));
    // Red pixel, head 0
    EXPECT_NEAR(result_.confidences[0].at<float>(128, 128), kPureColourConf, 1e-4);
    // Yellow pixel, head 0 (two classes tie)
    EXPECT_NEAR(result_.confidences[0].at<float>(384, 384), kYellowHead0Conf, 1e-4);
    // Yellow pixel, head 1
    EXPECT_NEAR(result_.confidences[1].at<float>(384, 384), kYellowHead1Conf, 1e-4);
    // Background pixel
    EXPECT_FLOAT_EQ(result_.confidences[0].at<float>(10, 500), 0.0f);
}

// The 7 correlation scores are copied as-is
TEST_F(SegEngineCpuTest, CorrelationScoresCopied) {
    // Run
    ASSERT_TRUE(engine_.Run(ColourFrame(), result_));
    // 0.1 .. 0.7
    for (int i = 0; i < 7; ++i) EXPECT_NEAR(result_.correlation_scores[i], 0.1f * (i + 1), 1e-6) << i;
}

// Masks are mapped back to the original frame size (here 854x480, stretched)
TEST_F(SegEngineCpuTest, OutputsHaveFrameSize) {
    // video01-sized frame with a red square at a known place
    const cv::Mat frame = ColourFrame(cv::Size(854, 480));
    // Run
    ASSERT_TRUE(engine_.Run(frame, result_));
    // Every head
    for (int h = 0; h < SegResult::kNumHeads; ++h) {
        // Mask
        EXPECT_EQ(result_.masks[h].size(), frame.size()) << h;
        // Confidence
        EXPECT_EQ(result_.confidences[h].size(), frame.size()) << h;
    }
    // Centre of the red square (x 107..320, y 60..180) is class 1
    EXPECT_EQ(result_.masks[0].at<uint8_t>(120, 213), 1);
}

// The engine can be run many times; results are identical
TEST_F(SegEngineCpuTest, RepeatedRunsAreDeterministic) {
    // First run
    ASSERT_TRUE(engine_.Run(ColourFrame(), result_));
    // Keep the mask
    const cv::Mat first = result_.masks[0].clone();
    // Several more
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(engine_.Run(ColourFrame(), result_));
    // Same
    EXPECT_EQ(cv::countNonZero(first != result_.masks[0]), 0);
}

// Letterbox preprocessing: padding never shows up as a class
TEST(SegEngineLetterboxTest, PaddingStaysBackground) {
    // Letterbox config, pad value 255 (white, which the fake model would see as "yellow + blue")
    SegPreprocConfig config;
    // Mode
    config.resize_mode = SegResizeMode::kLetterbox;
    // White padding
    config.pad_value = 255;
    // Engine
    SegInferenceEngine engine(FakeModel("seg.onnx"), false, 0, config);
    // All-black wide frame
    const cv::Mat frame(240, 854, CV_8UC3, cv::Scalar::all(0));
    // Output
    SegResult result;
    // Run
    ASSERT_TRUE(engine.Run(frame, result));
    // The white padding is cut away by RestoreMask: no class in the frame
    EXPECT_EQ(cv::countNonZero(result.masks[0]), 0);
}

// Asking for CUDA either runs on the GPU or falls back to CPU; the result is the same as on CPU
TEST(SegEngineCudaTest, CudaRequestGivesSameResultAsCpu) {
    // CUDA requested (falls back to CPU if the provider or the first GPU run fails)
    SegInferenceEngine gpu(FakeModel("seg.onnx"), true);
    // CPU reference
    SegInferenceEngine cpu(FakeModel("seg.onnx"), false);
    // Outputs
    SegResult a, b;
    // Both succeed
    ASSERT_TRUE(gpu.Run(ColourFrame(), a));
    // CPU
    ASSERT_TRUE(cpu.Run(ColourFrame(), b));
    // Same classes
    EXPECT_EQ(cv::countNonZero(a.masks[0] != b.masks[0]), 0);
    // Same confidence (GPU math may differ in the last bits)
    EXPECT_NEAR(a.confidences[0].at<float>(128, 128), b.confidences[0].at<float>(128, 128), 1e-4);
    // Report which device was used
    std::cout << "[ info ] CUDA requested, using " << (gpu.UsingCuda() ? "GPU" : "CPU (fallback)") << std::endl;
}

}  // namespace
