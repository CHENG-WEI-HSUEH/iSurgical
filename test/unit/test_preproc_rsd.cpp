// Unit tests for src/preproc/preproc_rsd.cpp: RsdPreprocessor (eval-notebook crop + preprocess_step.onnx).
// The geometry (CropFrame) needs no model; Process runs the fake rsd_pre.onnx (image = frame / 255).

// Code under test
#include "preproc/preproc_rsd.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::vector
#include <vector>

// FakeModel, RealModel
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// The documented constants
TEST(RsdPreprocConstantsTest, MatchTheNotebook) {
    // CenterCrop(224)
    EXPECT_EQ(RsdPreprocessor::kInputSize, 224);
    // Decode short side 256
    EXPECT_EQ(RsdPreprocessor::kDecodeShortSide, 256);
    // Resize(255) = int(224 * 1.14)
    EXPECT_EQ(RsdPreprocessor::kResizeShortSide, 255);
    // 10-step colour history
    EXPECT_EQ(RsdPreprocessor::kHistLen, 10);
    // 3 x 224 x 224
    EXPECT_EQ(RsdPreprocessor::kImageSize, 3u * 224 * 224);
}

// CropFrame always produces a 224x224 RGB uint8 image, whatever the input size
class CropSizeTest : public ::testing::TestWithParam<cv::Size> {};

// Each input size
TEST_P(CropSizeTest, OutputIs224x224Rgb) {
    // Uniform BGR frame
    const cv::Mat bgr(GetParam(), CV_8UC3, cv::Scalar(10, 20, 30));
    // Output
    cv::Mat rgb;
    // Crop
    RsdPreprocessor::CropFrame(bgr, rgb);
    // Size
    EXPECT_EQ(rgb.size(), cv::Size(224, 224));
    // Type
    EXPECT_EQ(rgb.type(), CV_8UC3);
    // BGR (10, 20, 30) → RGB (30, 20, 10)
    EXPECT_EQ(rgb.at<cv::Vec3b>(112, 112), cv::Vec3b(30, 20, 10));
}

// Video sizes (landscape, HD, portrait, square, smaller than the crop)
INSTANTIATE_TEST_SUITE_P(Sizes, CropSizeTest,
                         ::testing::Values(cv::Size(854, 480), cv::Size(1920, 1080), cv::Size(240, 320),
                                           cv::Size(256, 256), cv::Size(100, 50)));

// The centre-crop offset uses Python round() (halves to even): 114.5 → 114, as torchvision does
TEST(RsdCropTest, CentreCropRoundsHalfToEven) {
    // 453x255: short side 255 (no decode downscale), Resize(255) keeps it 453x255 → offsets (114.5, 15.5)
    cv::Mat bgr(255, 453, CV_8UC3);
    // Encode the coordinates in the pixel values: R = x - 100, G = y
    for (int y = 0; y < bgr.rows; ++y) {
        // Each column
        for (int x = 0; x < bgr.cols; ++x) {
            // BGR order: B = 0, G = y, R = x - 100 (clamped)
            bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(0, static_cast<uint8_t>(y), cv::saturate_cast<uint8_t>(x - 100));
        }
    }
    // Output
    cv::Mat rgb;
    // Crop
    RsdPreprocessor::CropFrame(bgr, rgb);
    // Top-left of the crop comes from x = 114 (std::lround would give 115 → R = 15)
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 0)[0], 14);
    // and y = 16 (15.5 rounds to the even 16)
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 0)[1], 16);
}

// The centre of the frame stays at the centre of the crop
TEST(RsdCropTest, CentreIsPreserved) {
    // Black 854x480
    cv::Mat bgr(480, 854, CV_8UC3, cv::Scalar::all(0));
    // White dot in the middle
    cv::circle(bgr, cv::Point(427, 240), 20, cv::Scalar::all(255), cv::FILLED);
    // Output
    cv::Mat rgb;
    // Crop
    RsdPreprocessor::CropFrame(bgr, rgb);
    // Centre is white
    EXPECT_GT(rgb.at<cv::Vec3b>(112, 112)[0], 200);
    // Corner is black
    EXPECT_LT(rgb.at<cv::Vec3b>(0, 0)[0], 10);
}

// Fixture: an ONNX Runtime environment and the fake preprocessing model
class RsdPreprocTest : public ::testing::Test {
protected:
    // One environment per test
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "test_preproc_rsd"};
    // Output buffer
    std::vector<float> dst_ = std::vector<float>(RsdPreprocessor::kImageSize, -1.0f);
};

// A missing model file throws from the constructor
TEST_F(RsdPreprocTest, MissingModelThrows) {
    // Not on disk
    EXPECT_THROW(RsdPreprocessor(env_, "/no/such/model.onnx"), Ort::Exception);
}

// An empty frame is rejected before the model runs
TEST_F(RsdPreprocTest, EmptyFrameIsRejected) {
    // Fake model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre.onnx"));
    // Empty
    EXPECT_FALSE(pre.Process(cv::Mat(), dst_.data()));
    // Untouched
    EXPECT_EQ(dst_[0], -1.0f);
}

// A frame that is not 8-bit BGR is rejected
TEST_F(RsdPreprocTest, WrongTypeIsRejected) {
    // Fake model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre.onnx"));
    // Grey frame
    EXPECT_FALSE(pre.Process(cv::Mat(100, 100, CV_8UC1, cv::Scalar(5)), dst_.data()));
}

// Process writes the model output (planar CHW RGB) into dst: fake model = frame / 255
TEST_F(RsdPreprocTest, ProcessWritesPlanarRgb) {
    // Fake model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre.onnx"));
    // BGR (51, 102, 204) → R = 204, G = 102, B = 51
    const cv::Mat bgr(480, 854, CV_8UC3, cv::Scalar(51, 102, 204));
    // Run
    ASSERT_TRUE(pre.Process(bgr, dst_.data()));
    // One colour plane
    const size_t plane = 224 * 224;
    // Plane 0 = R
    EXPECT_NEAR(dst_[0], 204.0f / 255.0f, 1e-6);
    // Plane 1 = G
    EXPECT_NEAR(dst_[plane], 102.0f / 255.0f, 1e-6);
    // Plane 2 = B, last pixel
    EXPECT_NEAR(dst_[3 * plane - 1], 51.0f / 255.0f, 1e-6);
}

// The output matches CropFrame exactly, pixel by pixel
TEST_F(RsdPreprocTest, ProcessMatchesCropFrame) {
    // Fake model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre.onnx"));
    // Random frame
    cv::Mat bgr(360, 640, CV_8UC3);
    // Fixed seed
    cv::theRNG().state = 7;
    // Noise
    cv::randu(bgr, cv::Scalar::all(0), cv::Scalar::all(256));
    // Run
    ASSERT_TRUE(pre.Process(bgr, dst_.data()));
    // Reference crop
    cv::Mat rgb;
    // Geometry only
    RsdPreprocessor::CropFrame(bgr, rgb);
    // Largest difference
    float worst = 0.0f;
    // Every pixel and channel
    for (int y = 0; y < 224; ++y) {
        // Columns
        for (int x = 0; x < 224; ++x) {
            // Channels
            for (int c = 0; c < 3; ++c) {
                // CHW index
                const float got = dst_[static_cast<size_t>(c) * 224 * 224 + y * 224 + x];
                // Expected value
                const float want = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;
                // Track the worst
                worst = std::max(worst, std::abs(got - want));
            }
        }
    }
    // Exact up to float rounding
    EXPECT_LT(worst, 1e-6f);
}

// Many consecutive steps and a Reset work (the history is carried between calls)
TEST_F(RsdPreprocTest, ManyStepsAndReset) {
    // Fake model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre.onnx"));
    // Frame
    const cv::Mat bgr(240, 320, CV_8UC3, cv::Scalar::all(128));
    // More steps than the 10-step history
    for (int i = 0; i < 15; ++i) ASSERT_TRUE(pre.Process(bgr, dst_.data())) << "step " << i;
    // New video
    pre.Reset();
    // Still works
    EXPECT_TRUE(pre.Process(bgr, dst_.data()));
}

// A model without the expected output names fails at Process (caught, returns false)
TEST_F(RsdPreprocTest, ModelWithWrongOutputsFails) {
    // hist_count_out is missing in this model
    RsdPreprocessor pre(env_, FakeModel("rsd_pre_bad.onnx"));
    // Capture the error message
    ::testing::internal::CaptureStderr();
    // Fails
    EXPECT_FALSE(pre.Process(cv::Mat(100, 100, CV_8UC3, cv::Scalar::all(0)), dst_.data()));
    // Reported
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("RSD preprocessing failed"), std::string::npos);
}

// The real preprocess_step.onnx produces ImageNet-normalised values (skipped without data/)
TEST_F(RsdPreprocTest, RealModelNormalises) {
    // data/models/preprocess_step.onnx
    REQUIRE_REAL_MODEL(path, "preprocess_step.onnx");
    // Real model
    RsdPreprocessor pre(env_, path);
    // Mid-grey frame
    const cv::Mat bgr(480, 854, CV_8UC3, cv::Scalar::all(128));
    // Run
    ASSERT_TRUE(pre.Process(bgr, dst_.data()));
    // ImageNet normalisation keeps values roughly within [-3, 3]
    for (float v : dst_) ASSERT_TRUE(v > -3.0f && v < 3.0f) << v;
}

}  // namespace
