// Unit tests for src/infer/infer_rsd.cpp: RsdInferenceEngine with the fake rsd_pre.onnx + rsd.onnx.
// Fake model: phase 2 with 0.7; pred_rsd = 60 × (steps already taken) + 60; pred_progress = mean(frame / 255).

// Code under test
#include "infer/infer_rsd.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::accumulate
#include <numeric>
// std::string
#include <string>

// FakeModel
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// Mid-grey frame: progress = 128 / 255
const cv::Mat kGrey(480, 854, CV_8UC3, cv::Scalar::all(128));

// Fixture: the fake models on CPU
class RsdEngineTest : public ::testing::Test {
protected:
    // CPU engine
    RsdInferenceEngine engine_{FakeModel("rsd_pre.onnx"), FakeModel("rsd.onnx"), false};
    // Output
    RsdResult result_;
};

// Result defaults before any step
TEST(RsdResultTest, Defaults) {
    // Fresh
    const RsdResult r;
    // No phase
    EXPECT_EQ(r.phase, -1);
    // 7 phases
    EXPECT_EQ(RsdResult::kNumPhases, 7);
    // 512-d embedding
    EXPECT_EQ(RsdResult::kEmbedDim, 512);
    // No steps
    EXPECT_EQ(r.step_index, 0);
}

// CPU was requested
TEST_F(RsdEngineTest, UsesCpuWhenAsked) {
    // Not CUDA
    EXPECT_FALSE(engine_.UsingCuda());
}

// One step fills every field of the result
TEST_F(RsdEngineTest, OneStepFillsResult) {
    // Run at 12.5 s source time
    ASSERT_TRUE(engine_.Run(kGrey, 12.5, result_));
    // argmax phase
    EXPECT_EQ(result_.phase, 2);
    // Its probability
    EXPECT_NEAR(result_.phase_confidence, 0.7f, 1e-6);
    // All probabilities copied
    EXPECT_NEAR(std::accumulate(result_.phase_probs.begin(), result_.phase_probs.end(), 0.0f), 1.0f, 1e-5);
    // First step: state mean 0 → 60 s
    EXPECT_NEAR(result_.rsd_s, 60.0f, 1e-4);
    // Mean brightness
    EXPECT_NEAR(result_.progress, 128.0f / 255.0f, 1e-4);
    // Embedding z_t[i] = i / 512
    EXPECT_NEAR(result_.z_t[256], 0.5f, 1e-6);
    // Time passed through
    EXPECT_DOUBLE_EQ(result_.video_time_s, 12.5);
    // First step
    EXPECT_EQ(result_.step_index, 1);
}

// The recurrent state is carried from one step to the next
TEST_F(RsdEngineTest, StateIsCarriedBetweenSteps) {
    // Five steps
    for (int step = 1; step <= 5; ++step) {
        // Run
        ASSERT_TRUE(engine_.Run(kGrey, step, result_));
        // 60, 120, 180, ... (conv_state grows by 1 each step)
        EXPECT_NEAR(result_.rsd_s, 60.0f * step, 1e-3) << step;
        // Step counter
        EXPECT_EQ(result_.step_index, step);
    }
}

// Reset clears the temporal state and the step counter
TEST_F(RsdEngineTest, ResetClearsState) {
    // Three steps
    for (int i = 0; i < 3; ++i) engine_.Run(kGrey, i, result_);
    // New video
    engine_.Reset();
    // Next step
    ASSERT_TRUE(engine_.Run(kGrey, 0.0, result_));
    // Back to the first-step values
    EXPECT_NEAR(result_.rsd_s, 60.0f, 1e-4);
    // Counter restarted
    EXPECT_EQ(result_.step_index, 1);
}

// A black frame gives progress 0, a white one progress 1 (the image reaches the model)
TEST_F(RsdEngineTest, ImageReachesModel) {
    // Black
    ASSERT_TRUE(engine_.Run(cv::Mat(240, 320, CV_8UC3, cv::Scalar::all(0)), 0.0, result_));
    // 0
    EXPECT_NEAR(result_.progress, 0.0f, 1e-6);
    // White
    ASSERT_TRUE(engine_.Run(cv::Mat(240, 320, CV_8UC3, cv::Scalar::all(255)), 1.0, result_));
    // 1
    EXPECT_NEAR(result_.progress, 1.0f, 1e-6);
}

// An empty frame fails and does not advance the step counter
TEST_F(RsdEngineTest, EmptyFrameFailsWithoutAdvancing) {
    // One good step
    ASSERT_TRUE(engine_.Run(kGrey, 0.0, result_));
    // Bad frame
    EXPECT_FALSE(engine_.Run(cv::Mat(), 1.0, result_));
    // Next good step is step 2, with the state of step 2
    ASSERT_TRUE(engine_.Run(kGrey, 2.0, result_));
    // Counter
    EXPECT_EQ(result_.step_index, 2);
    // State
    EXPECT_NEAR(result_.rsd_s, 120.0f, 1e-3);
}

// A main model without phase_probs fails at Run (reported on stderr)
TEST(RsdEngineBadModelTest, MissingOutputFailsRun) {
    // phase_probs renamed in rsd_bad.onnx
    RsdInferenceEngine engine(FakeModel("rsd_pre.onnx"), FakeModel("rsd_bad.onnx"), false);
    // Output
    RsdResult result;
    // Capture the message
    ::testing::internal::CaptureStderr();
    // Fails
    EXPECT_FALSE(engine.Run(kGrey, 0.0, result));
    // Reported
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("RSD inference failed"), std::string::npos);
    // Nothing written
    EXPECT_EQ(result.phase, -1);
}

// A broken preprocessing model fails at Run too
TEST(RsdEngineBadModelTest, BrokenPreprocessingFailsRun) {
    // hist_count_out missing
    RsdInferenceEngine engine(FakeModel("rsd_pre_bad.onnx"), FakeModel("rsd.onnx"), false);
    // Output
    RsdResult result;
    // Silence the message
    ::testing::internal::CaptureStderr();
    // Fails
    EXPECT_FALSE(engine.Run(kGrey, 0.0, result));
    // Drop the output
    ::testing::internal::GetCapturedStderr();
}

// Missing model files throw from the constructor
TEST(RsdEngineBadModelTest, MissingModelsThrow) {
    // Preprocessing model missing
    EXPECT_THROW(RsdInferenceEngine("/no/pre.onnx", FakeModel("rsd.onnx"), false), Ort::Exception);
    // Main model missing
    EXPECT_THROW(RsdInferenceEngine(FakeModel("rsd_pre.onnx"), "/no/rsd.onnx", false), Ort::Exception);
}

// Asking for CUDA runs on the GPU or falls back to CPU, with the same numbers
TEST(RsdEngineCudaTest, CudaRequestGivesSameResultAsCpu) {
    // CUDA requested
    RsdInferenceEngine gpu(FakeModel("rsd_pre.onnx"), FakeModel("rsd.onnx"), true);
    // Two steps on the GPU engine
    RsdResult a;
    // Step 1
    ASSERT_TRUE(gpu.Run(kGrey, 0.0, a));
    // Step 2
    ASSERT_TRUE(gpu.Run(kGrey, 1.0, a));
    // Second step value
    EXPECT_NEAR(a.rsd_s, 120.0f, 1e-3);
    // Same phase as on CPU
    EXPECT_EQ(a.phase, 2);
    // Report the device
    std::cout << "[ info ] CUDA requested, using " << (gpu.UsingCuda() ? "GPU" : "CPU (fallback)") << std::endl;
}

}  // namespace
