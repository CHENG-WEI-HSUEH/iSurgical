// Tests with the REAL models in data/models (and the real video in data/inputs, when present).
// data/ is not in the repository, so every test here skips itself when its files are missing.
// CPU runs take about 1-2 s per model call; the GPU test also skips when there is no NVIDIA device.

// SegInferenceEngine
#include "infer/infer_seg.hpp"
// RsdInferenceEngine
#include "infer/infer_rsd.hpp"
// SegPostProcessor (to check the real output gives regions)
#include "postproc/postproc_seg.hpp"
// iSurgical (full run)
#include "isurgical/isurgical.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::isfinite
#include <cmath>
// std::accumulate
#include <numeric>

// RealModel, ProjectFile, TempDir, ConfigPath
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// A real frame: frame 750 (30 s) of data/inputs/video01.mp4, or "" if the video is missing
cv::Mat RealFrame() {
    // The clip
    const std::string path = ProjectFile("data/inputs/video01.mp4");
    // Missing
    if (!fs::exists(path)) return cv::Mat();
    // Open
    cv::VideoCapture cap(path);
    // Jump to 30 s (frame 0 is often a dark intro)
    cap.set(cv::CAP_PROP_POS_FRAMES, 750);
    // Frame
    cv::Mat frame;
    // Read it
    cap.read(frame);
    // Return (empty on failure)
    return frame;
}

// Skip the test when the real video is missing
#define REQUIRE_REAL_FRAME(frame)                                                      \
    /* Load */                                                                         \
    const cv::Mat frame = RealFrame();                                                 \
    /* Missing */                                                                      \
    if (frame.empty()) GTEST_SKIP() << "data/inputs/video01.mp4 not found (data/ is not in git)"

// The real segmentation model on CPU finds plausible anatomy on a real frame
TEST(RealModelsTest, SegmentationOnCpu) {
    // Model
    REQUIRE_REAL_MODEL(model, "lc_seg_model.onnx");
    // Frame
    REQUIRE_REAL_FRAME(frame);
    // CPU engine (passes CheckModelIO: the real model has the expected inputs / outputs)
    SegInferenceEngine engine(model, false);
    // Output
    SegResult seg;
    // Run
    ASSERT_TRUE(engine.Run(frame, seg));
    // Masks at frame size
    EXPECT_EQ(seg.masks[0].size(), frame.size());
    // Finite scores
    for (float s : seg.correlation_scores) EXPECT_TRUE(std::isfinite(s));
    // Post-process with the real config
    SegPostProcessor post(SegPostConfig::FromYaml(ConfigPath(), kProcedure));
    // Regions
    SegPostResult out;
    // Run
    post.Process(seg, out);
    // Some anatomy or tool is found on a mid-surgery frame
    EXPECT_GT(cv::countNonZero(out.label_map), 0);
}

// On the GPU the result matches the CPU (Pascal-capable cuDNN required, see the README)
TEST(RealModelsTest, SegmentationGpuMatchesCpu) {
    // Model
    REQUIRE_REAL_MODEL(model, "lc_seg_model.onnx");
    // Frame
    REQUIRE_REAL_FRAME(frame);
    // No GPU
    if (!fs::exists("/dev/nvidia0")) GTEST_SKIP() << "no NVIDIA GPU";
    // GPU engine
    SegInferenceEngine gpu(model, true);
    // CPU engine
    SegInferenceEngine cpu(model, false);
    // Outputs
    SegResult a, b;
    // GPU run
    ASSERT_TRUE(gpu.Run(frame, a));
    // CPU run
    ASSERT_TRUE(cpu.Run(frame, b));
    // Still on the GPU after the first run (no fallback to CPU)
    EXPECT_TRUE(gpu.UsingCuda()) << "fell back to CPU: check CUDNN_LIB_DIR (cuDNN 9.5.1 for Pascal)";
    // Each head: at least 99 % of the pixels have the same class
    for (int h = 0; h < SegResult::kNumHeads; ++h) {
        // Differing pixels
        const int diff = cv::countNonZero(a.masks[h] != b.masks[h]);
        // Small
        EXPECT_LT(diff, static_cast<int>(frame.total() / 100)) << "head " << h;
    }
}

// The real phase / RSD models produce valid probabilities, a non-negative RSD and progress in [0, 1]
TEST(RealModelsTest, PhaseAndRsdOnCpu) {
    // Preprocessing model
    REQUIRE_REAL_MODEL(pre, "preprocess_step.onnx");
    // Main model
    REQUIRE_REAL_MODEL(rsd, "multimodal_rsd_step.onnx");
    // Frame
    REQUIRE_REAL_FRAME(frame);
    // CPU engine
    RsdInferenceEngine engine(pre, rsd, false);
    // Output
    RsdResult result;
    // Three 1-fps steps
    for (int step = 0; step < 3; ++step) ASSERT_TRUE(engine.Run(frame, step, result)) << step;
    // Valid phase
    EXPECT_GE(result.phase, 0);
    // Valid phase
    EXPECT_LT(result.phase, RsdResult::kNumPhases);
    // Probabilities sum to 1
    EXPECT_NEAR(std::accumulate(result.phase_probs.begin(), result.phase_probs.end(), 0.0f), 1.0f, 1e-3);
    // Non-negative RSD
    EXPECT_GE(result.rsd_s, 0.0f);
    // Progress in [0, 1]
    EXPECT_GE(result.progress, 0.0f);
    // Upper bound
    EXPECT_LE(result.progress, 1.0f);
    // Three steps
    EXPECT_EQ(result.step_index, 3);
}

// The whole application on one real frame (image mode, CPU, every real model)
TEST(RealModelsTest, FullRunOnRealFrame) {
    // Every model
    REQUIRE_REAL_MODEL(seg, "lc_seg_model.onnx");
    // Preprocessing
    REQUIRE_REAL_MODEL(pre, "preprocess_step.onnx");
    // RSD
    REQUIRE_REAL_MODEL(rsd, "multimodal_rsd_step.onnx");
    // Frame
    REQUIRE_REAL_FRAME(frame);
    // Scratch folder
    TempDir tmp;
    // Save the frame as the input image
    const std::string image = tmp / "frame.png";
    // Write
    ASSERT_TRUE(cv::imwrite(image, frame));
    // Config
    iSurgicalConfig config;
    // Real models
    config.model_path = seg;
    // Preprocessing
    config.rsd_preprocess_path = pre;
    // RSD
    config.rsd_model_path = rsd;
    // Real config
    config.config_path = ConfigPath();
    // Image mode
    config.input_method = "image";
    // The frame
    config.input = image;
    // Output
    config.output = tmp / "vis.png";
    // CPU
    config.use_cuda = false;
    // Quiet
    ::testing::internal::CaptureStdout();
    // Run
    const int code = iSurgical(config).Run();
    // Output text
    const std::string out = ::testing::internal::GetCapturedStdout();
    // Success
    ASSERT_EQ(code, 0);
    // Annotated image written
    EXPECT_FALSE(cv::imread(config.output).empty());
    // RSD line printed with a real phase name
    EXPECT_NE(out.find("[rsd step 1 @ 00:00] phase: "), std::string::npos) << out;
}

}  // namespace
