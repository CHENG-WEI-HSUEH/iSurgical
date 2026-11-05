// Unit tests for iSurgicalConfig (src/isurgical/isurgical.cpp): defaults, FromArgs and PrintUsage.

// Code under test
#include "isurgical/isurgical.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::isalpha
#include <cctype>
// std::string
#include <string>
// std::vector
#include <vector>

// Anonymous namespace
namespace {

// Call FromArgs with a list of words (argv[0] = "isurgical" is added automatically)
bool Parse(std::vector<std::string> words, iSurgicalConfig& config) {
    // Program name first
    words.insert(words.begin(), "isurgical");
    // argv as mutable char pointers (FromArgs takes char* argv[])
    std::vector<char*> argv;
    // One pointer per word
    for (std::string& w : words) argv.push_back(w.data());
    // Parse
    return iSurgicalConfig::FromArgs(static_cast<int>(argv.size()), argv.data(), config);
}

// The documented defaults
TEST(ConfigDefaultsTest, MatchTheReadme) {
    // Fresh config
    const iSurgicalConfig c;
    // Model
    EXPECT_EQ(c.model_path, "data/models/lc_seg_model.onnx");
    // Config file
    EXPECT_EQ(c.config_path, "configs/config.yaml");
    // Procedure
    EXPECT_EQ(c.procedure, "laparoscopic cholecystectomy");
    // RSD models
    EXPECT_EQ(c.rsd_preprocess_path, "data/models/preprocess_step.onnx");
    // RSD model
    EXPECT_EQ(c.rsd_model_path, "data/models/multimodal_rsd_step.onnx");
    // RSD on
    EXPECT_TRUE(c.rsd);
    // Video input
    EXPECT_EQ(c.input_method, "video");
    // Default clip
    EXPECT_EQ(c.input, "data/inputs/video01.mp4");
    // No outputs
    EXPECT_TRUE(c.output.empty());
    // No raw output
    EXPECT_TRUE(c.raw_output.empty());
    // All heads fused
    EXPECT_EQ(c.head, -1);
    // GPU
    EXPECT_TRUE(c.use_cuda);
    // No window
    EXPECT_FALSE(c.show);
    // Full screen when shown
    EXPECT_FALSE(c.windowed);
    // Stretch
    EXPECT_EQ(c.preproc_config.resize_mode, SegResizeMode::kStretch);
    // No border crop
    EXPECT_FALSE(c.preproc_config.crop_black_border);
    // No shared memory
    EXPECT_TRUE(c.shm_name.empty());
}

// No arguments: success, defaults kept
TEST(FromArgsTest, NoArgumentsKeepsDefaults) {
    // Config
    iSurgicalConfig c;
    // Parse nothing
    EXPECT_TRUE(Parse({}, c));
    // Unchanged
    EXPECT_EQ(c.input, "data/inputs/video01.mp4");
}

// Every option that takes a value
TEST(FromArgsTest, ValueOptionsAreStored) {
    // Config
    iSurgicalConfig c;
    // All of them at once
    ASSERT_TRUE(Parse({"--model", "m.onnx", "--config", "c.yaml", "--procedure", "appendectomy", "--input-method",
                       "camera", "--input", "0", "--output", "o.mp4", "--raw-output", "r.mp4", "--head", "2",
                       "--shm", "/isurgical"},
                      c));
    // --model
    EXPECT_EQ(c.model_path, "m.onnx");
    // --config
    EXPECT_EQ(c.config_path, "c.yaml");
    // --procedure
    EXPECT_EQ(c.procedure, "appendectomy");
    // --input-method
    EXPECT_EQ(c.input_method, "camera");
    // --input
    EXPECT_EQ(c.input, "0");
    // --output
    EXPECT_EQ(c.output, "o.mp4");
    // --raw-output
    EXPECT_EQ(c.raw_output, "r.mp4");
    // --head
    EXPECT_EQ(c.head, 2);
    // --shm
    EXPECT_EQ(c.shm_name, "/isurgical");
}

// Every flag
TEST(FromArgsTest, FlagsAreStored) {
    // Config
    iSurgicalConfig c;
    // All flags
    ASSERT_TRUE(Parse({"--cpu", "--letterbox", "--crop-border", "--show", "--windowed", "--no-rsd"}, c));
    // --cpu
    EXPECT_FALSE(c.use_cuda);
    // --letterbox
    EXPECT_EQ(c.preproc_config.resize_mode, SegResizeMode::kLetterbox);
    // --crop-border
    EXPECT_TRUE(c.preproc_config.crop_black_border);
    // --show
    EXPECT_TRUE(c.show);
    // --windowed
    EXPECT_TRUE(c.windowed);
    // --no-rsd
    EXPECT_FALSE(c.rsd);
}

// The last occurrence of an option wins
TEST(FromArgsTest, LastValueWins) {
    // Config
    iSurgicalConfig c;
    // Twice
    ASSERT_TRUE(Parse({"--input", "a.mp4", "--input", "b.mp4"}, c));
    // Second
    EXPECT_EQ(c.input, "b.mp4");
}

// A value may itself start with "--" (it is taken verbatim)
TEST(FromArgsTest, ValueMayLookLikeAnOption) {
    // Config
    iSurgicalConfig c;
    // "--cpu" as the output file name
    ASSERT_TRUE(Parse({"--output", "--cpu"}, c));
    // Stored as the output
    EXPECT_EQ(c.output, "--cpu");
    // Not treated as the flag
    EXPECT_TRUE(c.use_cuda);
}

// --head with a non-number is read as 0 (std::atoi), documented current behaviour
TEST(FromArgsTest, NonNumericHeadBecomesZero) {
    // Config
    iSurgicalConfig c;
    // "abc"
    ASSERT_TRUE(Parse({"--head", "abc"}, c));
    // atoi → 0
    EXPECT_EQ(c.head, 0);
}

// An option missing its value fails
class MissingValueTest : public ::testing::TestWithParam<std::string> {};

// Each value option without a value
TEST_P(MissingValueTest, Fails) {
    // Config
    iSurgicalConfig c;
    // Capture the message
    ::testing::internal::CaptureStderr();
    // Option is the last word
    EXPECT_FALSE(Parse({GetParam()}, c));
    // Message names the option
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("Missing value for " + GetParam()), std::string::npos);
}

// Every value option
INSTANTIATE_TEST_SUITE_P(Options, MissingValueTest,
                         ::testing::Values("--model", "--config", "--procedure", "--input-method", "--input",
                                           "--output", "--raw-output", "--head", "--shm"),
                         // "--raw-output" → "rawoutput"
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             // Name without dashes
                             std::string name;
                             // Keep letters only
                             for (char ch : info.param) if (std::isalpha(static_cast<unsigned char>(ch))) name += ch;
                             // Use it
                             return name;
                         });

// Unknown options (and positional words) fail; --help is "unknown" too, so main() prints the usage
class UnknownOptionTest : public ::testing::TestWithParam<std::string> {};

// Each one
TEST_P(UnknownOptionTest, Fails) {
    // Config
    iSurgicalConfig c;
    // Rejected
    EXPECT_FALSE(Parse({GetParam()}, c));
}

// The words
INSTANTIATE_TEST_SUITE_P(Words, UnknownOptionTest,
                         ::testing::Values("--help", "-h", "--gpu", "video.mp4", "--CPU", ""));

// PrintUsage lists every option and the program name
TEST(PrintUsageTest, ListsEveryOption) {
    // Capture stdout
    ::testing::internal::CaptureStdout();
    // Print
    iSurgicalConfig::PrintUsage("./build/isurgical");
    // Text
    const std::string out = ::testing::internal::GetCapturedStdout();
    // Program name
    EXPECT_NE(out.find("Usage: ./build/isurgical [options]"), std::string::npos);
    // Every option
    for (const char* opt : {"--model", "--config", "--procedure", "--input-method", "--input ", "--output",
                            "--raw-output", "--head", "--cpu", "--letterbox", "--crop-border", "--show", "--windowed",
                            "--no-rsd", "--shm"}) {
        // Present
        EXPECT_NE(out.find(opt), std::string::npos) << opt;
    }
}

// iSurgical keeps the config it was given
TEST(ISurgicalTest, ConfigAccessorReturnsTheConfig) {
    // Custom config
    iSurgicalConfig c;
    // A field
    c.input = "camera.mp4";
    // App (constructing it loads nothing)
    iSurgical app(c);
    // Same value
    EXPECT_EQ(app.Config().input, "camera.mp4");
}

}  // namespace
