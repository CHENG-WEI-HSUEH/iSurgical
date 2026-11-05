// Unit tests for src/postproc/postproc_rsd.cpp: RsdConfig::FromYaml, FormatDuration, RsdPostProcessor.

// Code under test
#include "postproc/postproc_rsd.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::runtime_error
#include <stdexcept>
// std::string
#include <string>

// TempDir, WriteFile, ConfigPath
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// The 7 phases as a YAML list (for custom configs)
const char* kPhases =
    "Phase: [Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, "
    "GallbladderPackaging, CleaningCoagulation, GallbladderRetraction]\n";

// A result with the given phase / time / rsd / progress
RsdResult Result(int phase, double time_s, float rsd_s = 600.0f, float progress = 0.5f) {
    // Start empty
    RsdResult r;
    // Phase id
    r.phase = phase;
    // Confidence
    r.phase_confidence = 0.9f;
    // Source time
    r.video_time_s = time_s;
    // Remaining seconds
    r.rsd_s = rsd_s;
    // Progress
    r.progress = progress;
    // Step
    r.step_index = static_cast<int64_t>(time_s) + 1;
    // Done
    return r;
}

// ---------------------------------------------------------------------------
// RsdConfig::FromYaml
// ---------------------------------------------------------------------------

// Fixture: scratch folder for YAML files
class RsdConfigTest : public ::testing::Test {
protected:
    // Write a YAML file and return its path
    std::string Yaml(const std::string& text) {
        // File path
        const std::string path = tmp_ / "config.yaml";
        // Write
        WriteFile(path, text);
        // Path
        return path;
    }
    // Scratch folder
    TempDir tmp_;
};

// The defaults when nothing is configured
TEST_F(RsdConfigTest, DefaultsWithoutKeys) {
    // Empty map
    const RsdConfig c = RsdConfig::FromYaml(Yaml("{}\n"));
    // 15 s
    EXPECT_DOUBLE_EQ(c.display_period_s, 15.0);
    // 1 fps
    EXPECT_DOUBLE_EQ(c.infer_fps, 1.0);
    // Cholec80 phases
    ASSERT_EQ(c.phase_names.size(), 7u);
    // First
    EXPECT_EQ(c.phase_names[0], "Preparation");
    // Last
    EXPECT_EQ(c.phase_names[6], "GallbladderRetraction");
}

// The real configs/config.yaml section
TEST_F(RsdConfigTest, RealProjectConfig) {
    // Procedure section
    const RsdConfig c = RsdConfig::FromYaml(ConfigPath(), kProcedure);
    // RSD_Period: 15
    EXPECT_DOUBLE_EQ(c.display_period_s, 15.0);
    // 7 phases
    EXPECT_EQ(c.phase_names.size(), 7u);
    // Third phase
    EXPECT_EQ(c.phase_names[2], "ClippingCutting");
}

// All three keys are read
TEST_F(RsdConfigTest, ReadsAllKeys) {
    // Custom values
    const RsdConfig c = RsdConfig::FromYaml(Yaml(std::string("RSD_Period: 5\nRSD_InferFPS: 2.5\n") + kPhases));
    // Period
    EXPECT_DOUBLE_EQ(c.display_period_s, 5.0);
    // Rate
    EXPECT_DOUBLE_EQ(c.infer_fps, 2.5);
}

// A procedure section is selected by name
TEST_F(RsdConfigTest, ProcedureSectionIsSelected) {
    // Two sections
    const std::string path = Yaml("a:\n  RSD_Period: 3\nb:\n  RSD_Period: 9\n");
    // Section b
    EXPECT_DOUBLE_EQ(RsdConfig::FromYaml(path, "b").display_period_s, 9.0);
}

// Every invalid file / value throws std::runtime_error with the file path in the message
struct BadRsdCase {
    // Test name
    const char* name;
    // File content
    std::string yaml;
    // Procedure to select
    std::string procedure;
    // Expected text in the message
    std::string message;
};

// Parameterised fixture
class BadRsdConfigTest : public RsdConfigTest, public ::testing::WithParamInterface<BadRsdCase> {};

// Each bad case
TEST_P(BadRsdConfigTest, Throws) {
    // The file
    const std::string path = Yaml(GetParam().yaml);
    // Must throw runtime_error with the expected text
    try {
        // Load
        RsdConfig::FromYaml(path, GetParam().procedure);
        // Did not throw
        FAIL() << "no exception";
    // Expected
    } catch (const std::runtime_error& e) {
        // Message names the problem
        EXPECT_NE(std::string(e.what()).find(GetParam().message), std::string::npos) << e.what();
        // And the file
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
    }
}

// The cases
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Cases,
    // Fixture
    BadRsdConfigTest,
    // Rows
    ::testing::Values(
        // Procedure not in the file
        BadRsdCase{"MissingProcedure", "a: {}\n", "zzz", "missing procedure 'zzz'"},
        // Zero period
        BadRsdCase{"ZeroPeriod", "RSD_Period: 0\n", "", "RSD_Period must be > 0"},
        // Negative period
        BadRsdCase{"NegativePeriod", "RSD_Period: -1\n", "", "RSD_Period must be > 0"},
        // Zero rate
        BadRsdCase{"ZeroFps", "RSD_InferFPS: 0\n", "", "RSD_InferFPS must be > 0"},
        // Wrong number of phases
        BadRsdCase{"SixPhases", "Phase: [a, b, c, d, e, f]\n", "", "Phase must have 7 entries"},
        // Not a number
        BadRsdCase{"PeriodNotNumber", "RSD_Period: soon\n", "", "bad conversion"},
        // Syntax error
        BadRsdCase{"SyntaxError", "RSD_Period: [1\n", "", ""}),
    // Row name
    [](const ::testing::TestParamInfo<BadRsdCase>& info) { return std::string(info.param.name); });

// A missing file throws
TEST_F(RsdConfigTest, MissingFileThrows) {
    // Not on disk
    EXPECT_THROW(RsdConfig::FromYaml("/no/such/config.yaml"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// FormatDuration
// ---------------------------------------------------------------------------

// One FormatDuration case
struct DurationCase {
    // Name
    const char* name;
    // Seconds
    double seconds;
    // Text
    const char* expected;
};

// Parameterised fixture
class FormatDurationTest : public ::testing::TestWithParam<DurationCase> {};

// Each case
TEST_P(FormatDurationTest, Formats) {
    // Exact text
    EXPECT_EQ(FormatDuration(GetParam().seconds), GetParam().expected);
}

// The cases
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Cases,
    // Fixture
    FormatDurationTest,
    // Rows
    ::testing::Values(
        // Zero
        DurationCase{"Zero", 0, "00:00"},
        // Rounded, not truncated
        DurationCase{"RoundsHalfUp", 59.5, "01:00"},
        // Just under
        DurationCase{"RoundsDown", 59.4, "00:59"},
        // The README example
        DurationCase{"ReadmeExample", 1513, "25:13"},
        // Just under an hour: still mm:ss
        DurationCase{"JustUnderAnHour", 3599, "59:59"},
        // One hour: h:mm:ss
        DurationCase{"OneHour", 3600, "1:00:00"},
        // Several hours
        DurationCase{"Hours", 3 * 3600 + 5 * 60 + 7, "3:05:07"},
        // Negative values are clamped to zero
        DurationCase{"NegativeClamped", -42, "00:00"}),
    // Row name
    [](const ::testing::TestParamInfo<DurationCase>& info) { return std::string(info.param.name); });

// ---------------------------------------------------------------------------
// RsdPostProcessor
// ---------------------------------------------------------------------------

// Fixture: a processor with a 15 s display period
class RsdPostTest : public ::testing::Test {
protected:
    // Default config (15 s, 1 fps, Cholec80)
    RsdPostProcessor post_{RsdConfig()};
};

// Nothing displayed initially
TEST_F(RsdPostTest, NothingDisplayedInitially) {
    // phase -1 = no result
    EXPECT_EQ(post_.Displayed().phase, -1);
}

// The first result is displayed, then updates only every 15 s of source time
TEST_F(RsdPostTest, UpdatesOnlyEveryDisplayPeriod) {
    // t = 0: displayed
    EXPECT_TRUE(post_.Update(Result(0, 0.0)));
    // t = 1 .. 14: not displayed
    for (int t = 1; t < 15; ++t) EXPECT_FALSE(post_.Update(Result(1, t))) << t;
    // Still the first one
    EXPECT_EQ(post_.Displayed().phase, 0);
    // t = 15: displayed
    EXPECT_TRUE(post_.Update(Result(2, 15.0)));
    // The new one
    EXPECT_EQ(post_.Displayed().phase, 2);
}

// A gap longer than one period skips the missed display slots
TEST_F(RsdPostTest, LongGapSkipsMissedSlots) {
    // t = 0
    EXPECT_TRUE(post_.Update(Result(0, 0.0)));
    // Jump to t = 50 (slots 15, 30, 45 missed)
    EXPECT_TRUE(post_.Update(Result(3, 50.0)));
    // Next slot is 60: t = 55 is not shown
    EXPECT_FALSE(post_.Update(Result(4, 55.0)));
    // t = 60 is
    EXPECT_TRUE(post_.Update(Result(4, 60.0)));
}

// Reset starts over: the next result is displayed immediately
TEST_F(RsdPostTest, ResetStartsOver) {
    // t = 100
    post_.Update(Result(5, 100.0));
    // New video
    post_.Reset();
    // Nothing displayed
    EXPECT_EQ(post_.Displayed().phase, -1);
    // t = 0 displayed again
    EXPECT_TRUE(post_.Update(Result(1, 0.0)));
}

// Phase names by index; out of range gives "-"
TEST_F(RsdPostTest, PhaseNameLookup) {
    // Index 0
    EXPECT_EQ(post_.PhaseName(0), "Preparation");
    // Index 6
    EXPECT_EQ(post_.PhaseName(6), "GallbladderRetraction");
    // Before the first result
    EXPECT_EQ(post_.PhaseName(-1), "-");
    // Too large
    EXPECT_EQ(post_.PhaseName(7), "-");
}

// The console line matches the README format
TEST_F(RsdPostTest, DescribeMatchesReadmeFormat) {
    // The README example: step 31 @ 30 s, phase 1 (0.98), RSD 25:13, progress 31.2 %
    RsdResult r = Result(1, 30.0, 1513.0f, 0.312f);
    // Step
    r.step_index = 31;
    // Confidence
    r.phase_confidence = 0.98f;
    // Exact text
    EXPECT_EQ(post_.Describe(r),
              "[rsd step 31 @ 00:30] phase: CalotTriangleDissection (0.98)  RSD: 25:13  progress: 31.2%");
}

// Config() returns the configuration
TEST_F(RsdPostTest, ConfigAccessor) {
    // 15 s
    EXPECT_DOUBLE_EQ(post_.Config().display_period_s, 15.0);
}

// Draw darkens a bottom-left panel and leaves the rest of the frame alone
TEST_F(RsdPostTest, DrawPanelBottomLeft) {
    // Mid-grey 640x480
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(200));
    // Draw a result
    post_.Draw(frame, Result(2, 30.0, 900.0f, 0.5f));
    // Bottom-left corner inside the panel is darkened (0.4 × 200 = 80, text aside)
    EXPECT_LT(frame.at<cv::Vec3b>(470, 10)[0], 200);
    // Top-right is untouched
    EXPECT_EQ(frame.at<cv::Vec3b>(10, 630), cv::Vec3b(200, 200, 200));
}

// Before the first result a "waiting" panel is drawn
TEST_F(RsdPostTest, DrawWaitingPanel) {
    // Frame
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(200));
    // A copy for comparison
    const cv::Mat before = frame.clone();
    // No result yet
    post_.Draw(frame, RsdResult());
    // Something changed (panel drawn)
    EXPECT_GT(cv::norm(frame, before, cv::NORM_L1), 0.0);
}

// The progress bar fill grows with progress
TEST_F(RsdPostTest, ProgressBarFillGrowsWithProgress) {
    // Count the bar's fill colour (BGR 80, 200, 255) in a frame drawn with this progress
    auto filled = [&](float progress) {
        // Black frame
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
        // Draw
        post_.Draw(frame, Result(2, 0.0, 100.0f, progress));
        // Fill-coloured pixels
        cv::Mat mask;
        // Exact colour
        cv::inRange(frame, cv::Scalar(80, 200, 255), cv::Scalar(80, 200, 255), mask);
        // Count
        return cv::countNonZero(mask);
    };
    // 0 % → no fill
    EXPECT_EQ(filled(0.0f), 0);
    // 50 % < 100 %
    EXPECT_LT(filled(0.5f), filled(1.0f));
    // Values above 1 are clamped
    EXPECT_EQ(filled(5.0f), filled(1.0f));
}

// A frame too small for the panel does not crash
TEST_F(RsdPostTest, DrawOnTinyFrame) {
    // 4x4
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar::all(0));
    // Must not throw
    EXPECT_NO_THROW(post_.Draw(frame, Result(1, 0.0)));
}

}  // namespace
