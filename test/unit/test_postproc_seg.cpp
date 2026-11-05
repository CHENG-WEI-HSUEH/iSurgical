// Unit tests for src/postproc/postproc_seg.cpp: SegPostConfig (YAML + validation) and SegPostProcessor
// (thresholds, morphology, connected regions, head fusion, drawing).

// Code under test
#include "postproc/postproc_seg.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::map
#include <map>
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

// Frame size used by the Process tests (min_area = 0.001 × 512 × 512 = 262 pixels)
const cv::Size kSize(512, 512);

// A valid top-level config, one key per line so single lines can be replaced
const std::map<std::string, std::string> kBaseYaml = {
    // Same groups as configs/config.yaml
    {"head_groups", "head_groups: [[1, 2, 3, 13], [4, 15], [5, 12, 14], [6, 7, 8, 9, 10, 11]]"},
    // 16 names
    {"label_names", "label_names: [bg, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]"},
    // 16 opaque grey colours
    {"kclass_colours", "kclass_colours: [" + [] {
         // Build the list
         std::string s;
         // 16 entries
         for (int i = 0; i < 16; ++i) s += std::string(i ? ", " : "") + "[0.5, 0.5, 0.5, 1.0]";
         // Return it
         return s;
     }() + "]"},
    // Class count
    {"NumClass", "NumClass: 16"},
    // Pixel threshold
    {"MaskThreshold", "MaskThreshold: 0.5"},
    // Region threshold
    {"ConfidenceThreshold", "ConfidenceThreshold: 0.25"},
    // Parsed, unused
    {"IoUThreshold", "IoUThreshold: 0.5"},
};

// The base config with some lines replaced ("" removes the key)
std::string YamlWith(const std::map<std::string, std::string>& changes = {}) {
    // Output text
    std::string out;
    // Each key of the base
    for (const auto& [key, line] : kBaseYaml) {
        // Changed?
        const auto it = changes.find(key);
        // New line (or removed when empty)
        const std::string& text = it == changes.end() ? line : it->second;
        // Append
        if (!text.empty()) out += text + "\n";
    }
    // Extra keys not in the base
    for (const auto& [key, line] : changes) {
        // Only new keys
        if (!kBaseYaml.count(key) && !line.empty()) out += line + "\n";
    }
    // Done
    return out;
}

// An empty SegResult of kSize: every head Background with confidence 0
SegResult EmptySeg() {
    // Result
    SegResult r;
    // Each head
    for (int h = 0; h < SegResult::kNumHeads; ++h) {
        // Class 0 everywhere
        r.masks[h] = cv::Mat::zeros(kSize, CV_8UC1);
        // Confidence 0
        r.confidences[h] = cv::Mat::zeros(kSize, CV_32FC1);
    }
    // Scores 0.1 .. 0.7
    for (int i = 0; i < SegResult::kNumScores; ++i) r.correlation_scores[i] = 0.1f * (i + 1);
    // Done
    return r;
}

// Paint a rectangle of head-local class `local` with confidence `conf` into one head
void Paint(SegResult& r, int head, cv::Rect rect, int local, float conf) {
    // Class id
    r.masks[head](rect).setTo(local);
    // Confidence
    r.confidences[head](rect).setTo(conf);
}

// ---------------------------------------------------------------------------
// SegPostConfig
// ---------------------------------------------------------------------------

// Fixture: scratch folder for YAML files
class SegConfigTest : public ::testing::Test {
protected:
    // Write YAML and return the path
    std::string Yaml(const std::string& text) {
        // Path
        const std::string path = tmp_ / "config.yaml";
        // Write
        WriteFile(path, text);
        // Path
        return path;
    }
    // Scratch folder
    TempDir tmp_;
};

// The real configs/config.yaml loads and converts RGBA [0,1] to BGR [0,255] + alpha
TEST_F(SegConfigTest, RealProjectConfig) {
    // Procedure section
    const SegPostConfig c = SegPostConfig::FromYaml(ConfigPath(), kProcedure);
    // 16 classes
    EXPECT_EQ(c.num_classes, 16);
    // Names
    EXPECT_EQ(c.label_names[1], "Gallbladder");
    // Group sizes match the model heads (5, 3, 4, 7 channels minus Background)
    EXPECT_EQ(c.head_groups[0].size(), 4u);
    // Head 3 = tools
    EXPECT_EQ(c.head_groups[3].size(), 6u);
    // Gallbladder RGBA (0, 1, 0, 0.2) → BGR (0, 255, 0)
    EXPECT_EQ(c.class_colours[1], cv::Scalar(0, 255, 0));
    // Alpha 0.2
    EXPECT_FLOAT_EQ(c.class_alphas[1], 0.2f);
    // Thresholds
    EXPECT_FLOAT_EQ(c.mask_threshold, 0.5f);
    // Region threshold
    EXPECT_FLOAT_EQ(c.confidence_threshold, 0.25f);
    // Defaults not in the file
    EXPECT_EQ(c.morph_kernel_size, 5);
    // Default
    EXPECT_FLOAT_EQ(c.min_area_ratio, 0.001f);
}

// Optional keys override the defaults
TEST_F(SegConfigTest, OptionalKeysOverride) {
    // MorphKernelSize / MinAreaRatio / IoUThreshold
    const SegPostConfig c = SegPostConfig::FromYaml(
        Yaml(YamlWith({{"x1", "MorphKernelSize: 0"}, {"x2", "MinAreaRatio: 0.05"}, {"IoUThreshold", "IoUThreshold: 0.7"}})));
    // Morphology off
    EXPECT_EQ(c.morph_kernel_size, 0);
    // Area
    EXPECT_FLOAT_EQ(c.min_area_ratio, 0.05f);
    // IoU
    EXPECT_FLOAT_EQ(c.iou_threshold, 0.7f);
}

// IoUThreshold is optional
TEST_F(SegConfigTest, IouThresholdIsOptional) {
    // Removed
    const SegPostConfig c = SegPostConfig::FromYaml(Yaml(YamlWith({{"IoUThreshold", ""}})));
    // Default
    EXPECT_FLOAT_EQ(c.iou_threshold, 0.5f);
}

// Alpha is clamped to [0, 1]; an RGB-only entry gets alpha 1
TEST_F(SegConfigTest, AlphaIsClampedAndDefaultsToOne) {
    // Class 1 alpha 2.0, class 2 alpha -1, class 3 RGB only
    std::string colours = "kclass_colours: [[0, 0, 0, 0], [1, 0, 0, 2.0], [0, 1, 0, -1], [0, 0, 1]";
    // The other 12 entries
    for (int i = 4; i < 16; ++i) colours += ", [0.5, 0.5, 0.5, 1.0]";
    // Close the list
    colours += "]";
    // Load
    const SegPostConfig c = SegPostConfig::FromYaml(Yaml(YamlWith({{"kclass_colours", colours}})));
    // Clamped high
    EXPECT_FLOAT_EQ(c.class_alphas[1], 1.0f);
    // Clamped low
    EXPECT_FLOAT_EQ(c.class_alphas[2], 0.0f);
    // RGB only
    EXPECT_FLOAT_EQ(c.class_alphas[3], 1.0f);
    // RGB (1, 0, 0) → BGR (0, 0, 255)
    EXPECT_EQ(c.class_colours[1], cv::Scalar(0, 0, 255));
}

// One invalid configuration
struct BadSegCase {
    // Test name
    const char* name;
    // Line changes against the base config
    std::map<std::string, std::string> changes;
    // Expected text in the exception message
    std::string message;
};

// Parameterised fixture
class BadSegConfigTest : public SegConfigTest, public ::testing::WithParamInterface<BadSegCase> {};

// Each case throws runtime_error with a helpful message
TEST_P(BadSegConfigTest, Throws) {
    // File
    const std::string path = Yaml(YamlWith(GetParam().changes));
    // Load
    try {
        // Must throw
        SegPostConfig::FromYaml(path);
        // Did not
        FAIL() << "no exception";
    // Expected
    } catch (const std::runtime_error& e) {
        // Message
        EXPECT_NE(std::string(e.what()).find(GetParam().message), std::string::npos) << e.what();
    }
}

// Helper: 16 colours where entry 5 has only two values
std::string ShortColour() {
    // Start
    std::string s = "kclass_colours: [";
    // 16 entries
    for (int i = 0; i < 16; ++i) s += std::string(i ? ", " : "") + (i == 5 ? "[0.5, 0.5]" : "[0.5, 0.5, 0.5]");
    // Close
    return s + "]";
}

// The cases
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Cases,
    // Fixture
    BadSegConfigTest,
    // Rows
    ::testing::Values(
        // Each required key missing
        BadSegCase{"NoHeadGroups", {{"head_groups", ""}}, "missing key 'head_groups'"},
        // Names
        BadSegCase{"NoLabelNames", {{"label_names", ""}}, "missing key 'label_names'"},
        // Colours
        BadSegCase{"NoColours", {{"kclass_colours", ""}}, "missing key 'kclass_colours'"},
        // Count
        BadSegCase{"NoNumClass", {{"NumClass", ""}}, "missing key 'NumClass'"},
        // Pixel threshold
        BadSegCase{"NoMaskThreshold", {{"MaskThreshold", ""}}, "missing key 'MaskThreshold'"},
        // Region threshold
        BadSegCase{"NoConfidenceThreshold", {{"ConfidenceThreshold", ""}}, "missing key 'ConfidenceThreshold'"},
        // Colour entry with two values
        BadSegCase{"ColourTooShort", {{"kclass_colours", ShortColour()}}, "need at least 3 values"},
        // NumClass 0
        BadSegCase{"NumClassZero", {{"NumClass", "NumClass: 0"}}, "NumClass must be in [1, 256]"},
        // NumClass too big
        BadSegCase{"NumClassTooBig", {{"NumClass", "NumClass: 300"}}, "NumClass must be in [1, 256]"},
        // Names / colours do not match NumClass
        BadSegCase{"NumClassMismatch", {{"NumClass", "NumClass: 17"}}, "must both have NumClass entries"},
        // Three groups instead of four
        BadSegCase{"ThreeGroups", {{"head_groups", "head_groups: [[1, 2, 3, 13], [4, 15], [5, 12, 14]]"}},
                   "head_groups must have 4 groups"},
        // Head 1 with three classes (model has 2 + Background)
        BadSegCase{"GroupSizeMismatch",
                   {{"head_groups", "head_groups: [[1, 2, 3, 13], [4, 15, 5], [12, 14, 6], [7, 8, 9, 10, 11, 1]]"}},
                   "head_groups[1] must have 2 classes"},
        // Class id 0 (Background) in a group
        BadSegCase{"BackgroundInGroup",
                   {{"head_groups", "head_groups: [[0, 2, 3, 13], [4, 15], [5, 12, 14], [6, 7, 8, 9, 10, 11]]"}},
                   "class id 0 out of range"},
        // Class id beyond NumClass
        BadSegCase{"ClassIdTooBig",
                   {{"head_groups", "head_groups: [[1, 2, 3, 16], [4, 15], [5, 12, 14], [6, 7, 8, 9, 10, 11]]"}},
                   "class id 16 out of range"},
        // Wrong type
        BadSegCase{"ThresholdNotNumber", {{"MaskThreshold", "MaskThreshold: high"}}, "bad conversion"}),
    // Row name
    [](const ::testing::TestParamInfo<BadSegCase>& info) { return std::string(info.param.name); });

// Missing file
TEST_F(SegConfigTest, MissingFileThrows) {
    // Not on disk
    EXPECT_THROW(SegPostConfig::FromYaml("/no/such/config.yaml"), std::runtime_error);
}

// Missing procedure section
TEST_F(SegConfigTest, MissingProcedureThrows) {
    // Real file, wrong section
    EXPECT_THROW(SegPostConfig::FromYaml(ConfigPath(), "appendectomy"), std::runtime_error);
}

// The constructor validates too (a hand-built config)
TEST(SegPostProcessorTest, ConstructorValidates) {
    // Empty config: NumClass 0
    EXPECT_THROW(SegPostProcessor{SegPostConfig()}, std::runtime_error);
}

// ---------------------------------------------------------------------------
// SegPostProcessor::Process
// ---------------------------------------------------------------------------

// Fixture: the real config's processor
class SegProcessTest : public ::testing::Test {
protected:
    // Real config
    SegPostProcessor post_{SegPostConfig::FromYaml(ConfigPath(), kProcedure)};
    // Input
    SegResult seg_ = EmptySeg();
    // Output
    SegPostResult out_;
};

// No foreground: no regions, all-Background masks, scores copied
TEST_F(SegProcessTest, EmptyInputHasNoRegions) {
    // Process
    post_.Process(seg_, out_);
    // No regions in any head
    for (const auto& head : out_.heads) EXPECT_TRUE(head.regions.empty());
    // Fused map is all Background
    EXPECT_EQ(cv::countNonZero(out_.label_map), 0);
    // Scores copied
    EXPECT_FLOAT_EQ(out_.correlation_scores[6], 0.7f);
}

// One confident square becomes one region with the right global class, area, box and centroid
TEST_F(SegProcessTest, SingleRegionIsReported) {
    // 100x100 square of head-0 local class 1 (= Gallbladder, global 1) at (100, 100)
    Paint(seg_, 0, cv::Rect(100, 100, 100, 100), 1, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // One region
    ASSERT_EQ(out_.heads[0].regions.size(), 1u);
    // The region
    const SegRegion& r = out_.heads[0].regions[0];
    // Global class
    EXPECT_EQ(r.class_id, 1);
    // Head
    EXPECT_EQ(r.head, 0);
    // Area: 100 x 100, minus the few corner pixels the 5x5 elliptical opening removes
    EXPECT_GE(r.area, 9980);
    // Never more than the painted square
    EXPECT_LE(r.area, 10000);
    // Fraction of the frame matches the pixel count
    EXPECT_NEAR(r.area_ratio, r.area / (512.0f * 512.0f), 1e-6);
    // Mean confidence
    EXPECT_NEAR(r.confidence, 0.9f, 1e-5);
    // Bounding box
    EXPECT_EQ(r.bbox, cv::Rect(100, 100, 100, 100));
    // Centroid at the centre
    EXPECT_NEAR(r.centroid.x, 149.5f, 0.01f);
    // Centroid
    EXPECT_NEAR(r.centroid.y, 149.5f, 0.01f);
    // A closed outline
    EXPECT_GE(r.contour.size(), 4u);
    // Head mask holds the global class id
    EXPECT_EQ(out_.heads[0].mask.at<uint8_t>(150, 150), 1);
    // Fused map too
    EXPECT_EQ(out_.label_map.at<uint8_t>(150, 150), 1);
}

// Local class ids map through head_groups: head 0 local 4 → Common bile duct (13), head 1 local 1 → Liver (4)
TEST_F(SegProcessTest, LocalIdsMapToGlobalIds) {
    // Head 0, local 4
    Paint(seg_, 0, cv::Rect(0, 0, 100, 100), 4, 0.9f);
    // Head 1, local 1
    Paint(seg_, 1, cv::Rect(300, 300, 100, 100), 1, 0.9f);
    // Head 3, local 6 (scissors = 11)
    Paint(seg_, 3, cv::Rect(300, 0, 100, 100), 6, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // 13
    EXPECT_EQ(out_.heads[0].regions.at(0).class_id, 13);
    // 4
    EXPECT_EQ(out_.heads[1].regions.at(0).class_id, 4);
    // 11
    EXPECT_EQ(out_.heads[3].regions.at(0).class_id, 11);
    // Label names
    EXPECT_EQ(post_.LabelName(11), "scissors");
}

// Pixels below MaskThreshold (0.5) are Background
TEST_F(SegProcessTest, BelowMaskThresholdIsDropped) {
    // Confidence 0.4
    Paint(seg_, 0, cv::Rect(100, 100, 100, 100), 1, 0.4f);
    // Process
    post_.Process(seg_, out_);
    // Nothing
    EXPECT_TRUE(out_.heads[0].regions.empty());
}

// Exactly at MaskThreshold is kept (>=)
TEST_F(SegProcessTest, AtMaskThresholdIsKept) {
    // Confidence exactly 0.5
    Paint(seg_, 0, cv::Rect(100, 100, 100, 100), 1, 0.5f);
    // Process
    post_.Process(seg_, out_);
    // Kept
    EXPECT_EQ(out_.heads[0].regions.size(), 1u);
}

// Components smaller than MinAreaRatio (262 px here) are removed
TEST_F(SegProcessTest, SmallComponentsAreRemoved) {
    // 15x15 = 225 px (survives the 5x5 opening, but below the area limit)
    Paint(seg_, 0, cv::Rect(10, 10, 15, 15), 1, 0.9f);
    // 20x20 = 400 px: kept
    Paint(seg_, 0, cv::Rect(100, 100, 20, 20), 1, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // Only the larger one
    ASSERT_EQ(out_.heads[0].regions.size(), 1u);
    // It is the 20x20 one
    EXPECT_EQ(out_.heads[0].regions[0].bbox.x, 100);
    // The small one is not in the mask either
    EXPECT_EQ(out_.heads[0].mask.at<uint8_t>(15, 15), 0);
}

// Regions are sorted by area, largest first
TEST_F(SegProcessTest, RegionsSortedByArea) {
    // Small
    Paint(seg_, 0, cv::Rect(0, 0, 40, 40), 1, 0.9f);
    // Large
    Paint(seg_, 0, cv::Rect(200, 200, 120, 120), 1, 0.9f);
    // Medium, other class
    Paint(seg_, 0, cv::Rect(0, 300, 80, 80), 2, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // Three
    ASSERT_EQ(out_.heads[0].regions.size(), 3u);
    // Descending
    EXPECT_GT(out_.heads[0].regions[0].area, out_.heads[0].regions[1].area);
    // Descending
    EXPECT_GT(out_.heads[0].regions[1].area, out_.heads[0].regions[2].area);
}

// Where two heads overlap, the fused map takes the more confident one
TEST_F(SegProcessTest, FusionPicksMostConfidentHead) {
    // Head 0: Gallbladder, 0.7
    Paint(seg_, 0, cv::Rect(100, 100, 200, 200), 1, 0.7f);
    // Head 1: Liver, 0.9, overlapping the right half
    Paint(seg_, 1, cv::Rect(200, 100, 200, 200), 1, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // Only head 0
    EXPECT_EQ(out_.label_map.at<uint8_t>(150, 150), 1);
    // Overlap → Liver
    EXPECT_EQ(out_.label_map.at<uint8_t>(250, 250), 4);
    // Only head 1 (row 200, column 350: inside head 1's square, right of head 0's)
    EXPECT_EQ(out_.label_map.at<uint8_t>(200, 350), 4);
}

// Empty head masks (no output) are handled
TEST_F(SegProcessTest, EmptyMasksGiveEmptyOutputs) {
    // No masks at all
    SegResult empty;
    // Process
    post_.Process(empty, out_);
    // Empty head mask
    EXPECT_TRUE(out_.heads[0].mask.empty());
    // Empty fused map
    EXPECT_TRUE(out_.label_map.empty());
}

// A custom config with low MaskThreshold but high ConfidenceThreshold drops low-confidence regions
TEST(SegProcessCustomTest, LowMeanConfidenceRegionIsDropped) {
    // Start from the real config
    SegPostConfig c = SegPostConfig::FromYaml(ConfigPath(), kProcedure);
    // Pixels kept from 0.1
    c.mask_threshold = 0.1f;
    // Regions need a mean of 0.8
    c.confidence_threshold = 0.8f;
    // Processor
    SegPostProcessor post(c);
    // Input
    SegResult seg = EmptySeg();
    // Confidence 0.5: passes the pixel test, fails the region test
    Paint(seg, 0, cv::Rect(100, 100, 100, 100), 1, 0.5f);
    // Output
    SegPostResult out;
    // Process
    post.Process(seg, out);
    // Dropped
    EXPECT_TRUE(out.heads[0].regions.empty());
}

// MorphKernelSize 0 disables morphology: thin structures survive
TEST(SegProcessCustomTest, MorphologyOffKeepsThinLines) {
    // Real config
    SegPostConfig c = SegPostConfig::FromYaml(ConfigPath(), kProcedure);
    // No morphology
    c.morph_kernel_size = 0;
    // No area limit
    c.min_area_ratio = 0.0f;
    // Processor
    SegPostProcessor post(c);
    // Input
    SegResult seg = EmptySeg();
    // 2-pixel-wide line (an opening with a 5x5 kernel would erase it)
    Paint(seg, 0, cv::Rect(10, 10, 300, 2), 1, 0.9f);
    // Output
    SegPostResult out;
    // Process
    post.Process(seg, out);
    // Kept
    EXPECT_EQ(out.heads[0].regions.size(), 1u);
}

// With the default 5x5 opening the same thin line is removed
TEST_F(SegProcessTest, MorphologyRemovesThinLines) {
    // 2-pixel-wide line, 600 px
    Paint(seg_, 0, cv::Rect(10, 10, 300, 2), 1, 0.9f);
    // Process
    post_.Process(seg_, out_);
    // Removed
    EXPECT_TRUE(out_.heads[0].regions.empty());
}

// ---------------------------------------------------------------------------
// SegPostProcessor::Draw
// ---------------------------------------------------------------------------

// Fixture: a processed result with one Gallbladder square (head 0) and one Liver square (head 1)
class SegDrawTest : public SegProcessTest {
protected:
    // Before each test
    void SetUp() override {
        // Gallbladder at (50, 300)
        Paint(seg_, 0, cv::Rect(50, 300, 100, 100), 1, 0.9f);
        // Liver at (250, 300)
        Paint(seg_, 1, cv::Rect(250, 300, 100, 100), 1, 0.9f);
        // Process
        post_.Process(seg_, out_);
    }
    // Black input frame
    cv::Mat frame_ = cv::Mat(kSize, CV_8UC3, cv::Scalar::all(0));
    // Output
    cv::Mat vis_;
};

// Fill colour = 0.8 × frame + 0.2 × class colour (alpha 0.2 from the config)
TEST_F(SegDrawTest, FillIsAlphaBlended) {
    // Fused
    post_.Draw(frame_, out_, vis_);
    // Inside the Gallbladder square: green (0, 255, 0) × 0.2 on black = (0, 51, 0)
    EXPECT_EQ(vis_.at<cv::Vec3b>(350, 100), cv::Vec3b(0, 51, 0));
    // Background stays black
    EXPECT_EQ(vis_.at<cv::Vec3b>(480, 480), cv::Vec3b(0, 0, 0));
    // Input not modified
    EXPECT_EQ(cv::countNonZero(frame_.reshape(1)), 0);
}

// The fused legend lists all 15 classes; a single head lists only its own
TEST_F(SegDrawTest, LegendSizeDependsOnHead) {
    // Rows darkened by the legend on a white frame
    auto legend_height = [&](int head) {
        // White frame
        const cv::Mat white(kSize, CV_8UC3, cv::Scalar::all(255));
        // Draw
        post_.Draw(white, out_, vis_, head);
        // Count rows near the right edge that are darker than white
        int rows = 0;
        // Each row
        for (int y = 0; y < vis_.rows; ++y) {
            // Darkened = legend background
            if (vis_.at<cv::Vec3b>(y, vis_.cols - 12)[0] < 200) ++rows;
        }
        // Height
        return rows;
    };
    // Fused (15 classes) is taller than head 1 (2 classes)
    EXPECT_GT(legend_height(-1), 4 * legend_height(1));
    // Head 1 still has a legend
    EXPECT_GT(legend_height(1), 0);
}

// Drawing a single head shows only that head's classes
TEST_F(SegDrawTest, SingleHeadShowsOnlyItsClasses) {
    // Head 1 only (Liver)
    post_.Draw(frame_, out_, vis_, 1);
    // Gallbladder square is not drawn
    EXPECT_EQ(vis_.at<cv::Vec3b>(350, 100), cv::Vec3b(0, 0, 0));
    // Liver square is
    EXPECT_NE(vis_.at<cv::Vec3b>(350, 300), cv::Vec3b(0, 0, 0));
}

// Without a label map (or with a different frame size) the frame is copied unchanged
TEST_F(SegDrawTest, NoLabelMapCopiesFrame) {
    // Empty result
    SegPostResult empty;
    // Draw
    post_.Draw(frame_, empty, vis_);
    // Identical
    EXPECT_EQ(cv::norm(vis_, frame_, cv::NORM_L1), 0.0);
    // Frame of another size
    const cv::Mat other(100, 100, CV_8UC3, cv::Scalar::all(7));
    // Draw the 512x512 result on it
    post_.Draw(other, out_, vis_);
    // Identical copy
    EXPECT_EQ(cv::norm(vis_, other, cv::NORM_L1), 0.0);
}

// Alpha 0 hides a class completely
TEST(SegDrawCustomTest, AlphaZeroHidesClass) {
    // Real config
    SegPostConfig c = SegPostConfig::FromYaml(ConfigPath(), kProcedure);
    // Gallbladder invisible
    c.class_alphas[1] = 0.0f;
    // Processor
    SegPostProcessor post(c);
    // Input
    SegResult seg = EmptySeg();
    // Gallbladder square
    Paint(seg, 0, cv::Rect(50, 300, 100, 100), 1, 0.9f);
    // Process
    SegPostResult out;
    // Run
    post.Process(seg, out);
    // Draw on black
    cv::Mat vis;
    // Fused
    post.Draw(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(0)), out, vis);
    // Square interior unchanged
    EXPECT_EQ(vis.at<cv::Vec3b>(350, 100), cv::Vec3b(0, 0, 0));
}

// Class colour accessor returns the BGR colour
TEST_F(SegProcessTest, ClassColourAccessor) {
    // Liver RGBA (0.4706, 0.7333, 1.0) → BGR ≈ (255, 187, 120)
    const cv::Scalar c = post_.ClassColour(4);
    // B
    EXPECT_NEAR(c[0], 255.0, 0.5);
    // R
    EXPECT_NEAR(c[2], 120.0, 0.5);
    // Config accessor
    EXPECT_EQ(post_.Config().num_classes, 16);
}

}  // namespace
