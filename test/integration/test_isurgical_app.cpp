// End-to-end tests for iSurgical::Run (src/isurgical/isurgical.cpp) with the fake models on CPU:
// image mode, the 4-thread video pipeline, recording, shared memory, and every failure path of Init().

// Code under test
#include "isurgical/isurgical.hpp"
// SharedState, ReadStatus (the parent's side of the shared memory)
#include "common/isurgical_shm.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::function
#include <functional>
// std::string
#include <string>

// shm_open, mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// TempDir, FakeModel, ConfigPath, ColourFrame, MakeVideo, CountFrames
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// What a Run produced
struct RunResult {
    // Run()'s return value
    int code = 0;
    // std::cout
    std::string out;
    // std::cerr
    std::string err;
};

// Run the app with stdout / stderr captured. Exception-safe: if Run() throws, the captures are ended
// first and the exception is passed on (gtest aborts if a capture is started while another is active)
RunResult RunApp(const iSurgicalConfig& config) {
    // Capture std::cout
    ::testing::internal::CaptureStdout();
    // Capture std::cerr
    ::testing::internal::CaptureStderr();
    // Result
    RunResult r;
    // Run, catching anything it throws
    try {
        // Run
        r.code = iSurgical(config).Run();
    // Run() threw
    } catch (...) {
        // End the stdout capture
        ::testing::internal::GetCapturedStdout();
        // End the stderr capture
        ::testing::internal::GetCapturedStderr();
        // Let the caller (EXPECT_NO_THROW) see the exception
        throw;
    }
    // Collect stdout
    r.out = ::testing::internal::GetCapturedStdout();
    // Collect stderr
    r.err = ::testing::internal::GetCapturedStderr();
    // Done
    return r;
}

// A shared-memory segment created the way the parent process (iProvision) creates it
class ShmSegment {
public:
    // Create /<name> of `size` bytes, zero-filled, with the given magic / version
    ShmSegment(const std::string& name, size_t size, uint32_t magic = isurgical_shm::kMagic,
               uint32_t version = isurgical_shm::kVersion)
        : name_(name), size_(size) {
        // Remove a leftover from a crashed run
        shm_unlink(name_.c_str());
        // Create
        const int fd = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
        // Size (new pages are zero)
        if (fd >= 0 && ftruncate(fd, static_cast<off_t>(size_)) == 0) {
            // Map
            void* addr = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            // Keep it
            if (addr != MAP_FAILED) mem_ = addr;
        }
        // The mapping keeps the segment alive
        if (fd >= 0) close(fd);
        // Big enough to hold the header: set magic and version
        if (mem_ && size_ >= sizeof(isurgical_shm::SharedState)) {
            // Header
            State().magic = magic;
            // Version
            State().version = version;
        }
    }
    // Unmap and remove
    ~ShmSegment() {
        // Unmap
        if (mem_) munmap(mem_, size_);
        // Remove the name
        shm_unlink(name_.c_str());
    }
    // Creation worked
    bool Ok() const { return mem_ != nullptr; }
    // The state as the parent sees it
    isurgical_shm::SharedState& State() { return *static_cast<isurgical_shm::SharedState*>(mem_); }
    // The name passed to --shm
    const std::string& Name() const { return name_; }

private:
    // "/isurgical_test_..."
    std::string name_;
    // Bytes
    size_t size_;
    // Mapping
    void* mem_ = nullptr;
};

// Unique shared-memory name for this process and test
std::string ShmName(const char* tag) {
    // e.g. /isurgical_test_1234_img
    return "/isurgical_test_" + std::to_string(getpid()) + "_" + tag;
}

// Fixture: a config using the fake models on CPU, with an image input
class AppTest : public ::testing::Test {
protected:
    // Before each test
    void SetUp() override {
        // Fake segmentation model
        config_.model_path = FakeModel("seg.onnx");
        // Fake RSD models
        config_.rsd_preprocess_path = FakeModel("rsd_pre.onnx");
        // Fake RSD model
        config_.rsd_model_path = FakeModel("rsd.onnx");
        // Real procedure config
        config_.config_path = ConfigPath();
        // CPU: deterministic and works without a GPU
        config_.use_cuda = false;
        // Test image: red square (Gallbladder) + yellow square (Liver)
        image_ = tmp_ / "frame.png";
        // Write it
        cv::imwrite(image_, ColourFrame());
        // Image mode
        config_.input_method = "image";
        // That file
        config_.input = image_;
    }

    // Write a test video (frame i: the colour frame, with a moving white dot) and select it as input
    void UseVideo(int frames, double fps) {
        // Path
        const std::string path = tmp_ / "clip.mp4";
        // Write
        ASSERT_TRUE(MakeVideo(path, frames, cv::Size(160, 120), fps, [](int i, cv::Mat& f) {
            // Base picture
            ColourFrame(f.size()).copyTo(f);
            // Moving dot so frames differ
            cv::circle(f, cv::Point(10 + i, 10), 3, cv::Scalar::all(255), cv::FILLED);
        }));
        // Video mode
        config_.input_method = "video";
        // That file
        config_.input = path;
    }

    // Scratch folder
    TempDir tmp_;
    // The config under test
    iSurgicalConfig config_;
    // Input image path
    std::string image_;
};

// ---------------------------------------------------------------------------
// Image mode
// ---------------------------------------------------------------------------

// Image mode: segmentation + RSD, both outputs written, results printed
TEST_F(AppTest, ImageModeWritesOutputsAndPrintsResult) {
    // Annotated output
    config_.output = tmp_ / "vis.png";
    // Raw output
    config_.raw_output = tmp_ / "raw.png";
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // Raw = input, pixel for pixel (PNG is lossless)
    EXPECT_EQ(cv::norm(cv::imread(config_.raw_output), cv::imread(image_), cv::NORM_L1), 0.0);
    // Annotated differs from the input (overlay + legend + RSD panel)
    EXPECT_GT(cv::norm(cv::imread(config_.output), cv::imread(image_), cv::NORM_L1), 0.0);
    // Per-frame line with the correlation scores
    EXPECT_NE(r.out.find("frame 0"), std::string::npos) << r.out;
    // Both regions found
    EXPECT_NE(r.out.find("[Gallbladder area"), std::string::npos) << r.out;
    // Liver from head 1
    EXPECT_NE(r.out.find("[Liver area"), std::string::npos) << r.out;
    // RSD line: fake model → phase 2 (ClippingCutting) with 0.70, 60 s remaining
    EXPECT_NE(r.out.find("[rsd step 1 @ 00:00] phase: ClippingCutting (0.70)  RSD: 01:00"), std::string::npos)
        << r.out;
}

// --no-rsd: no RSD model is loaded (bogus paths are fine) and nothing RSD-related is printed
TEST_F(AppTest, ImageModeWithoutRsd) {
    // Off
    config_.rsd = false;
    // Would fail if loaded
    config_.rsd_model_path = "/no/such/rsd.onnx";
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // No RSD line
    EXPECT_EQ(r.out.find("rsd step"), std::string::npos) << r.out;
}

// --head selects one head for drawing; the run still succeeds
TEST_F(AppTest, ImageModeSingleHead) {
    // Head 1 only
    config_.head = 1;
    // Output
    config_.output = tmp_ / "head1.png";
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // Output exists
    EXPECT_FALSE(cv::imread(config_.output).empty());
}

// An unreadable image fails with a message
TEST_F(AppTest, ImageModeUnreadableImage) {
    // Not an image
    config_.input = tmp_ / "missing.png";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Could not read the image"), std::string::npos) << r.err;
}

// An output file that cannot be written fails with a message (the folder exists, the file cannot be created)
TEST_F(AppTest, ImageModeOutputNotWritable) {
    // /proc exists but nothing can be created in it
    config_.output = "/proc/isurgical_vis.png";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Failed to write /proc/isurgical_vis.png"), std::string::npos) << r.err;
}

// Same for the raw output
TEST_F(AppTest, ImageModeRawOutputNotWritable) {
    // Not writable
    config_.raw_output = "/proc/isurgical_raw.png";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Failed to write /proc/isurgical_raw.png"), std::string::npos) << r.err;
}

// An output with an unknown image extension should make Run() return -1.
// KNOWN DEFECT: cv::imwrite throws cv::Exception ("could not find a writer for the specified extension"),
// RunImage does not catch it, so Run() throws instead of returning -1 (the CLI would abort).
TEST_F(AppTest, ImageModeUnknownOutputExtensionReturnsError) {
    // ".xyz" has no image writer
    config_.output = tmp_ / "vis.xyz";
    // Result
    RunResult r;
    // Must not throw
    EXPECT_NO_THROW(r = RunApp(config_));
    // Must report failure
    EXPECT_EQ(r.code, -1);
}

// ---------------------------------------------------------------------------
// Init() failure paths (each fails before or while loading, returns -1, explains why)
// ---------------------------------------------------------------------------

// One failure scenario: how to break the config, and the expected message
struct InitFailure {
    // Test name
    const char* name;
    // Change applied to a working config
    std::function<void(iSurgicalConfig&)> breakit;
    // Expected text on stderr
    std::string message;
};

// Parameterised fixture
class InitFailureTest : public AppTest, public ::testing::WithParamInterface<InitFailure> {};

// Each scenario
TEST_P(InitFailureTest, ReturnsMinusOneWithMessage) {
    // Break the config
    GetParam().breakit(config_);
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Explained
    EXPECT_NE(r.err.find(GetParam().message), std::string::npos) << r.err;
}

// The scenarios
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Cases,
    // Fixture
    InitFailureTest,
    // Rows
    ::testing::Values(
        // Head too large
        InitFailure{"HeadTooLarge", [](iSurgicalConfig& c) { c.head = 4; }, "head must be in [0, 3]"},
        // Head below -1
        InitFailure{"HeadTooSmall", [](iSurgicalConfig& c) { c.head = -2; }, "head must be in [0, 3]"},
        // Output folder missing
        InitFailure{"OutputDirMissing", [](iSurgicalConfig& c) { c.output = "/no/such/dir/out.mp4"; },
                    "Output directory does not exist: /no/such/dir"},
        // Raw output folder missing
        InitFailure{"RawOutputDirMissing", [](iSurgicalConfig& c) { c.raw_output = "/no/such/dir/raw.mp4"; },
                    "Output directory does not exist"},
        // Config file missing
        InitFailure{"ConfigMissing", [](iSurgicalConfig& c) { c.config_path = "/no/such/config.yaml"; },
                    "Invalid config"},
        // Procedure missing from the config
        InitFailure{"ProcedureMissing", [](iSurgicalConfig& c) { c.procedure = "appendectomy"; },
                    "missing procedure 'appendectomy'"},
        // Segmentation model missing
        InitFailure{"SegModelMissing", [](iSurgicalConfig& c) { c.model_path = "/no/such/seg.onnx"; },
                    "Failed to load model /no/such/seg.onnx"},
        // Segmentation model with the wrong outputs
        InitFailure{"SegModelWrongOutputs",
                    [](iSurgicalConfig& c) { c.model_path = FakeModel("seg_missing_scores.onnx"); },
                    "Unexpected model output count"},
        // RSD model missing
        InitFailure{"RsdModelMissing", [](iSurgicalConfig& c) { c.rsd_model_path = "/no/such/rsd.onnx"; },
                    "Failed to load phase / RSD models"},
        // Shared memory that does not exist
        InitFailure{"ShmMissing", [](iSurgicalConfig& c) { c.shm_name = "/isurgical_test_does_not_exist"; },
                    "Cannot open shared memory /isurgical_test_does_not_exist"},
        // Unknown input method (fails after Init succeeded)
        InitFailure{"UnknownInputMethod", [](iSurgicalConfig& c) { c.input_method = "fax"; },
                    "Unknown input method: fax"}),
    // Row name
    [](const ::testing::TestParamInfo<InitFailure>& info) { return std::string(info.param.name); });

// ---------------------------------------------------------------------------
// Video mode (capture → inference → display threads, RSD side thread)
// ---------------------------------------------------------------------------

// The raw recording keeps every source frame; the annotated one keeps the source timeline
TEST_F(AppTest, VideoModeRecordsRawAndAnnotated) {
    // 20 frames at 25 fps (0.8 s replay)
    UseVideo(20, 25);
    // Annotated video
    config_.output = tmp_ / "vis.mp4";
    // Raw video
    config_.raw_output = tmp_ / "raw.mp4";
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // Every source frame in the raw video
    EXPECT_EQ(CountFrames(config_.raw_output), 20);
    // Annotated: skipped frames are filled, so the count runs up to the last processed frame
    const int annotated = CountFrames(config_.output);
    // At least one
    EXPECT_GE(annotated, 1);
    // At most all
    EXPECT_LE(annotated, 20);
    // Summary line
    EXPECT_NE(r.out.find("frames skipped by inference:"), std::string::npos) << r.out;
    // First RSD step printed
    EXPECT_NE(r.out.find("[rsd step 1 @ 00:00]"), std::string::npos) << r.out;
    // Latency is reported per frame
    EXPECT_NE(r.out.find("latency"), std::string::npos) << r.out;
}

// Video mode without outputs or RSD still processes frames
TEST_F(AppTest, VideoModeConsoleOnly) {
    // 10 frames
    UseVideo(10, 50);
    // No RSD
    config_.rsd = false;
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // At least the first frame was processed
    EXPECT_NE(r.out.find("frame 0"), std::string::npos) << r.out;
}

// "camera" and "rtsp" input methods use the same stream pipeline (a file stands in for the device here)
TEST_F(AppTest, CameraAndRtspMethodsUseStreamPipeline) {
    // Clip
    UseVideo(5, 50);
    // Each method
    for (const char* method : {"camera", "rtsp"}) {
        // Method
        config_.input_method = method;
        // Run
        const RunResult r = RunApp(config_);
        // Success
        EXPECT_EQ(r.code, 0) << method << ": " << r.err;
    }
}

// A video that cannot be opened fails
TEST_F(AppTest, VideoModeMissingInput) {
    // Video mode
    config_.input_method = "video";
    // Not there
    config_.input = tmp_ / "missing.mp4";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Failed to open capture source"), std::string::npos) << r.err;
}

// The annotated video cannot be created → stop with -1
TEST_F(AppTest, VideoModeOutputNotWritable) {
    // Clip
    UseVideo(10, 50);
    // /proc exists, files cannot be created
    config_.output = "/proc/isurgical_vis.mp4";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Stopped: could not record to /proc/isurgical_vis.mp4"), std::string::npos) << r.err;
}

// The raw video cannot be created → stop with -1
TEST_F(AppTest, VideoModeRawOutputNotWritable) {
    // Clip
    UseVideo(10, 50);
    // Not writable
    config_.raw_output = "/proc/isurgical_raw.mp4";
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("Stopped: could not record to /proc/isurgical_raw.mp4"), std::string::npos) << r.err;
}

// ---------------------------------------------------------------------------
// Shared memory (--shm)
// ---------------------------------------------------------------------------

// Image mode publishes the tables once and the status of frame 0
TEST_F(AppTest, ShmPublishesTablesAndStatus) {
    // Segment created like the parent does
    ShmSegment shm(ShmName("img"), sizeof(isurgical_shm::SharedState));
    // Created
    ASSERT_TRUE(shm.Ok());
    // Attach
    config_.shm_name = shm.Name();
    // Run
    const RunResult r = RunApp(config_);
    // Success
    ASSERT_EQ(r.code, 0) << r.err;
    // The state
    isurgical_shm::SharedState& s = shm.State();
    // Tables are ready
    EXPECT_EQ(s.tables_ready.load(), 1u);
    // 7 phases
    EXPECT_EQ(s.tables.num_phases, 7);
    // Third phase
    EXPECT_STREQ(s.tables.phase_names[2], "ClippingCutting");
    // 16 classes
    EXPECT_EQ(s.tables.num_classes, 16);
    // Class name
    EXPECT_STREQ(s.tables.class_names[1], "Gallbladder");
    // Gallbladder colour RGB (0, 1, 0)
    EXPECT_FLOAT_EQ(s.tables.class_colours[1][1], 1.0f);
    // Red component
    EXPECT_FLOAT_EQ(s.tables.class_colours[1][0], 0.0f);
    // Read the status as the parent would
    isurgical_shm::Status status{};
    // One write → sequence 2
    EXPECT_EQ(isurgical_shm::ReadStatus(s, status), 2u);
    // Frame 0
    EXPECT_EQ(status.frame_index, 0);
    // One Gallbladder region
    EXPECT_EQ(status.detected[1], 1);
    // One Liver region
    EXPECT_EQ(status.detected[4], 1);
    // Nothing else
    EXPECT_EQ(status.detected[2], 0);
    // Phase name
    EXPECT_STREQ(status.phase, "ClippingCutting");
    // 60 s → 1 minute
    EXPECT_EQ(status.remaining_minutes, 1);
    // Progress in [0, 1], rounded to 2 decimals
    EXPECT_GE(status.progress, 0.0f);
    // Upper bound
    EXPECT_LE(status.progress, 1.0f);
    // Timestamp set
    EXPECT_GT(status.timestamp, 0);
}

// Without RSD the status says "unknown" for phase / progress / minutes
TEST_F(AppTest, ShmStatusWithoutRsd) {
    // Segment
    ShmSegment shm(ShmName("norsd"), sizeof(isurgical_shm::SharedState));
    // Created
    ASSERT_TRUE(shm.Ok());
    // Attach
    config_.shm_name = shm.Name();
    // No RSD
    config_.rsd = false;
    // Run
    ASSERT_EQ(RunApp(config_).code, 0);
    // Status
    isurgical_shm::Status status{};
    // Read
    ASSERT_NE(isurgical_shm::ReadStatus(shm.State(), status), 0u);
    // No phase
    EXPECT_STREQ(status.phase, "");
    // Unknown progress
    EXPECT_FLOAT_EQ(status.progress, -1.0f);
    // Unknown minutes
    EXPECT_EQ(status.remaining_minutes, -1);
}

// A segment smaller than SharedState is refused
TEST_F(AppTest, ShmTooSmallIsRefused) {
    // 16 bytes
    ShmSegment shm(ShmName("small"), 16);
    // Created
    ASSERT_TRUE(shm.Ok());
    // Attach
    config_.shm_name = shm.Name();
    // Run
    const RunResult r = RunApp(config_);
    // Failure
    EXPECT_EQ(r.code, -1);
    // Message
    EXPECT_NE(r.err.find("is too small"), std::string::npos) << r.err;
}

// Wrong magic or version: refused (not an iSurgical segment, or another layout version)
TEST_F(AppTest, ShmWrongMagicOrVersionIsRefused) {
    // Wrong magic
    {
        // Segment with magic 0
        ShmSegment shm(ShmName("magic"), sizeof(isurgical_shm::SharedState), 0, isurgical_shm::kVersion);
        // Created
        ASSERT_TRUE(shm.Ok());
        // Attach
        config_.shm_name = shm.Name();
        // Run
        const RunResult r = RunApp(config_);
        // Failure
        EXPECT_EQ(r.code, -1);
        // Message
        EXPECT_NE(r.err.find("has an unexpected layout"), std::string::npos) << r.err;
    }
    // Wrong version
    {
        // Version 2
        ShmSegment shm(ShmName("version"), sizeof(isurgical_shm::SharedState), isurgical_shm::kMagic, 2);
        // Created
        ASSERT_TRUE(shm.Ok());
        // Attach
        config_.shm_name = shm.Name();
        // Run
        const RunResult r = RunApp(config_);
        // Failure
        EXPECT_EQ(r.code, -1);
        // Message names both versions
        EXPECT_NE(r.err.find("(version 2, expected 1)"), std::string::npos) << r.err;
    }
}

// Video mode publishes a status for processed frames
TEST_F(AppTest, ShmVideoModePublishesEveryFrame) {
    // Clip
    UseVideo(15, 50);
    // Segment
    ShmSegment shm(ShmName("video"), sizeof(isurgical_shm::SharedState));
    // Created
    ASSERT_TRUE(shm.Ok());
    // Attach
    config_.shm_name = shm.Name();
    // Run
    ASSERT_EQ(RunApp(config_).code, 0);
    // Status
    isurgical_shm::Status status{};
    // At least one write
    EXPECT_GE(isurgical_shm::ReadStatus(shm.State(), status), 2u);
    // Last frame index within the clip
    EXPECT_LT(status.frame_index, 15);
    // Regions counted
    EXPECT_EQ(status.detected[1], 1);
}

// The parent's stop_requested flag ends a video run cleanly after the first frame
TEST_F(AppTest, ShmStopRequestEndsRunEarly) {
    // 75 frames at 25 fps = 3 s of replay
    UseVideo(75, 25);
    // Segment
    ShmSegment shm(ShmName("stop"), sizeof(isurgical_shm::SharedState));
    // Created
    ASSERT_TRUE(shm.Ok());
    // Stop requested before the run starts
    shm.State().stop_requested.store(1);
    // Attach
    config_.shm_name = shm.Name();
    // Start time
    const auto start = std::chrono::steady_clock::now();
    // Run
    const RunResult r = RunApp(config_);
    // Clean exit
    EXPECT_EQ(r.code, 0) << r.err;
    // Well before the 3 s of video
    EXPECT_LT(ElapsedMs(start), 2000);
    // Exactly one status written
    isurgical_shm::Status status{};
    // Sequence 2 = one write
    EXPECT_EQ(isurgical_shm::ReadStatus(shm.State(), status), 2u);
}

}  // namespace
