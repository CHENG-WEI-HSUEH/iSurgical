// Unit tests for src/cap/cap.cpp: CaptureImg on synthetic video files (real-time replay pacing, FPS, frame size).

// Code under test
#include "cap/cap.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::runtime_error
#include <stdexcept>
// std::string
#include <string>

// TempDir, MakeVideo, ElapsedMs
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// Fixture: writes a test video into the scratch folder
class CaptureTest : public ::testing::Test {
protected:
    // Write <frames> frames of 64x48 at <fps>; frame i has blue level 20 × i
    std::string Video(int frames, double fps) {
        // Path
        const std::string path = tmp_ / "clip.mp4";
        // Write it
        EXPECT_TRUE(MakeVideo(path, frames, cv::Size(64, 48), fps,
                              [](int i, cv::Mat& f) { f.setTo(cv::Scalar(20 * i % 256, 0, 0)); }));
        // Path
        return path;
    }
    // Scratch folder
    TempDir tmp_;
};

// A file that does not exist throws with the source in the message
TEST_F(CaptureTest, MissingFileThrows) {
    // Must throw
    try {
        // Open
        CaptureImg cap(std::string("/no/such/video.mp4"));
        // Did not throw
        FAIL() << "no exception";
    // Expected
    } catch (const std::runtime_error& e) {
        // Names the source
        EXPECT_NE(std::string(e.what()).find("Failed to open capture source /no/such/video.mp4"), std::string::npos);
    }
}

// A camera index with no device behind it throws (both constructors)
TEST_F(CaptureTest, MissingCameraThrows) {
    // Silence OpenCV's own warnings about the missing device
    ::testing::internal::CaptureStderr();
    // int constructor
    EXPECT_THROW(CaptureImg(97), std::runtime_error);
    // string constructor with a numeric source = camera index
    EXPECT_THROW(CaptureImg(std::string("98")), std::runtime_error);
    // Drop the warnings
    ::testing::internal::GetCapturedStderr();
}

// Every frame is read once, then Capture returns false
TEST_F(CaptureTest, ReadsEveryFrameThenStops) {
    // 10 frames at 200 fps (fast replay)
    CaptureImg cap(Video(10, 200));
    // Frame buffer
    cv::Mat frame;
    // Count
    int n = 0;
    // Read all
    while (cap.Capture(frame)) ++n;
    // All of them
    EXPECT_EQ(n, 10);
    // Still false after the end
    EXPECT_FALSE(cap.Capture(frame));
}

// Frame size and source frame rate come from the file
TEST_F(CaptureTest, ReportsFrameSizeAndSourceFps) {
    // 25 fps clip
    CaptureImg cap(Video(3, 25));
    // 64x48
    EXPECT_EQ(cap.GetFrameSize(), cv::Size(64, 48));
    // 25 fps
    EXPECT_NEAR(cap.GetSourceFPS(), 25.0, 0.01);
}

// A file is not a live source
TEST_F(CaptureTest, FileIsNotLive) {
    // Any clip
    CaptureImg cap(Video(2, 25));
    // Not a camera or stream
    EXPECT_FALSE(cap.IsLive());
}

// Frames come back in order with their content (mp4v is lossy, so allow a small error)
TEST_F(CaptureTest, FramesKeepOrderAndContent) {
    // 6 frames, blue = 0, 20, 40, ...
    CaptureImg cap(Video(6, 200));
    // Frame buffer
    cv::Mat frame;
    // Each frame
    for (int i = 0; i < 6; ++i) {
        // Read
        ASSERT_TRUE(cap.Capture(frame)) << i;
        // Mean blue
        EXPECT_NEAR(cv::mean(frame)[0], 20.0 * i, 8.0) << i;
        // Size
        EXPECT_EQ(frame.size(), cv::Size(64, 48));
    }
}

// Video files are replayed in real time: 10 frames at 20 fps take about 9 / 20 = 0.45 s
TEST_F(CaptureTest, ReplayIsPacedToSourceFps) {
    // 20 fps clip
    CaptureImg cap(Video(10, 20));
    // Frame buffer
    cv::Mat frame;
    // Start
    const auto start = std::chrono::steady_clock::now();
    // Read all
    while (cap.Capture(frame)) {}
    // Elapsed
    const long long ms = ElapsedMs(start);
    // At least the 9 frame intervals (small tolerance for timer granularity)
    EXPECT_GE(ms, 430);
    // Not much slower
    EXPECT_LT(ms, 1500);
}

// The measured FPS is 0 until one second has passed, then close to the source rate
TEST_F(CaptureTest, MeasuredFpsAfterOneSecond) {
    // 30 frames at 20 fps = 1.45 s of replay
    CaptureImg cap(Video(30, 20));
    // Nothing measured yet
    EXPECT_DOUBLE_EQ(cap.GetFPS(), 0.0);
    // Frame buffer
    cv::Mat frame;
    // Read all
    while (cap.Capture(frame)) {}
    // About 20 fps
    EXPECT_NEAR(cap.GetFPS(), 20.0, 3.0);
}

}  // namespace
