// Unit tests for src/record/record.cpp: VideoRecord (background-thread video writer with a bounded queue).

// Code under test
#include "record/record.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::string
#include <string>

// TempDir, CountFrames
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Helpers
using namespace test_helpers;

// The codec iSurgical uses
const int kMp4v = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
// Small frames keep the tests fast
const cv::Size kSize(64, 48);

// Fixture: scratch folder
class RecordTest : public ::testing::Test {
protected:
    // Output path
    std::string Out() const { return tmp_ / "out.mp4"; }
    // Scratch folder
    TempDir tmp_;
};

// Every frame written is in the file after Release
TEST_F(RecordTest, WritesEveryFrame) {
    // Recorder
    VideoRecord rec(Out(), kMp4v, 25, kSize);
    // Opened
    ASSERT_TRUE(rec.IsOpened());
    // 20 frames
    for (int i = 0; i < 20; ++i) ASSERT_TRUE(rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(i * 10))));
    // Flush and close
    rec.Release();
    // No longer open
    EXPECT_FALSE(rec.IsOpened());
    // 20 frames on disk
    EXPECT_EQ(CountFrames(Out()), 20);
}

// A queue of one still loses nothing (WriteFrame blocks instead of dropping)
TEST_F(RecordTest, TinyQueueLosesNothing) {
    // Queue size 1
    VideoRecord rec(Out(), kMp4v, 25, kSize, 1);
    // 50 frames as fast as possible
    for (int i = 0; i < 50; ++i) ASSERT_TRUE(rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(i))));
    // Close
    rec.Release();
    // All 50
    EXPECT_EQ(CountFrames(Out()), 50);
}

// The caller may reuse its buffer right after WriteFrame (the frame is copied)
TEST_F(RecordTest, FrameIsCopiedSoBufferCanBeReused) {
    // Recorder
    VideoRecord rec(Out(), kMp4v, 25, kSize);
    // One buffer for every frame
    cv::Mat buffer(kSize, CV_8UC3);
    // First frame white
    buffer.setTo(cv::Scalar::all(255));
    // Write
    rec.WriteFrame(buffer);
    // Immediately overwritten with black
    buffer.setTo(cv::Scalar::all(0));
    // Write
    rec.WriteFrame(buffer);
    // Close
    rec.Release();
    // Read back
    cv::VideoCapture cap(Out());
    // Frame
    cv::Mat frame;
    // First
    ASSERT_TRUE(cap.read(frame));
    // Still white
    EXPECT_GT(cv::mean(frame)[0], 200.0);
    // Second
    ASSERT_TRUE(cap.read(frame));
    // Black
    EXPECT_LT(cv::mean(frame)[0], 50.0);
}

// Frames of the wrong size are rejected (and not written)
TEST_F(RecordTest, WrongSizeIsRejected) {
    // Recorder for 64x48
    VideoRecord rec(Out(), kMp4v, 25, kSize);
    // Silence the message
    ::testing::internal::CaptureStderr();
    // 32x32 rejected
    EXPECT_FALSE(rec.WriteFrame(cv::Mat(32, 32, CV_8UC3, cv::Scalar::all(0))));
    // Message
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("does not match recorder size"), std::string::npos);
    // A good frame
    rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(0)));
    // Close
    rec.Release();
    // Only the good one
    EXPECT_EQ(CountFrames(Out()), 1);
}

// Empty frames are rejected
TEST_F(RecordTest, EmptyFrameIsRejected) {
    // Recorder
    VideoRecord rec(Out(), kMp4v, 25, kSize);
    // Empty
    EXPECT_FALSE(rec.WriteFrame(cv::Mat()));
}

// After Release, WriteFrame fails and Release can be called again
TEST_F(RecordTest, WriteAfterReleaseFailsAndReleaseIsIdempotent) {
    // Recorder
    VideoRecord rec(Out(), kMp4v, 25, kSize);
    // Close
    rec.Release();
    // Rejected
    EXPECT_FALSE(rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(0))));
    // Second Release: no crash, no hang
    rec.Release();
}

// The destructor flushes the queue and finishes the file
TEST_F(RecordTest, DestructorFinishesFile) {
    // Scope
    {
        // Recorder
        VideoRecord rec(Out(), kMp4v, 25, kSize);
        // 12 frames
        for (int i = 0; i < 12; ++i) rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(i * 20)));
    }
    // All on disk
    EXPECT_EQ(CountFrames(Out()), 12);
}

// A file that cannot be created: not opened, and every write fails
TEST_F(RecordTest, UnwritablePathIsNotOpened) {
    // Capture OpenCV's and VideoRecord's messages
    ::testing::internal::CaptureStderr();
    // Folder does not exist
    VideoRecord rec("/no/such/dir/out.mp4", kMp4v, 25, kSize);
    // Message from VideoRecord
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("Failed to open video writer"), std::string::npos);
    // Not open
    EXPECT_FALSE(rec.IsOpened());
    // Writes fail
    EXPECT_FALSE(rec.WriteFrame(cv::Mat(kSize, CV_8UC3, cv::Scalar::all(0))));
    // Release is a no-op
    rec.Release();
}

}  // namespace
