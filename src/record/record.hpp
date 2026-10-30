#pragma once

#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Writes frames to a video file on a background thread.
// WriteFrame copies the frame into a bounded queue; it blocks only when the queue is full.
class VideoRecord {
public:
    VideoRecord(const std::string& filename, int codec, double fps, cv::Size frame_size,
                size_t max_queue_size = 64);
    ~VideoRecord();
    VideoRecord(const VideoRecord&) = delete;
    VideoRecord& operator=(const VideoRecord&) = delete;

    bool IsOpened() const;
    bool WriteFrame(const cv::Mat& frame);
    // Flushes queued frames, stops the writer thread and closes the file
    void Release();
private:
    void WriteFrameThread();

    std::unique_ptr<cv::VideoWriter> writer_;
    cv::Size frame_size_;
    size_t max_queue_size_;
    std::deque<cv::Mat> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread write_thread_;
    bool is_open_ = false;
    bool stop_ = false;
};
