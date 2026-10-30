#include "record.hpp"

#include <iostream>

VideoRecord::VideoRecord(const std::string& filename, int codec, double fps, cv::Size frame_size,
                         size_t max_queue_size)
    : writer_(std::make_unique<cv::VideoWriter>(filename, cv::CAP_FFMPEG, codec, fps, frame_size)),
      frame_size_(frame_size),
      max_queue_size_(max_queue_size) {
    if (!writer_->isOpened()) {
        std::cerr << "Failed to open video writer for file: " << filename << std::endl;
        return;
    }
    is_open_ = true;
    write_thread_ = std::thread(&VideoRecord::WriteFrameThread, this);
}

VideoRecord::~VideoRecord() {
    Release();
}

bool VideoRecord::IsOpened() const {
    return is_open_;
}

void VideoRecord::WriteFrameThread() {
    while (true) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) break;      // stop_ is set and everything is flushed
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        cv_.notify_all();                   // wake a producer waiting on a full queue
        writer_->write(frame);
    }
}

bool VideoRecord::WriteFrame(const cv::Mat& frame) {
    if (!is_open_ || frame.empty()) {
        return false;
    }
    if (frame.size() != frame_size_) {
        std::cerr << "Frame size " << frame.size() << " does not match recorder size " << frame_size_ << std::endl;
        return false;
    }

    cv::Mat copy = frame.clone();           // caller may reuse its buffer right after this call
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return stop_ || queue_.size() < max_queue_size_; });
        if (stop_) return false;
        queue_.push_back(std::move(copy));
    }
    cv_.notify_all();
    return true;
}

void VideoRecord::Release() {
    if (!is_open_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (write_thread_.joinable()) {
        write_thread_.join();
    }
    writer_->release();
    is_open_ = false;
}
