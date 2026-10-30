#include "cap.hpp"

#include <stdexcept>
#include <thread>

CaptureImg::CaptureImg(int camera_index)
    : CaptureImg(std::to_string(camera_index)) {}

CaptureImg::CaptureImg(const std::string& source)
    : cap_(std::make_unique<cv::VideoCapture>()),
      source_(source),
      fps_(0.0),
      frame_count_(0),
      fps_start_(std::chrono::steady_clock::now()),
      replay_index_(0) {
    Open(source_);
}

CaptureImg::~CaptureImg() {
    cap_->release();
}

// Open the capture source, which can be a camera index, video file, or stream URL.
void CaptureImg::Open(const std::string& source) {
    const bool is_index = !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
    const bool opened = is_index ? cap_->open(std::stoi(source)) : cap_->open(source);
    if (!opened || !cap_->isOpened()) {
        throw std::runtime_error("Failed to open capture source " + source);
    }
}

// Capture a frame from the source. Returns false if the source is closed or the frame is empty.
bool CaptureImg::Capture(cv::Mat& frame) {
    if (!cap_->isOpened() || !cap_->read(frame) || frame.empty()) {
        return false;
    }
    if (!IsLive()) {
        PaceReplay();
    }

    // Measured FPS over one-second windows
    ++frame_count_;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - fps_start_).count();
    if (elapsed >= 1.0) {
        fps_ = frame_count_ / elapsed;
        frame_count_ = 0;
        fps_start_ = now;
    }
    return true;
}

// Frame pacing for video files, which are replayed in real time at their own frame rate, like a camera.
void CaptureImg::PaceReplay() {
    // Frame n is released at replay_start_ + n / fps; if decoding falls behind, frames are returned immediately
    const auto now = std::chrono::steady_clock::now();
    if (replay_index_ == 0) {
        replay_start_ = now;
    } else {
        const auto due = replay_start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(replay_index_ / GetSourceFPS()));
        std::this_thread::sleep_until(due);
    }
    ++replay_index_;
}

// Get the measured capture rate, updated once per second
double CaptureImg::GetFPS() const {
    return fps_;
}

// Get the source frame rate, defaulting to 30 if unknown
double CaptureImg::GetSourceFPS() const {
    const double fps = cap_->get(cv::CAP_PROP_FPS);
    return fps > 0.0 ? fps : 30.0;
}

// Get the size of the captured frames
cv::Size CaptureImg::GetFrameSize() const {
    return cv::Size(static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH)),
                    static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT)));
}

bool CaptureImg::IsLive() const {
    const bool is_index = !source_.empty() && source_.find_first_not_of("0123456789") == std::string::npos;
    return is_index || source_.find("://") != std::string::npos;
}
