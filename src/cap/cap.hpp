#pragma once

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

class CaptureImg {
  public:
    // source: camera index ("0"), video file or stream URL.
    explicit CaptureImg(int camera_index = 0);
    // Video files are replayed in real time at their own frame rate, like a camera.
    explicit CaptureImg(const std::string& source);
    ~CaptureImg();

    // Capture a frame from the source. Returns false if the source is closed or the frame is empty.
    bool Capture(cv::Mat& frame);

    // measured capture rate, updated once per second
    double GetFPS() const;     
    // rate reported by the source, 30 if unknown     
    double GetSourceFPS() const;    

    // Get the size of the captured frames
    cv::Size GetFrameSize() const;
    bool IsLive() const;            // camera or network stream, as opposed to a video file
  private:
    void Open(const std::string& source);
    void PaceReplay();

    std::unique_ptr<cv::VideoCapture> cap_;
    std::string source_;
    std::atomic<double> fps_;
    int frame_count_;
    std::chrono::steady_clock::time_point fps_start_;
    std::chrono::steady_clock::time_point replay_start_;
    int64_t replay_index_;
};
