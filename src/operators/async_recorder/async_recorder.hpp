#ifndef ASYNC_RECORDER_HPP
#define ASYNC_RECORDER_HPP

#include <holoscan/holoscan.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

struct RecordFrame {
  cv::Mat raw;
  cv::Mat segmentation;
};

class AsyncRecorderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncRecorderOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> recording_path_;
  holoscan::Parameter<int> source_width_;
  holoscan::Parameter<int> source_height_;
  holoscan::Parameter<int> overlay_width_;
  holoscan::Parameter<int> overlay_height_;
  holoscan::Parameter<int> fps_;

  // Writer thread
  std::thread writer_thread_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<RecordFrame> frame_queue_;
  std::atomic<bool> is_running_{false};
  std::atomic<uint64_t> frames_written_{0};

  cv::VideoWriter raw_writer_;
  cv::VideoWriter seg_writer_;

  // Latest raw frame (updated via kNone polling)
  std::mutex raw_buffer_mutex_;
  cv::Mat raw_buffer_;
  bool has_raw_ = false;

  void writerLoop();
};

#endif // ASYNC_RECORDER_HPP
