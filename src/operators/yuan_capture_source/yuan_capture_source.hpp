#ifndef  _YUAN_CAPTURE_SOURCE_HPP_
#define  _YUAN_CAPTURE_SOURCE_HPP_

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>

class YuanCaptureSourceOp : public holoscan::Operator {
  public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(YuanCaptureSourceOp)
    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
                  holoscan::OutputContext& op_output,
                  holoscan::ExecutionContext& context) override;
    void framePumper();

  private:
    holoscan::Parameter<std::string> device_;
    holoscan::Parameter<int> width_;
    holoscan::Parameter<int> height_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

    cv::VideoCapture cap_;
    cudaStream_t cuda_stream_ = nullptr;

    std::thread capture_thread_;
    std::atomic<bool> is_running_{false};
    std::mutex frame_mutex_;
    std::queue<cv::Mat> frame_queue_;
};
#endif //  "_YUAN_CAPTURE_SOURCE_HPP_"