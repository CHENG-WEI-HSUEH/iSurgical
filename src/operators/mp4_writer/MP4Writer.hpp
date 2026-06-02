#pragma once

#ifndef MP4_WRITER_OPERATOR_HPP
#define MP4_WRITER_OPERATOR_HPP

#include <holoscan/holoscan.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

class Mp4WriterOp : public holoscan::Operator {
public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Mp4WriterOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&, holoscan::ExecutionContext&) override;

private:
  holoscan::Parameter<std::string> output_path_;
  holoscan::Parameter<int> width_;
  holoscan::Parameter<int> height_;
  holoscan::Parameter<int> fps_;

  cv::VideoWriter writer_;
  bool initialized_ = false;
};

#endif // MP4_WRITER_OPERATOR_HPP