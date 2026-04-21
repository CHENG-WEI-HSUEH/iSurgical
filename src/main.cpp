#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

class MyQCAPSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MyQCAPSourceOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(device_, "device", "Device", "Video device", std::string("/dev/video0"));
    spec.param(allocator_, "allocator", "Allocator", "Allocator for output tensor");
  }

  void start() override {
    const std::string device = device_.get();

    // 先用最小可驗證 pipeline
    std::string pipeline =
        "v4l2src device=" + device + " ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink sync=false drop=true max-buffers=1";

    HOLOSCAN_LOG_INFO("Opening GStreamer pipeline: {}", pipeline);

    cap_.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap_.isOpened()) {
      throw std::runtime_error("Failed to open QCAP device: " + device);
    }

    HOLOSCAN_LOG_INFO("MyQCAPSourceOp started with device: {}", device);
  }

  void stop() override {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

  void compute(holoscan::InputContext&,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      HOLOSCAN_LOG_WARN("Empty frame from source");
      return;
    }

    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    auto entity = nvidia::gxf::Entity::New(context.context());
    if (!entity) {
      throw std::runtime_error("Failed to create GXF entity");
    }

    auto tensor = entity.value().add<nvidia::gxf::Tensor>("video");
    if (!tensor) {
      throw std::runtime_error("Failed to add tensor 'video'");
    }

    auto gxf_allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
            context.context(), allocator_->gxf_cid());
    if (!gxf_allocator) {
      throw std::runtime_error("Failed to create GXF allocator handle");
    }

    nvidia::gxf::Shape shape({rgb_frame.rows, rgb_frame.cols, rgb_frame.channels()});

    auto result = tensor.value()->reshape<uint8_t>(
        shape,
        nvidia::gxf::MemoryStorageType::kHost,
        gxf_allocator.value());

    if (!result) {
      throw std::runtime_error("Failed to reshape tensor");
    }

    std::memcpy(
        tensor.value()->pointer(),
        rgb_frame.data,
        rgb_frame.total() * rgb_frame.elemSize());

    auto out_entity = holoscan::gxf::Entity(std::move(entity.value()));
    op_output.emit(out_entity, "output");
  }

 private:
  holoscan::Parameter<std::string> device_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  cv::VideoCapture cap_;
};

class iSurgicalApp : public holoscan::Application {
 public:
  void compose() override {
    constexpr int32_t kWidth = 3840;
    constexpr int32_t kHeight = 1080;
    constexpr int32_t kChannels = 3;

    auto cuda_stream_pool = make_resource<holoscan::CudaStreamPool>(
        "cuda_stream_pool", 0, 0, 0, 1, 5);

    auto source_allocator =
        make_resource<holoscan::UnboundedAllocator>("source_allocator");

    auto source_op = make_operator<MyQCAPSourceOp>(
        "qcap_source",
        holoscan::Arg("device") = std::string("/dev/video0"),
        holoscan::Arg("allocator") = source_allocator);

    const size_t src_fc_block_size =
        static_cast<size_t>(kWidth) *
        static_cast<size_t>(kHeight) *
        6;

    auto src_fc_pool = make_resource<holoscan::BlockMemoryPool>(
        "src_fc_pool",
        1,           // storage_type: device
        src_fc_block_size,
        3);          // 官方文件建議 1~3 blocks，resize + host input 時通常至少要多留。:contentReference[oaicite:4]{index=4}

    auto format_converter = make_operator<holoscan::ops::FormatConverterOp>(
        "src_format_converter",
        holoscan::Arg("in_tensor_name") = std::string("video"),
        holoscan::Arg("out_tensor_name") = std::string("video"),
        holoscan::Arg("out_dtype") = std::string("rgb888"),
        holoscan::Arg("resize_width") = 3840,
        holoscan::Arg("resize_height") = 1080,
        holoscan::Arg("pool") = src_fc_pool,
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool);

    auto visualizer = make_operator<holoscan::ops::HolovizOp>(
        "holoviz",
        holoscan::Arg("width") = static_cast<int>(kWidth),
        holoscan::Arg("height") = static_cast<int>(kHeight),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool);

    add_flow(source_op, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, visualizer, {{"tensor", "receivers"}});

    HOLOSCAN_LOG_INFO("iSurgical application composed.");
  }
};

int main(int argc, char* argv[]) {
  iSurgicalApp app;
  app.run();
  return 0;
}