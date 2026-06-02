#include "yuan_capture_source.hpp"

#include <holoscan/holoscan.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>


void YuanCaptureSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<nvidia::gxf::Entity>("output");

  spec.param(this->device_, "device", "Device", "Video device", std::string("/dev/video0"));
  spec.param(this->width_, "width", "Width", "Capture width", 1920);
  spec.param(this->height_, "height", "Height", "Capture height", 1080);
  spec.param(this->allocator_, "allocator", "Allocator", "CUDA Allocator for GPU tensor output");
}

void YuanCaptureSourceOp::start() {
  const std::string device = device_.get();
  int w = width_.get();
  int h = height_.get();

  std::string pipeline;
  if (device == "test") {
    pipeline =
        "videotestsrc pattern=smpte ! "
        "video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=2 sync=false";
  } else {
    pipeline =
        "v4l2src device=" + device + " io-mode=mmap do-timestamp=true ! "
        "video/x-raw,format=YUY2,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=30/1 ! "
        "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "videoconvert n-threads=2 ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=2 sync=false";
  }

  this->cap_.open(pipeline, cv::CAP_GSTREAMER);
  if (!this->cap_.isOpened()) {
    throw std::runtime_error("Failed to open capture source via GStreamer: " + device);
  }

  cudaStreamCreate(&this->cuda_stream_);

  this->is_running_ = true;
  this->capture_thread_ = std::thread(&YuanCaptureSourceOp::framePumper, this);

  HOLOSCAN_LOG_INFO("YuanCaptureSourceOp started for source: {}", device);
}

void YuanCaptureSourceOp::stop() {
  this->is_running_ = false;
  if (this->capture_thread_.joinable()) {
    this->capture_thread_.join();
  }
  if (this->cap_.isOpened()) {
    this->cap_.release();
  }
  if (this->cuda_stream_) {
    cudaStreamDestroy(this->cuda_stream_);
    this->cuda_stream_ = nullptr;
  }
}

void YuanCaptureSourceOp::compute(holoscan::InputContext&,
              holoscan::OutputContext& op_output,
              holoscan::ExecutionContext& context) {

  cv::Mat frame;
  
  // Non-blocking/Quick-locking fetch from the asynchronous capture thread
  {
    std::lock_guard<std::mutex> lock(this->frame_mutex_);
    if (this->frame_queue_.empty()) {
      // High-speed mode: skip compute cycle if no fresh card frames have arrived yet
      return; 
    }
    frame = std::move(this->frame_queue_.front());
    this->frame_queue_.pop();
  }

  cv::Mat rgb_frame;
  cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

  auto entity = nvidia::gxf::Entity::New(context.context());
  if (!entity) throw std::runtime_error("Failed to create GXF entity");

  auto tensor = entity.value().add<nvidia::gxf::Tensor>("video");
  if (!tensor) throw std::runtime_error("Failed to add tensor 'video'");

  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_->gxf_cid());
  if (!gxf_allocator) throw std::runtime_error("Failed to create GXF allocator handle");

  nvidia::gxf::Shape shape{rgb_frame.rows, rgb_frame.cols, rgb_frame.channels()};

  // OPTIMIZATION 2: Reshape directly to Device (GPU) memory
  auto result = tensor.value()->reshape<uint8_t>(
      shape,
      nvidia::gxf::MemoryStorageType::kDevice, // Target GPU directly
      gxf_allocator.value());
  if (!result) throw std::runtime_error("Failed to reshape tensor to Device memory");

  cudaStream_t stream = this->cuda_stream_;

  // OPTIMIZATION 3: Asynchronous Hardware Memory Copy (CPU Host -> GPU Device)
  size_t data_size = rgb_frame.total() * rgb_frame.elemSize();
  cudaError_t cuda_status = cudaMemcpyAsync(
      tensor.value()->pointer(),
      rgb_frame.data,
      data_size,
      cudaMemcpyHostToDevice,
      stream
  );

  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(cuda_status));
  }

  // Emit the native Holoscan GXF wrapper
  auto out_entity = holoscan::gxf::Entity(std::move(entity.value()));
  op_output.emit(out_entity, "output");
}


void YuanCaptureSourceOp::framePumper() {
  while (this->is_running_) {
    cv::Mat local_frame;
    if (this->cap_.read(local_frame) && !local_frame.empty()) {
      std::lock_guard<std::mutex> lock(this->frame_mutex_);
      
      // Evict older frame if downstream compute is bottlenecked (keeps live preview real-time)
      if (this->frame_queue_.size() >= 2) {
        this->frame_queue_.pop();
      }
      this->frame_queue_.push(local_frame);
    }
  }
}