#include "async_recorder.hpp"

#include <filesystem>
#include <future>

void AsyncRecorderOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("raw_input")
      .condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::gxf::Entity>("seg_input");

  spec.param(recording_path_, "recording_path", "Recording Path",
             "Directory to save recordings",
             std::string("/workspace/iSurgical/data/outputs"));
  spec.param(source_width_, "source_width", "Source Width",
             "Raw capture frame width", 1920);
  spec.param(source_height_, "source_height", "Source Height",
             "Raw capture frame height", 1080);
  spec.param(overlay_width_, "overlay_width", "Overlay Width",
             "Segmentation overlay width", 1920);
  spec.param(overlay_height_, "overlay_height", "Overlay Height",
             "Segmentation overlay height", 1080);
  spec.param(fps_, "fps", "FPS", "Recording framerate", 30);
}

void AsyncRecorderOp::start() {
  std::string path = recording_path_.get();
  std::filesystem::create_directories(path);

  int W = overlay_width_.get();
  int H = overlay_height_.get();
  int f = fps_.get();

  std::string raw_file = path + "/raw.mp4";
  std::string seg_file = path + "/segmentation.mp4";

  raw_writer_.open(raw_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), f, cv::Size(W, H));
  seg_writer_.open(seg_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), f, cv::Size(W, H));

  if (!raw_writer_.isOpened() || !seg_writer_.isOpened()) {
    throw std::runtime_error("AsyncRecorderOp: failed to open video writers at " + path);
  }

  is_running_ = true;
  writer_thread_ = std::thread(&AsyncRecorderOp::writerLoop, this);

  HOLOSCAN_LOG_INFO("AsyncRecorderOp: recording to {}", path);
}

void AsyncRecorderOp::stop() {
  is_running_ = false;
  queue_cv_.notify_one();

  if (writer_thread_.joinable()) {
    std::future<void> fut = std::async(std::launch::async, [this]() {
      writer_thread_.join();
    });
    if (fut.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
      HOLOSCAN_LOG_WARN("AsyncRecorderOp: writer thread did not exit in time, detaching");
      writer_thread_.detach();
    }
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::queue<RecordFrame>().swap(frame_queue_);
  }

  raw_writer_.release();
  seg_writer_.release();

  HOLOSCAN_LOG_INFO("AsyncRecorderOp: stopped, wrote {} frames", frames_written_.load());
}

void AsyncRecorderOp::compute(holoscan::InputContext& input,
                              holoscan::OutputContext&,
                              holoscan::ExecutionContext&) {
  if (!is_running_) return;

  int srcW = source_width_.get();
  int srcH = source_height_.get();
  int ovW = overlay_width_.get();
  int ovH = overlay_height_.get();

  // Drain all raw frames (kNone port), keep latest
  auto raw_msg = input.receive<holoscan::TensorMap>("raw_input");
  while (raw_msg) {
    auto& raw_tmap = raw_msg.value();
    auto it = raw_tmap.find("tensor");
    if (it == raw_tmap.end()) it = raw_tmap.find("video");
    if (it != raw_tmap.end()) {
      auto raw_tensor = it->second;
      cv::Mat raw_rgb(srcH, srcW, CV_8UC3);
      size_t raw_bytes = static_cast<size_t>(srcW) * srcH * 3;
      cudaMemcpy(raw_rgb.data, raw_tensor->data(), raw_bytes, cudaMemcpyDeviceToHost);

      cv::Mat raw_resized;
      if (srcW != ovW || srcH != ovH) {
        cv::resize(raw_rgb, raw_resized, cv::Size(ovW, ovH));
      } else {
        raw_resized = raw_rgb;
      }

      cv::Mat raw_bgr;
      cv::cvtColor(raw_resized, raw_bgr, cv::COLOR_RGB2BGR);

      std::lock_guard<std::mutex> lock(raw_buffer_mutex_);
      raw_buffer_ = std::move(raw_bgr);
      has_raw_ = true;
    }
    raw_msg = input.receive<holoscan::TensorMap>("raw_input");
  }

  // seg_input triggers compute (default condition)
  auto seg_msg = input.receive<holoscan::TensorMap>("seg_input");
  if (!seg_msg) return;

  cv::Mat current_raw;
  {
    std::lock_guard<std::mutex> lock(raw_buffer_mutex_);
    if (!has_raw_) return;
    current_raw = raw_buffer_.clone();
  }

  // Process segmentation overlay
  auto& seg_tmap = seg_msg.value();
  if (seg_tmap.find("OverlayTensor") == seg_tmap.end()) return;

  auto seg_tensor = seg_tmap["OverlayTensor"];
  size_t seg_bytes = static_cast<size_t>(ovW) * ovH * 4 * sizeof(float);
  cv::Mat seg_rgba_f(ovH, ovW, CV_32FC4);
  cudaMemcpy(seg_rgba_f.data, seg_tensor->data(), seg_bytes, cudaMemcpyDeviceToHost);

  // Composite: blend segmentation overlay on top of raw frame
  cv::Mat seg_bgr(ovH, ovW, CV_8UC3);
  cv::Mat raw_float;
  current_raw.convertTo(raw_float, CV_32FC3, 1.0 / 255.0);

  for (int y = 0; y < ovH; ++y) {
    const cv::Vec4f* overlay_row = seg_rgba_f.ptr<cv::Vec4f>(y);
    const cv::Vec3f* base_row = raw_float.ptr<cv::Vec3f>(y);
    cv::Vec3b* out_row = seg_bgr.ptr<cv::Vec3b>(y);

    for (int x = 0; x < ovW; ++x) {
      float a = overlay_row[x][3];
      float r = overlay_row[x][0];
      float g = overlay_row[x][1];
      float b_ov = overlay_row[x][2];
      float base_b = base_row[x][0];
      float base_g = base_row[x][1];
      float base_r = base_row[x][2];

      float out_r = base_r * (1.0f - a) + r * a;
      float out_g = base_g * (1.0f - a) + g * a;
      float out_b = base_b * (1.0f - a) + b_ov * a;

      out_row[x] = cv::Vec3b(
          static_cast<uint8_t>(std::clamp(out_b * 255.0f, 0.0f, 255.0f)),
          static_cast<uint8_t>(std::clamp(out_g * 255.0f, 0.0f, 255.0f)),
          static_cast<uint8_t>(std::clamp(out_r * 255.0f, 0.0f, 255.0f))
      );
    }
  }

  // Enqueue for async writing
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() < 30) {
      frame_queue_.push(RecordFrame{std::move(current_raw), std::move(seg_bgr)});
    }
  }
  queue_cv_.notify_one();
}

void AsyncRecorderOp::writerLoop() {
  while (true) {
    RecordFrame frame;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() {
        return !frame_queue_.empty() || !is_running_;
      });

      if (!is_running_) break;

      frame = std::move(frame_queue_.front());
      frame_queue_.pop();
    }

    raw_writer_.write(frame.raw);
    seg_writer_.write(frame.segmentation);
    ++frames_written_;
  }
}
