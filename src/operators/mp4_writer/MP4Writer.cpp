#include "MP4Writer.hpp"


void Mp4WriterOp::setup(holoscan::OperatorSpec& spec) 
{
    spec.input<holoscan::TensorMap>("in_frame");

    spec.param(output_path_, "output_path",
              "Output Path",
              "MP4 file path",
              std::string("output.mp4"));

    spec.param(width_, "width");
    spec.param(height_, "height");

    spec.param(fps_, "fps",
              "FPS",
              "Frames per second",
              30);
}

void Mp4WriterOp::compute(holoscan::InputContext& input,
                          holoscan::OutputContext&,
                          holoscan::ExecutionContext&)
{
    // ============================================================
    // Step 1 — Receive TensorMap safely
    // ============================================================
    auto maybe_msg = input.receive<holoscan::TensorMap>("in_frame");

    if (!maybe_msg) 
    {
        HOLOSCAN_LOG_ERROR("Mp4WriterOp: did not receive TensorMap on port 'in_frame'");
        return;
    }

    auto tmap = maybe_msg.value();

    // ============================================================
    // Step 2 — Extract correct tensor by name
    // Must match FormatConverter out_tensor_name
    // ============================================================
    const std::string kName = "video_rgb";

    if (tmap.find(kName) == tmap.end()) 
    {
        HOLOSCAN_LOG_ERROR("Mp4WriterOp: TensorMap missing key '{}'", kName);
        return;
    }

    auto frame = tmap[kName];  // std::shared_ptr<holoscan::Tensor>

    // ============================================================
    // Step 3 — Get video properties
    // ============================================================
    int W = width_.get();
    int H = height_.get();
    size_t bytes = static_cast<size_t>(W) * H * 3;  // RGB888

    // ============================================================
    // Step 4 — Initialize VideoWriter once
    // ============================================================
    if (!initialized_) 
    {

        std::string filename = output_path_.get();
        if (filename.find(".mp4") == std::string::npos)
            filename += ".mp4";

        writer_.open(
            filename,
            cv::VideoWriter::fourcc('a','v','c','1'),  // H264 codec
            fps_.get(),
            cv::Size(W, H)
        );

        if (!writer_.isOpened()) {
            HOLOSCAN_LOG_ERROR("Mp4WriterOp: Failed to open MP4 file: {}", filename);
            return;
        }

        HOLOSCAN_LOG_INFO("Mp4WriterOp: Recording started -> {}", filename);

        initialized_ = true;
    }

    // ============================================================
    // Step 5 — Allocate CPU frame buffer (RGB)
    // ============================================================
    cv::Mat cpu_rgb(H, W, CV_8UC3);

    void* src_ptr = frame->data();

    // ============================================================
    // Step 6 — Copy GPU → CPU (Assume Device Tensor)
    // ============================================================
    cudaError_t err = cudaMemcpy(
        cpu_rgb.data,
        src_ptr,
        bytes,
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Mp4WriterOp: cudaMemcpy failed: {}",
                           cudaGetErrorString(err));
        return;
    }

    // ============================================================
    // Step 7 — Fix OpenCV color order: RGB → BGR
    // ============================================================
    cv::Mat cpu_bgr;
    cv::cvtColor(cpu_rgb, cpu_bgr, cv::COLOR_RGB2BGR);

    // ============================================================
    // Step 8 — Write frame into MP4 immediately
    // ============================================================
    writer_.write(cpu_bgr);
}
