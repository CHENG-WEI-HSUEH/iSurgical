#include "pipeline.hpp"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include "yuan_capture_source.hpp"

InferencePipeline::InferencePipeline(const YAML::Node& config)
    : config_(config)
{
    cuda_stream_pool_ = make_resource<holoscan::CudaStreamPool>(
        "cuda_stream_pool",
        holoscan::Arg("stream_flags", int64_t(0)),
        holoscan::Arg("stream_priority", int64_t(0)),
        holoscan::Arg("reserved_size", int64_t(1)),
        holoscan::Arg("max_size", int64_t(5))
    );
}

void InferencePipeline::compose() {
    auto source_allocator = make_resource<holoscan::UnboundedAllocator>("source_allocator");

    auto source_op = make_operator<YuanCaptureSourceOp>(
        "YuanCaptureSource",
        holoscan::Arg("device") = config_["Device"].as<std::string>(),
        holoscan::Arg("allocator") = source_allocator
    );

    uint32_t win_width = config_["Width"].as<uint32_t>();
    uint32_t win_height = config_["Height"].as<uint32_t>();

    auto visualizer = make_operator<holoscan::ops::HolovizOp>(
        "HoloViz",
        holoscan::Arg("width", win_width),
        holoscan::Arg("height", win_height),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_
    );

    add_flow(source_op, visualizer, {{"output", "receivers"}});
}
