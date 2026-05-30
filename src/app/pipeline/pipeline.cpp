#include "pipeline.hpp"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>

#include "./operators/yuan_capture_source/yuan_capture_source.hpp"
#include "./operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.hpp"


void InferencePipeline::ConfigureSourceOperator() {
    auto source_allocator = make_resource<holoscan::UnboundedAllocator>("source_allocator");

    source_op_ = make_operator<YuanCaptureSourceOp>(
        "YuanCaptureSource",
        holoscan::Arg("device") = config_["Device"].as<std::string>(),
        holoscan::Arg("allocator") = source_allocator
    );
}

void InferencePipeline::ConfigurePreProcessorOperator() {
    auto src_fc_block_size = config_["Width"].as<uint32_t>() * config_["Height"].as<uint32_t>() * 3 * sizeof(float);

    pre_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "InferFormatConverterOp",
        holoscan::Arg("out_dtype") = config_["InferenceFormat"].as<std::string>(),
        holoscan::Arg("in_tensor_name") = std::string("video"),
        holoscan::Arg("out_tensor_name") = std::string("raw_video"),
        holoscan::Arg("resize_height") = config_["InferenceResizeHeight"].as<int>(),
        holoscan::Arg("resize_width") = config_["InferenceResizeWidth"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureInferenceOperator() {
    holoscan::ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("segmentation inference", config_["ModelPath"].as<std::string>());

    holoscan::ops::InferenceOp::DataVecMap pre_processor_map;
    pre_processor_map.insert("segmentation inference", std::vector<std::string>{"raw_video"});

    holoscan::ops::InferenceOp::DataVecMap inference_map;
    inference_map.insert("segmentation inference", std::vector<std::string>{"outputs", "proto"});

    auto device_allocator = make_resource<holoscan::UnboundedAllocator>("device_allocator");

    inference_op_ = make_operator<holoscan::ops::InferenceOp>(
        "inference",
        holoscan::Arg("model_path_map", model_path_map),
        holoscan::Arg("pre_processor_map", pre_processor_map),
        holoscan::Arg("inference_map", inference_map),
        holoscan::Arg("backend", std::string("trt")),
        holoscan::Arg("enable_fp16", true),
        holoscan::Arg("allocator") = device_allocator,
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_
    );
}

void InferencePipeline::ConfigurePostProcessorOperator() {
    auto src_fc_block_size = config_["Width"].as<uint32_t>() * config_["Height"].as<uint32_t>() * 4 * sizeof(float);

    post_processor_op_ = make_operator<YoloSegPostprocesserGPUOp>(
        "YoloSegPostprocesserGPUOp",
        holoscan::Arg("ScoreThredshold") = config_["ScoreThreshold"].as<float>(),
        holoscan::Arg("NumClass") = config_["NumClass"].as<uint32_t>(),
        holoscan::Arg("MemoryAllocator") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, config_["NumBlocks"].as<uint32_t>()));
}

void InferencePipeline::ConfigureVisualizerOperator() {
    visualizer_op_ = make_operator<holoscan::ops::HolovizOp>(
        "HoloViz",
        holoscan::Arg("width") = config_["Width"].as<uint32_t>(),
        holoscan::Arg("height") = config_["Height"].as<uint32_t>(),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_
    );
}

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
    ConfigureSourceOperator();
    ConfigurePreProcessorOperator();
    ConfigureInferenceOperator();
    ConfigurePostProcessorOperator();
    ConfigureVisualizerOperator();

    add_flow(source_op_, pre_processor_op_, {{"output", "source_video"}});
    add_flow(source_op_, visualizer_op_, {{"output", "receivers"}});
    add_flow(pre_processor_op_, inference_op_, {{"tensor", "receivers"}});
    add_flow(inference_op_, post_processor_op_, {{"transmitter", "in"}});
    add_flow(post_processor_op_, visualizer_op_, {{"out", "receivers"}});

    HOLOSCAN_LOG_INFO("Inference Pipeline Composed.");
}
