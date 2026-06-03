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
#include "./operators/periodic_counter/periodic_counter.hpp"
#include "./operators/async_recorder/async_recorder.hpp"
#include "./operators/surgical_phase/surgical_phase.hpp"

void InferencePipeline::ConfigureSourceOperator() {
    auto source_allocator = make_resource<holoscan::UnboundedAllocator>("source_allocator");

    source_op_ = make_operator<YuanCaptureSourceOp>(
        "YuanCaptureSource",
        holoscan::Arg("device") = config_["Device"].as<std::string>(),
        holoscan::Arg("width") = config_["Width"].as<int>(),
        holoscan::Arg("height") = config_["Height"].as<int>(),
        holoscan::Arg("allocator") = source_allocator
    );
}

void InferencePipeline::ConfigureInferencePreProcessorOperator() {
    auto src_fc_block_size = config_["Width"].as<uint32_t>() * config_["Height"].as<uint32_t>() * 3 * sizeof(float);

    pre_infer_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "InferFormatConverterOp",
        holoscan::Arg("out_dtype") = config_["InferenceFormat"].as<std::string>(),
        holoscan::Arg("in_tensor_name") = std::string("video"),
        holoscan::Arg("out_tensor_name") = std::string("raw_video"),
        holoscan::Arg("resize_height") = config_["InferenceResizeHeight"].as<int>(),
        holoscan::Arg("resize_width") = config_["InferenceResizeWidth"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureRawPreProcessorOperator() {
    auto raw_fc_block_size = config_["Sys_Width"].as<uint32_t>() * config_["Sys_Height"].as<uint32_t>() * 3 * sizeof(uint8_t);

    pre_raw_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "RawFormatConverterOp",
        holoscan::Arg("out_dtype") = std::string("uint8"),
        holoscan::Arg("in_tensor_name") = std::string("video"),
        holoscan::Arg("out_tensor_name") = std::string("tensor"),
        holoscan::Arg("resize_height") = config_["Sys_Height"].as<int>(),
        holoscan::Arg("resize_width") = config_["Sys_Width"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, raw_fc_block_size, config_["NumBlocks"].as<uint32_t>()),
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
        holoscan::Arg("width") = config_["Sys_Width"].as<uint32_t>(),
        holoscan::Arg("height") = config_["Sys_Height"].as<uint32_t>(),
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

void InferencePipeline::ConfigurePeriodicCounterOperator() {
    auto async_condition = make_condition<holoscan::AsynchronousCondition>("async_condition");

    periodic_counter_op_ = make_operator<PeriodicCounterOp>(
        "PeriodicCounter",
        async_condition,
        holoscan::Arg("interval_ms") = int64_t(1000)
    );
}

void InferencePipeline::ConfigureRecordePreProcessorOperator() {
    auto record_fc_block_size = config_["Sys_Width"].as<uint32_t>() * config_["Sys_Height"].as<uint32_t>() * 3 * sizeof(uint8_t);

    pre_record_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "RecordFormatConverterOp",
        holoscan::Arg("out_dtype") = std::string("uint8"),
        holoscan::Arg("in_tensor_name") = std::string("video"),
        holoscan::Arg("out_tensor_name") = std::string("tensor"),
        holoscan::Arg("resize_height") = config_["Sys_Height"].as<int>(),
        holoscan::Arg("resize_width") = config_["Sys_Width"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, record_fc_block_size, config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureRecorderOperator() {
    recorder_op_ = make_operator<AsyncRecorderOp>(
        "AsyncRecorder",
        holoscan::Arg("recording_path") = config_["RecordingPath"].as<std::string>(),
        holoscan::Arg("source_width") = static_cast<int>(config_["Sys_Width"].as<uint32_t>()),
        holoscan::Arg("source_height") = static_cast<int>(config_["Sys_Height"].as<uint32_t>()),
        holoscan::Arg("overlay_width") = static_cast<int>(config_["Sys_Width"].as<uint32_t>()),
        holoscan::Arg("overlay_height") = static_cast<int>(config_["Sys_Height"].as<uint32_t>()),
        holoscan::Arg("fps") = 30
    );
}

void InferencePipeline::ConfigureSurgicalPhaseOperator() {
   std::string phase_model = config_["PhaseModelPath"].as<std::string>();
   if (config_["PhaseModelPath"]) {
       phase_model = config_["PhaseModelPath"].as<std::string>();
   }


   surgical_phase_op_ = make_operator<SurgicalPhaseOp>(
       "SurgicalPhase",
       holoscan::Arg("model_path") = phase_model,
       holoscan::Arg("interval_ms") = int64_t(10000),
       holoscan::Arg("seq_len") = 5
   );
}


void InferencePipeline::compose() {
    ConfigureSourceOperator();
    ConfigureRawPreProcessorOperator();
    ConfigureInferencePreProcessorOperator();
    ConfigureRecordePreProcessorOperator();
    ConfigureInferenceOperator();
    ConfigurePostProcessorOperator();
    ConfigureVisualizerOperator();
    ConfigurePeriodicCounterOperator();
    ConfigureRecorderOperator();
    ConfigureSurgicalPhaseOperator();

    add_flow(source_op_, pre_infer_processor_op_, {{"output", "source_video"}});
    add_flow(source_op_, pre_raw_processor_op_, {{"output", "source_video"}});

    add_flow(pre_infer_processor_op_, inference_op_, {{"tensor", "receivers"}});
    add_flow(inference_op_, post_processor_op_, {{"transmitter", "in"}});
    add_flow(pre_raw_processor_op_, visualizer_op_, {{"tensor", "receivers"}});
    add_flow(post_processor_op_, visualizer_op_, {{"out", "receivers"}});

    // Recorder receives raw source and segmentation overlay (same frame)
    add_flow(source_op_, pre_record_processor_op_, {{"output", "source_video"}});
    add_flow(pre_record_processor_op_, recorder_op_, {{"tensor", "raw_input"}});
    add_flow(post_processor_op_, recorder_op_, {{"out", "seg_input"}});

    // Surgical phase recognition receives raw frames from source
    //add_flow(source_op_, surgical_phase_op_, {{"output", "frame_input"}});

    add_operator(periodic_counter_op_);

    HOLOSCAN_LOG_INFO("Inference Pipeline Composed.");
}