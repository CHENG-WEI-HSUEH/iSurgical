#include "pipeline.hpp"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>


#include "./operators/yuan_capture_source/yuan_capture_source.hpp"
#include "./operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.hpp"
#include "./operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.hpp"

#include "./operators/async_recorder/async_recorder.hpp"


#include <stdexcept>

namespace {
// Name of the tensor the source operator puts on its output message. VideoStreamReplayerOp
// replays whatever convert_video_to_gxf_entities.py wrote, and that tool leaves the tensor
// unnamed; the live YuanCaptureSourceOp called it "video". An empty name makes
// FormatConverterOp fall back to the single tensor in the message.
constexpr const char* kSourceTensorName = "";
}  // namespace

void InferencePipeline::ConfigureSourceOperator() {


    source_op_ = make_operator<holoscan::ops::VideoStreamReplayerOp>(
        "VideoStreamReplayerOp",
        holoscan::Arg("directory") = config_["ReplayPath"].as<std::string>(),
        holoscan::Arg("basename") = config_["ReplayBasename"].as<std::string>(),
        holoscan::Arg("repeat") = config_["ReplayRepeat"].as<bool>()
    );
}

void InferencePipeline::ConfigureInferencePreProcessorOperator() {
    auto src_fc_block_size = config_["Width"].as<uint32_t>() * config_["Height"].as<uint32_t>() * 3 * sizeof(float);

    pre_infer_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "InferFormatConverterOp",
        holoscan::Arg("out_dtype") = procedure_config_["InferenceFormat"].as<std::string>(),
        holoscan::Arg("in_tensor_name") = std::string(kSourceTensorName),
        holoscan::Arg("out_tensor_name") = std::string("raw_video"),
        holoscan::Arg("resize_height") = procedure_config_["InferenceResizeHeight"].as<int>(),
        holoscan::Arg("resize_width") = procedure_config_["InferenceResizeWidth"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, procedure_config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureRawPreProcessorOperator() {
    auto raw_fc_block_size = procedure_config_["Sys_Width"].as<uint32_t>() * procedure_config_["Sys_Height"].as<uint32_t>() * 3 * sizeof(uint8_t);

    pre_raw_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "RawFormatConverterOp",
        holoscan::Arg("out_dtype") = std::string("uint8"),
        holoscan::Arg("in_tensor_name") = std::string(kSourceTensorName),
        holoscan::Arg("out_tensor_name") = std::string("tensor"),
        holoscan::Arg("resize_height") = procedure_config_["Sys_Height"].as<int>(),
        holoscan::Arg("resize_width") = procedure_config_["Sys_Width"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, raw_fc_block_size, procedure_config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureInferenceOperator() {
    holoscan::ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("segmentation inference", procedure_config_["ModelPath"].as<std::string>());

    holoscan::ops::InferenceOp::DataVecMap pre_processor_map;
    pre_processor_map.insert("segmentation inference", std::vector<std::string>{"raw_video"});

    // The inference_map names the model's outputs, in declaration order, with the tensor
    // names the downstream postprocessor consumes. Each procedure's model has a different
    // output signature, so this must match the specific ONNX:
    //   - Laparoscopic Cholecystectomy: MultiHeadGCNFormer -> head_0..head_3 + correlation_scores
    //   - Urology: YOLO-seg            -> outputs (detections) + proto (mask prototypes)
    const std::string procedure = config_["Procedure"].as<std::string>();
    holoscan::ops::InferenceOp::DataVecMap inference_map;
    if (procedure == "Laparoscopic Cholecystectomy") {
        inference_map.insert("segmentation inference",
            std::vector<std::string>{"head_out_0", "head_out_1", "head_out_2", "head_out_3", "corr_scores"});
    } else {
        inference_map.insert("segmentation inference", std::vector<std::string>{"outputs", "proto"});
    }

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
    const std::string procedure = config_["Procedure"].as<std::string>();
    if (procedure == "Laparoscopic Cholecystectomy") {
        post_processor_op_ = make_operator<LcSegPostprocesserGPUOp>(
            "LcSegPostprocesserGPUOp",
            holoscan::Arg("colour_labels") = procedure_config_,
            holoscan::Arg("MemoryAllocator") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, procedure_config_["NumBlocks"].as<uint32_t>()));
    } else if (procedure == "Urology") {
        post_processor_op_ = make_operator<YoloSegPostprocesserGPUOp>(
            "YoloSegPostprocesserGPUOp",
            holoscan::Arg("colour_labels") = procedure_config_,
            holoscan::Arg("MemoryAllocator") = make_resource<holoscan::BlockMemoryPool>("pool", 1, src_fc_block_size, procedure_config_["NumBlocks"].as<uint32_t>()));
    } else {
        HOLOSCAN_LOG_ERROR("Unsupported Procedure: {}", procedure);
        throw std::runtime_error("Unsupported Procedure: " + procedure);
    }
}

void InferencePipeline::ConfigureVisualizerOperator() {
    visualizer_op_ = make_operator<holoscan::ops::HolovizOp>(
        "HoloViz",
        holoscan::Arg("fullscreen") = true,
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_
    );
}

InferencePipeline::InferencePipeline(const YAML::Node& config) : config_(config)
{
    std::string procedure = config_["Procedure"].as<std::string>();
    is_recorded_ = config_["IsRecorded"].as<bool>();

    procedure_config_ = config_["ProcedureConfig"][procedure];
    if (!procedure_config_ || procedure_config_.IsNull()) {
        throw std::runtime_error("Unknown Procedure in config: " + procedure);
    }

    cuda_stream_pool_ = make_resource<holoscan::CudaStreamPool>(
        "cuda_stream_pool",
        holoscan::Arg("stream_flags", int64_t(0)),
        holoscan::Arg("stream_priority", int64_t(0)),
        holoscan::Arg("reserved_size", int64_t(1)),
        holoscan::Arg("max_size", int64_t(5))
    );


}


void InferencePipeline::ConfigureRecordePreProcessorOperator() {
    auto record_fc_block_size = procedure_config_["Sys_Width"].as<uint32_t>() * procedure_config_["Sys_Height"].as<uint32_t>() * 3 * sizeof(uint8_t);

    pre_record_processor_op_ = make_operator<holoscan::ops::FormatConverterOp>(
        "RecordFormatConverterOp",
        holoscan::Arg("out_dtype") = std::string("uint8"),
        holoscan::Arg("in_tensor_name") = std::string(kSourceTensorName),
        holoscan::Arg("out_tensor_name") = std::string("tensor"),
        holoscan::Arg("resize_height") = procedure_config_["Sys_Height"].as<int>(),
        holoscan::Arg("resize_width") = procedure_config_["Sys_Width"].as<int>(),
        holoscan::Arg("pool") = make_resource<holoscan::BlockMemoryPool>("pool", 1, record_fc_block_size, procedure_config_["NumBlocks"].as<uint32_t>()),
        holoscan::Arg("cuda_stream_pool") = cuda_stream_pool_);
}

void InferencePipeline::ConfigureRecorderOperator() {
    recorder_op_ = make_operator<AsyncRecorderOp>(
        "AsyncRecorder",
        holoscan::Arg("recording_path") = config_["RecordingPath"].as<std::string>(),
        holoscan::Arg("source_width") = static_cast<int>(procedure_config_["Sys_Width"].as<uint32_t>()),
        holoscan::Arg("source_height") = static_cast<int>(procedure_config_["Sys_Height"].as<uint32_t>()),
        holoscan::Arg("overlay_width") = static_cast<int>(procedure_config_["Sys_Width"].as<uint32_t>()),
        holoscan::Arg("overlay_height") = static_cast<int>(procedure_config_["Sys_Height"].as<uint32_t>()),
        holoscan::Arg("fps") = 30
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
    ConfigureRecorderOperator();


    add_flow(source_op_, pre_infer_processor_op_, {{"output", "source_video"}});
    add_flow(source_op_, pre_raw_processor_op_, {{"output", "source_video"}});

    add_flow(pre_infer_processor_op_, inference_op_, {{"tensor", "receivers"}});
    add_flow(inference_op_, post_processor_op_, {{"transmitter", "in"}});
    add_flow(pre_raw_processor_op_, visualizer_op_, {{"tensor", "receivers"}});
    add_flow(post_processor_op_, visualizer_op_, {{"out", "receivers"}});


    HOLOSCAN_LOG_INFO("Inference Pipeline Composed.");
}