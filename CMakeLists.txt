cmake_minimum_required(VERSION 3.20.1)
project(iSurgical LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "87")
set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")

# Set Package Manager
find_package(holoscan 3.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Load the PkgConfig module tool
find_package(PkgConfig REQUIRED)

# Query the system for yaml-cpp paths automatically
pkg_check_modules(YAML_CPP REQUIRED IMPORTED_TARGET yaml-cpp)

# gRPC and Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
  pkg_check_modules(GRPC REQUIRED IMPORTED_TARGET grpc++ grpc)
endif()

# Generate protobuf/gRPC sources
set(PROTO_SRC_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

set(PROTO_FILE "${CMAKE_SOURCE_DIR}/protos/surgical_phase.proto")

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

add_custom_command(
    OUTPUT
        "${PROTO_SRC_DIR}/surgical_phase.pb.cc"
        "${PROTO_SRC_DIR}/surgical_phase.pb.h"
        "${PROTO_SRC_DIR}/surgical_phase.grpc.pb.cc"
        "${PROTO_SRC_DIR}/surgical_phase.grpc.pb.h"
    COMMAND protobuf::protoc
        --proto_path="${CMAKE_SOURCE_DIR}/protos"
        --cpp_out="${PROTO_SRC_DIR}"
        --grpc_out="${PROTO_SRC_DIR}"
        --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
        "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating gRPC/protobuf sources for surgical_phase.proto"
)

add_library(surgical_phase_proto STATIC
    "${PROTO_SRC_DIR}/surgical_phase.pb.cc"
    "${PROTO_SRC_DIR}/surgical_phase.grpc.pb.cc"
)
target_include_directories(surgical_phase_proto PUBLIC "${PROTO_SRC_DIR}")
target_link_libraries(surgical_phase_proto PUBLIC protobuf::libprotobuf)
if(gRPC_FOUND)
    target_link_libraries(surgical_phase_proto PUBLIC gRPC::grpc++)
else()
    target_link_libraries(surgical_phase_proto PUBLIC PkgConfig::GRPC)
endif()

add_executable(iSurgical
    src/app/main.cpp
    src/app/pipeline/pipeline.cpp
    src/operators/yuan_capture_source/yuan_capture_source.cpp
    src/operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.cpp
    src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess.cu

    src/operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.cpp
    src/operators/lc_seg_postprocessor/lc_cuda_postprocess.cu

    src/operators/periodic_counter/periodic_counter.cpp
    src/operators/async_recorder/async_recorder.cpp
    src/operators/lc_surgical_phase/lc_surgical_phase.cpp
    src/operators/phase_frame_sampler/phase_frame_sampler.cpp
    src/operators/phase_grpc_client/phase_grpc_client.cpp
)

target_include_directories(iSurgical
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/app
        ${CMAKE_SOURCE_DIR}/src/app/pipeline
        ${CMAKE_SOURCE_DIR}/src/operators
        ${CMAKE_SOURCE_DIR}/src/operators/yuan_capture_source
        ${CMAKE_SOURCE_DIR}/src/operators/yolo_seg_postprocessor_gpu
        ${CMAKE_SOURCE_DIR}/src/operators/lc_seg_postprocessor
        ${CMAKE_SOURCE_DIR}/src/operators/periodic_counter
        ${CMAKE_SOURCE_DIR}/src/operators/async_recorder
        ${CMAKE_SOURCE_DIR}/src/operators/lc_surgical_phase
        ${CMAKE_SOURCE_DIR}/src/operators/phase_frame_sampler
        ${CMAKE_SOURCE_DIR}/src/operators/phase_grpc_client
        ${PROTO_SRC_DIR}
)

target_link_libraries(iSurgical
    PRIVATE
        holoscan::core
        holoscan::ops::format_converter
        holoscan::ops::holoviz
        holoscan::ops::inference
        PkgConfig::YAML_CPP
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
        nvonnxparser
        stdc++fs
        surgical_phase_proto
)

target_link_options(iSurgical PRIVATE "-Wl,--allow-shlib-undefined")