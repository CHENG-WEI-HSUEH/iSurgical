cmake_minimum_required(VERSION 3.20.1)
project(iSurgical LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "all")
set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")

# Set Package Manager
find_package(holoscan 3.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Load the PkgConfig module tool
find_package(PkgConfig REQUIRED)

# Query the system for yaml-cpp paths automatically
pkg_check_modules(YAML_CPP REQUIRED IMPORTED_TARGET yaml-cpp)







add_executable(iSurgical
    src/app/main.cpp
    src/app/pipeline/pipeline.cpp
    src/operators/yuan_capture_source/yuan_capture_source.cpp
    src/operators/yolo_seg_postprocessor_gpu/yolo_seg_postprocessor_gpu.cpp
    src/operators/yolo_seg_postprocessor_gpu/CUDA_Postprocess.cu

    src/operators/lc_seg_postprocessor/lc_seg_postprocessor_gpu.cpp
    src/operators/lc_seg_postprocessor/lc_cuda_postprocess.cu

    src/operators/async_recorder/async_recorder.cpp
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
)

target_link_libraries(iSurgical
    PRIVATE
        holoscan::core
        holoscan::ops::format_converter
        holoscan::ops::holoviz
        holoscan::ops::inference
        holoscan::ops::video_stream_replayer
        PkgConfig::YAML_CPP
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
        nvonnxparser
        stdc++fs

)

target_link_options(iSurgical PRIVATE "-Wl,--allow-shlib-undefined")