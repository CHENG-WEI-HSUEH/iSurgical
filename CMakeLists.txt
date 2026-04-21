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

add_executable(iSurgical 
./src/main.cpp)

target_include_directories(iSurgical
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(iSurgical
    PRIVATE
        holoscan::core
        holoscan::ops::format_converter
        holoscan::ops::holoviz
        ${OpenCV_LIBS}
)