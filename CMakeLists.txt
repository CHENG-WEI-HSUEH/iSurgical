cmake_minimum_required(VERSION 3.20)

project(iprovision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to onnxruntime directory")
set(CUDNN_LIB_DIR "" CACHE PATH "Path to a Pascal-capable cuDNN directory (its root or its lib/ folder)")

# Accept either the cuDNN root (with lib/ and include/) or its lib/ folder directly
set(CUDNN_LIBRARY_DIR "")
set(CUDNN_INCLUDE_DIR "")
if(CUDNN_LIB_DIR)
    if(EXISTS ${CUDNN_LIB_DIR}/lib)
        set(CUDNN_LIBRARY_DIR ${CUDNN_LIB_DIR}/lib)
        set(CUDNN_INCLUDE_DIR ${CUDNN_LIB_DIR}/include)
    else()
        set(CUDNN_LIBRARY_DIR ${CUDNN_LIB_DIR})
        get_filename_component(CUDNN_INCLUDE_DIR ${CUDNN_LIB_DIR}/../include ABSOLUTE)
    endif()
    if(NOT EXISTS ${CUDNN_LIBRARY_DIR}/libcudnn.so.9)
        message(WARNING "libcudnn.so.9 not found in ${CUDNN_LIBRARY_DIR}; the system cuDNN will be used")
    endif()
endif()


find_package(OpenCV)
find_package(yaml-cpp REQUIRED)

add_executable(iprovision
    src/main.cpp
    src/isurgical/isurgical.cpp
    src/cap/cap.cpp
    src/record/record.cpp
    src/preproc/preproc_seg.cpp
    src/preproc/preproc_rsd.cpp
    src/infer/infer_seg.cpp
    src/infer/infer_rsd.cpp
    src/postproc/postproc_seg.cpp
    src/postproc/postproc_rsd.cpp
)

target_link_libraries(iprovision PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    tbb
    cudart
    onnxruntime
    pthread
)

target_include_directories(iprovision PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

target_link_directories(iprovision PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/lib
    ${CUDNN_LIBRARY_DIR}
)
# Embed the runtime library paths so no LD_LIBRARY_PATH is needed.
# DT_RPATH (not RUNPATH) is used on purpose: onnxruntime dlopens libonnxruntime_providers_cuda.so,
# which has no search path of its own, and only the executable's DT_RPATH is applied to its cuDNN
# dependencies. This makes it pick the Pascal-capable cuDNN in CUDNN_LIB_DIR over the system one.
set(IPROVISION_RPATH ${ONNXRUNTIME_ROOT_DIR}/lib)
if(CUDNN_LIBRARY_DIR)
    list(PREPEND IPROVISION_RPATH ${CUDNN_LIBRARY_DIR})
endif()
set_target_properties(iprovision PROPERTIES BUILD_RPATH "${IPROVISION_RPATH}")
target_link_options(iprovision PRIVATE -Wl,--disable-new-dtags)
