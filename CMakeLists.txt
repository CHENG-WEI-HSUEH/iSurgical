cmake_minimum_required(VERSION 3.20)

project(isurgical VERSION 0.1.0 LANGUAGES CXX)

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

# The whole pipeline as a shared library (libisurgical.so) that other programs can link against
add_library(isurgical_api SHARED
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

set_target_properties(isurgical_api PROPERTIES
    OUTPUT_NAME isurgical
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# OpenCV and src/ are PUBLIC because isurgical.hpp includes preproc_seg.hpp, which includes OpenCV
target_link_libraries(isurgical_api
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        yaml-cpp::yaml-cpp
        tbb
        cudart
        onnxruntime
        pthread
)

target_include_directories(isurgical_api
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${ONNXRUNTIME_ROOT_DIR}/include
        ${CUDNN_INCLUDE_DIR}
)

target_link_directories(isurgical_api PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/lib
    ${CUDNN_LIBRARY_DIR}
)

# Command-line app: a thin client of the library
add_executable(isurgical src/main.cpp)
target_link_libraries(isurgical PRIVATE isurgical_api)

# Embed the runtime library paths so no LD_LIBRARY_PATH is needed.
# DT_RPATH (not RUNPATH) is used on purpose: onnxruntime dlopens libonnxruntime_providers_cuda.so,
# which has no search path of its own. glibc applies the DT_RPATH of every object in the loading
# chain (onnxruntime -> libisurgical.so -> executable) to its cuDNN dependencies, so the library
# carries the paths itself and works inside host programs we do not build. This makes it pick
# the Pascal-capable cuDNN in CUDNN_LIB_DIR over the system one.
set(ISURGICAL_RPATH ${ONNXRUNTIME_ROOT_DIR}/lib)
if(CUDNN_LIBRARY_DIR)
    list(PREPEND ISURGICAL_RPATH ${CUDNN_LIBRARY_DIR})
endif()
foreach(target isurgical_api isurgical)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${ISURGICAL_RPATH}")
    target_link_options(${target} PRIVATE -Wl,--disable-new-dtags)
endforeach()

install(TARGETS isurgical_api isurgical
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)