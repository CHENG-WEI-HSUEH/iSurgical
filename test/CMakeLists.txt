# =============================================================================
# GoogleTest suite for iSurgical
#
# Build and run everything (tests + pass-rate report + coverage report) with:
#     test/run_tests.sh
#
# Or by hand:
#     cmake -S test -B build-test -DCMAKE_BUILD_TYPE=Debug \
#           -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-gpu-1.20.1 \
#           -DCUDNN_LIB_DIR=/path/to/cudnn-9.5.1/lib
#     cmake --build build-test -j$(nproc)
#     ctest --test-dir build-test --output-on-failure
#
# Stand-alone CMake project: the main CMakeLists.txt is not modified. The library sources are compiled
# again here with --coverage. Tiny fake ONNX models (test/tools/make_fake_models.py) are generated at
# build time, so no real model, GPU or video is needed; tests that use the real models in data/ skip
# themselves when those files are missing.
# =============================================================================

# Same minimum version as the main project
cmake_minimum_required(VERSION 3.20)
# Separate project name
project(iSurgical_tests LANGUAGES CXX)

# The sources use C++17
set(CMAKE_CXX_STANDARD 17)
# No silent fallback to an older standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Coverage needs an unoptimised build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # Debug = -g, no -O2
    set(CMAKE_BUILD_TYPE Debug)
# End of if
endif()

# Project root (the folder above test/)
get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
# gtest XML results and generated reports
set(REPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/reports)
# Scratch folder for files created by the tests (videos, images, YAML)
set(TEST_TMP_DIR ${CMAKE_BINARY_DIR}/tmp)
# Folder the fake ONNX models are generated into
set(FAKE_MODEL_DIR ${CMAKE_BINARY_DIR}/fake_models)

# Same cache options as the main project
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to the onnxruntime directory (with include/ and lib/)")
# Pascal-capable cuDNN (root or its lib/ folder); optional for the tests, which run on CPU
set(CUDNN_LIB_DIR "" CACHE PATH "Path to a Pascal-capable cuDNN directory (its root or its lib/ folder)")
# ONNX Runtime is required to compile the inference code
if(NOT EXISTS ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h)
    # Stop with a clear message
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT_DIR to the ONNX Runtime folder (got '${ONNXRUNTIME_ROOT_DIR}')")
# End of if
endif()
# cuDNN lib folder, accepting the root or lib/ like the main CMakeLists.txt
set(CUDNN_LIBRARY_DIR "")
# cuDNN include folder
set(CUDNN_INCLUDE_DIR "")
# Only when given
if(CUDNN_LIB_DIR)
    # Root folder with lib/
    if(EXISTS ${CUDNN_LIB_DIR}/lib)
        # lib/
        set(CUDNN_LIBRARY_DIR ${CUDNN_LIB_DIR}/lib)
        # include/
        set(CUDNN_INCLUDE_DIR ${CUDNN_LIB_DIR}/include)
    # The lib/ folder itself
    else()
        # As given
        set(CUDNN_LIBRARY_DIR ${CUDNN_LIB_DIR})
        # ../include
        get_filename_component(CUDNN_INCLUDE_DIR ${CUDNN_LIB_DIR}/../include ABSOLUTE)
    # End of if
    endif()
# End of if
endif()

# Coverage on or off (cmake -DENABLE_COVERAGE=OFF ...)
option(ENABLE_COVERAGE "Compile with --coverage so gcov can produce a coverage report" ON)
# Coverage wanted
if(ENABLE_COVERAGE)
    # Counters, no optimisation, thread-safe counter updates (the pipeline has 4 threads)
    add_compile_options(--coverage -O0 -g -fprofile-update=atomic)
    # Link libgcov
    add_link_options(--coverage)
# End of if
endif()

# All test executables directly in the build folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------
# External packages
# ---------------------------------------------------------------------------
# OpenCV: images, video I/O, drawing
find_package(OpenCV REQUIRED)
# yaml-cpp: configs/config.yaml
find_package(yaml-cpp REQUIRED)
# pthread
find_package(Threads REQUIRED)
# GoogleTest (sudo apt install libgtest-dev)
find_package(GTest REQUIRED)
# Python 3 to generate the fake models
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# Fake ONNX models, regenerated whenever the generator changes
# ---------------------------------------------------------------------------
# Every file the generator writes
set(FAKE_MODELS seg.onnx rsd_pre.onnx rsd.onnx seg_bad_input.onnx seg_missing_scores.onnx
    seg_bad_head_name.onnx seg_bad_scores_name.onnx rsd_pre_bad.onnx rsd_bad.onnx)
# Their full paths
list(TRANSFORM FAKE_MODELS PREPEND ${FAKE_MODEL_DIR}/ OUTPUT_VARIABLE FAKE_MODEL_FILES)
# How to make them: python3 -I (isolated) make_fake_models.py <folder>
add_custom_command(
    # Files produced
    OUTPUT ${FAKE_MODEL_FILES}
    # The generator
    COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_fake_models.py ${FAKE_MODEL_DIR}
    # Re-run when the script changes
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_fake_models.py
    # Shown while building
    COMMENT "Generating fake ONNX models"
# End of add_custom_command
)
# A target the tests depend on
add_custom_target(fake_models DEPENDS ${FAKE_MODEL_FILES})

# ---------------------------------------------------------------------------
# Code under test, rebuilt with coverage
# ---------------------------------------------------------------------------
# The same sources as libisurgical.so (isurgical_api in the main CMakeLists.txt), as a static library
add_library(isurgical_core STATIC
    # iSurgicalConfig, iSurgical
    ${PROJECT_ROOT}/src/isurgical/isurgical.cpp
    # CaptureImg
    ${PROJECT_ROOT}/src/cap/cap.cpp
    # VideoRecord
    ${PROJECT_ROOT}/src/record/record.cpp
    # SegPreprocessor
    ${PROJECT_ROOT}/src/preproc/preproc_seg.cpp
    # RsdPreprocessor
    ${PROJECT_ROOT}/src/preproc/preproc_rsd.cpp
    # SegInferenceEngine
    ${PROJECT_ROOT}/src/infer/infer_seg.cpp
    # RsdInferenceEngine
    ${PROJECT_ROOT}/src/infer/infer_rsd.cpp
    # SegPostConfig, SegPostProcessor
    ${PROJECT_ROOT}/src/postproc/postproc_seg.cpp
    # RsdConfig, RsdPostProcessor, FormatDuration
    ${PROJECT_ROOT}/src/postproc/postproc_rsd.cpp
# End of add_library
)
# src/ for "infer/infer_seg.hpp" etc.; ONNX Runtime and cuDNN headers; OpenCV
target_include_directories(isurgical_core PUBLIC
    # Project sources
    ${PROJECT_ROOT}/src
    # onnxruntime_cxx_api.h (the tests use Ort::Env for RsdPreprocessor)
    ${ONNXRUNTIME_ROOT_DIR}/include
    # OpenCV
    ${OpenCV_INCLUDE_DIRS}
# End of target_include_directories
)
# cuDNN headers, only if given
if(CUDNN_INCLUDE_DIR)
    # Same as the main build
    target_include_directories(isurgical_core PUBLIC ${CUDNN_INCLUDE_DIR})
# End of if
endif()
# Where libonnxruntime.so (and cuDNN) live
target_link_directories(isurgical_core PUBLIC ${ONNXRUNTIME_ROOT_DIR}/lib ${CUDNN_LIBRARY_DIR})
# Same libraries as libisurgical.so (rt for shm_open in the tests)
target_link_libraries(isurgical_core PUBLIC ${OpenCV_LIBS} yaml-cpp::yaml-cpp onnxruntime Threads::Threads rt)
# Same warnings as usual
target_compile_options(isurgical_core PRIVATE -Wall -Wextra)

# Runtime library search path baked into every test (ONNX Runtime, cuDNN first), as DT_RPATH like the main build
set(TEST_RPATH ${ONNXRUNTIME_ROOT_DIR}/lib)
# cuDNN before ONNX Runtime
if(CUDNN_LIBRARY_DIR)
    # Prepend
    list(PREPEND TEST_RPATH ${CUDNN_LIBRARY_DIR})
# End of if
endif()

# ---------------------------------------------------------------------------
# Test executables
# ---------------------------------------------------------------------------
# ctest support
enable_testing()

# add_gtest(<name> <sources...>): one GoogleTest executable + one ctest entry
function(add_gtest name)
    # The executable
    add_executable(${name} ${ARGN})
    # Code under test + GoogleTest (gtest_main provides main())
    target_link_libraries(${name} PRIVATE isurgical_core GTest::gtest GTest::gtest_main)
    # test/common/test_helpers.hpp
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    # Paths the tests need
    target_compile_definitions(${name} PRIVATE
        # configs/config.yaml, data/models
        PROJECT_ROOT_DIR="${PROJECT_ROOT}"
        # Scratch folder
        TEST_TMP_DIR="${TEST_TMP_DIR}"
        # Generated fake models
        FAKE_MODEL_DIR="${FAKE_MODEL_DIR}"
    # End of target_compile_definitions
    )
    # Warnings on the test code too
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    # Find ONNX Runtime / cuDNN at run time without LD_LIBRARY_PATH
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${TEST_RPATH}")
    # DT_RPATH also applies to libraries ONNX Runtime dlopens (CUDA provider → cuDNN)
    target_link_options(${name} PRIVATE -Wl,--disable-new-dtags)
    # Fake models must exist before the tests run
    add_dependencies(${name} fake_models)
    # Register with ctest; XML result to test/reports/xml/<name>.xml
    add_test(NAME ${name} COMMAND ${name} --gtest_output=xml:${REPORT_DIR}/xml/${name}.xml)
    # gtest's temporary files go to the scratch folder
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "TEST_TMPDIR=${TEST_TMP_DIR}")
# End of function
endfunction()

# src/common/blocking_queue.hpp
add_gtest(test_blocking_queue     unit/test_blocking_queue.cpp)
# src/common/isurgical_shm.hpp
add_gtest(test_isurgical_shm      unit/test_isurgical_shm.cpp)
# src/preproc/preproc_seg.cpp
add_gtest(test_preproc_seg        unit/test_preproc_seg.cpp)
# src/preproc/preproc_rsd.cpp
add_gtest(test_preproc_rsd        unit/test_preproc_rsd.cpp)
# src/postproc/postproc_seg.cpp
add_gtest(test_postproc_seg       unit/test_postproc_seg.cpp)
# src/postproc/postproc_rsd.cpp
add_gtest(test_postproc_rsd       unit/test_postproc_rsd.cpp)
# src/infer/infer_seg.cpp
add_gtest(test_infer_seg          unit/test_infer_seg.cpp)
# src/infer/infer_rsd.cpp
add_gtest(test_infer_rsd          unit/test_infer_rsd.cpp)
# src/cap/cap.cpp
add_gtest(test_capture            unit/test_capture.cpp)
# src/record/record.cpp
add_gtest(test_record             unit/test_record.cpp)
# iSurgicalConfig::FromArgs / PrintUsage
add_gtest(test_isurgical_config   unit/test_isurgical_config.cpp)
# iSurgical::Run end to end (image, video, shared memory, failure paths)
add_gtest(test_isurgical_app      integration/test_isurgical_app.cpp)
# The real models in data/ (skipped when missing)
add_gtest(test_real_models        integration/test_real_models.cpp)
