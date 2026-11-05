// Shared helpers for every iSurgical test executable: paths, scratch folders, files, synthetic images and videos.
#pragma once

// std::chrono
#include <chrono>
// std::getenv
#include <cstdlib>
// std::filesystem
#include <filesystem>
// std::function
#include <functional>
// std::ifstream / std::ofstream
#include <fstream>
// std::ostringstream
#include <sstream>
// std::string
#include <string>
// std::this_thread
#include <thread>

// getpid
#include <unistd.h>

// GoogleTest: current test name, GTEST_SKIP
#include <gtest/gtest.h>
// cv::Mat, cv::VideoWriter, cv::VideoCapture
#include <opencv2/opencv.hpp>

// Own namespace
namespace test_helpers {

// Short alias
namespace fs = std::filesystem;

// Absolute path of a file in the project (e.g. "configs/config.yaml")
inline std::string ProjectFile(const std::string& relative) {
    // PROJECT_ROOT_DIR comes from test/CMakeLists.txt
    return (fs::path(PROJECT_ROOT_DIR) / relative).string();
}

// The real procedure config
inline std::string ConfigPath() {
    // configs/config.yaml
    return ProjectFile("configs/config.yaml");
}

// The procedure section used everywhere
inline const char* kProcedure = "laparoscopic cholecystectomy";

// Absolute path of a generated fake model (see test/tools/make_fake_models.py)
inline std::string FakeModel(const std::string& name) {
    // FAKE_MODEL_DIR comes from test/CMakeLists.txt
    return (fs::path(FAKE_MODEL_DIR) / name).string();
}

// Absolute path of a real model in data/models; "" when it is not there (data/ is not in git)
inline std::string RealModel(const std::string& name) {
    // Full path
    const std::string path = ProjectFile("data/models/" + name);
    // Only if present
    return fs::exists(path) ? path : "";
}

// Write text to a file (creating folders), replacing it
inline void WriteFile(const fs::path& path, const std::string& text) {
    // Folder
    fs::create_directories(path.parent_path());
    // Truncate and write
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

// Read a whole file ("" if missing)
inline std::string ReadFile(const fs::path& path) {
    // Open
    std::ifstream in(path, std::ios::binary);
    // Buffer
    std::ostringstream buf;
    // Copy
    buf << in.rdbuf();
    // Text
    return buf.str();
}

// Number of frames in a video file, counted by decoding (CAP_PROP_FRAME_COUNT is only an estimate)
inline int CountFrames(const std::string& path) {
    // Open
    cv::VideoCapture cap(path);
    // Count
    int n = 0;
    // One frame at a time
    cv::Mat frame;
    // Until the end
    while (cap.read(frame)) ++n;
    // Total
    return n;
}

// Write a small MP4 (mp4v); frame i is filled by paint(i, frame). Returns true if every frame was written
inline bool MakeVideo(const std::string& path, int frames, cv::Size size, double fps,
                      const std::function<void(int, cv::Mat&)>& paint) {
    // Same backend and codec as the program uses
    cv::VideoWriter writer(path, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    // Could not open
    if (!writer.isOpened()) return false;
    // Each frame
    for (int i = 0; i < frames; ++i) {
        // Black frame
        cv::Mat frame(size, CV_8UC3, cv::Scalar::all(0));
        // Let the caller draw
        paint(i, frame);
        // Append
        writer.write(frame);
    }
    // Finish the file
    writer.release();
    // Done
    return true;
}

// A BGR test frame: black, with a pure red square (fake model → class 1, Gallbladder) and a
// yellow square (fake model → Liver) at known positions
inline cv::Mat ColourFrame(cv::Size size = cv::Size(512, 512)) {
    // Black background
    cv::Mat frame(size, CV_8UC3, cv::Scalar::all(0));
    // Red square in the top-left quarter (BGR: 0, 0, 255)
    cv::rectangle(frame, cv::Rect(size.width / 8, size.height / 8, size.width / 4, size.height / 4),
                  cv::Scalar(0, 0, 255), cv::FILLED);
    // Yellow square in the bottom-right quarter (BGR: 0, 255, 255)
    cv::rectangle(frame, cv::Rect(size.width * 5 / 8, size.height * 5 / 8, size.width / 4, size.height / 4),
                  cv::Scalar(0, 255, 255), cv::FILLED);
    // Return it
    return frame;
}

// Milliseconds since start
inline long long ElapsedMs(std::chrono::steady_clock::time_point start) {
    // Difference in ms
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

// Unique scratch folder per test, deleted afterwards (kept if KEEP_TEST_TMP is set)
class TempDir {
public:
    // Create <TEST_TMP_DIR>/<Suite>.<Test>_<pid>_<n>
    TempDir() {
        // Per-process counter
        static int counter = 0;
        // Default label
        std::string label = "global";
        // Running test
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        // Use its name
        if (info) label = std::string(info->test_suite_name()) + "." + info->name();
        // '/' from parameterised names is not allowed in a folder name
        for (char& c : label) if (c == '/') c = '_';
        // Full path
        path_ = fs::path(TEST_TMP_DIR) / (label + "_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        // Start empty
        fs::remove_all(path_);
        // Create
        fs::create_directories(path_);
    }
    // Remove
    ~TempDir() {
        // Never throw
        std::error_code ec;
        // Unless asked to keep it
        if (!std::getenv("KEEP_TEST_TMP")) fs::remove_all(path_, ec);
    }
    // Not copyable
    TempDir(const TempDir&) = delete;
    // Not assignable
    TempDir& operator=(const TempDir&) = delete;
    // The folder
    const fs::path& path() const { return path_; }
    // A file inside it, as a string
    std::string operator/(const std::string& name) const { return (path_ / name).string(); }

private:
    // Absolute path
    fs::path path_;
};

}  // namespace test_helpers

// Skip the current test when a real model is missing (data/ is not part of the repository)
#define REQUIRE_REAL_MODEL(path, name)                                                          \
    /* Look the model up */                                                                     \
    const std::string path = test_helpers::RealModel(name);                                     \
    /* Missing: skip with a clear message */                                                    \
    if (path.empty()) GTEST_SKIP() << "data/models/" << name << " not found (data/ is not in git)"
