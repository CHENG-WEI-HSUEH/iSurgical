#pragma once

#include <memory>
#include <string>

#include "preproc/preproc_seg.hpp"

class SegInferenceEngine;
class SegPostProcessor;
class RsdInferenceEngine;
class RsdPostProcessor;
struct SegPostResult;
struct RsdResult;
namespace isurgical_shm { struct SharedState; }

struct iSurgicalConfig {
    // Segmentation model path
    std::string model_path = "data/models/lc_seg_model.onnx";
    // Procedure config path
    std::string config_path = "configs/config.yaml";
    // Procedure name in the config file
    std::string procedure = "laparoscopic cholecystectomy";
    // Remaining surgical duration (RSD) model paths
    std::string rsd_preprocess_path = "data/models/preprocess_step.onnx";
    std::string rsd_model_path = "data/models/multimodal_rsd_step.onnx";

    // Turn on phase recognition + remaining surgical duration
    bool rsd = true;                    

    // Input method: camera index, video file or RTSP stream; video files are replayed in real time, like a camera
    // video/ camera/ rtsp
    std::string input_method = "video";
    // Video input: camera index, video file or RTSP stream; video files are replayed in real time, like a camera
    std::string input = "data/inputs/video01.mp4";
    // Video output: annotated visualization
    std::string output;                 
    // Video output: unannotated source
    std::string raw_output;            
    int head = -1;                      // segmentation head to visualize, -1 = all heads fused
    bool use_cuda = true;
    bool show = false;
    bool windowed = false;              // show uses full screen unless this is set

    // Preprocessing config for the segmentation model
    SegPreprocConfig preproc_config;

    // shared memory created by a parent process, "" = none
    std::string shm_name;               

    // Fills the config from command-line options; false on an unknown or invalid option
    static bool FromArgs(int argc, char* argv[], iSurgicalConfig& config);
    static void PrintUsage(const char* prog);
};

// The whole application: set up an iSurgicalConfig, then call Run().
//
//     iSurgicalConfig config;
//     config.input = "0";             // camera 0
//     config.show = true;
//     iSurgical app(config);
//     return app.Run();
class iSurgical {
  public:
    // Explicit constructor to avoid accidental copies of the config; the config is moved into the iSurgical instance
    explicit iSurgical(iSurgicalConfig config);
    ~iSurgical();

    // Loads the config and models, then processes the input until it ends or the user quits.
    // Returns the process exit code: 0 on success, -1 on failure.
    int Run();

    // Returns the config used to create this instance; the config is not modified after construction
    const iSurgicalConfig& Config() const { return config_; }

  private:
    // State shared by the stream threads, defined in isurgical.cpp
    struct StreamPipeline;

    bool Init();
    int RunImage();
    int RunStream();
    // RunStream() stages: one per thread; DisplayLoop() runs on the main thread
    void CaptureLoop(StreamPipeline& pipeline);
    void InferenceLoop(StreamPipeline& pipeline);
    void RsdLoop(StreamPipeline& pipeline);
    int DisplayLoop(StreamPipeline& pipeline);
    void PublishStreamStatus(StreamPipeline& pipeline, int64_t frame_index, const SegPostResult& seg);
    void DrawStreamFrame(StreamPipeline& pipeline, const cv::Mat& frame, const SegPostResult& seg, cv::Mat& vis);
    bool OpenSharedMemory();
    void PublishTables();
    // rsd is null when no phase result exists yet
    void PublishStatus(int64_t frame_index, const SegPostResult& seg, const RsdResult* rsd);
    bool StopRequested() const;

    iSurgicalConfig config_;
    std::unique_ptr<SegPostProcessor> postprocessor_;
    std::unique_ptr<RsdPostProcessor> rsd_postprocessor_;
    std::unique_ptr<SegInferenceEngine> engine_;
    std::unique_ptr<RsdInferenceEngine> rsd_engine_;         // null when rsd is off
    isurgical_shm::SharedState* shared_ = nullptr;          // mapped when shm_name is set
};
