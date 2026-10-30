#include "isurgical/isurgical.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "infer/infer_seg.hpp"
#include "postproc/postproc_seg.hpp"
#include "infer/infer_rsd.hpp"
#include "postproc/postproc_rsd.hpp"
#include "cap/cap.hpp"
#include "record/record.hpp"
#include "common/blocking_queue.hpp"
#include "common/isurgical_shm.hpp"

// The following are only used in the iSurgical.cpp implementation, so they are in an anonymous namespace to avoid polluting the global namespace.
namespace {

  // Rename std::chrono::steady_clock as Clock for brevity
  // It is used for timestamps and measuring elapsed time
  using Clock = std::chrono::steady_clock;

  // Source frame captured from the camera or video file.
  struct CapturedFrame {
      // The index of the frame in the source stream, starting at 0
      int64_t index = 0;
      // The timestamp of the frame capture, in steady_clock time.
      Clock::time_point timestamp;
      cv::Mat frame;
  };

  // The result of running the segmentation model on a frame, plus the original frame and its index.
  struct InferredFrame {
      int64_t index = 0;
      // The timestamp of the frame capture, in steady_clock time.
      Clock::time_point timestamp;
      cv::Mat frame;
      SegPostResult result;
      double infer_ms = 0.0;
  };

  struct RsdFrame {
      cv::Mat frame;
      double time_s = 0.0;
  };

  // Print the inference result to stdout, including the correlation scores and the detected regions for each head.
  void PrintResult(int64_t frame_index, const SegPostResult& result, const SegPostProcessor& postprocessor,
                  double infer_ms) {
      std::cout << "frame " << frame_index << "  " << infer_ms << " ms  scores:";
      for (float s : result.correlation_scores) std::cout << " " << s;
      std::cout << "\n";
      for (int head = 0; head < SegResult::kNumHeads; ++head) {
          std::cout << "  head_" << head << ":";
          for (const auto& region : result.heads[head].regions) {
              std::cout << " [" << postprocessor.LabelName(region.class_id) << " area " << region.area_ratio * 100.0f
                        << "% conf " << region.confidence << "]";
          }
          std::cout << "\n";
      }
  }

  constexpr const char* kWindowName = "iprovision";

  // Full-screen display window; the frame is scaled to the screen with its aspect ratio kept
  void OpenDisplay(bool windowed) {
      cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
      if (!windowed) {
          cv::setWindowProperty(kWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
      }
  }

}  

// Print usage and exit if the command-line arguments are invalid or missing
void iSurgicalConfig::PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --model <path>     ONNX model (default: data/models/lc_seg_model.onnx)\n"
              << "  --config <path>    class / threshold config (default: configs/config.yaml)\n"
              << "  --procedure <name> section of the config to use (default: laparoscopic cholecystectomy)\n"
              << "  --input-method <video|camera|rtsp>  input type (default: video)\n"
              << "  --input <src>      camera index, video file or image file (default: data/inputs/video01.mp4)\n"
              << "                     video files are replayed in real time, like a camera\n"
              << "  --output <path>    write the visualization to a video or image file\n"
              << "  --raw-output <path> write the unannotated source (every frame) to a video or image file\n"
              << "  --head <0-3>       draw only this head (default: all heads fused)\n"
              << "  --cpu              run on CPU instead of CUDA\n"
              << "  --letterbox        keep aspect ratio when resizing to the model input\n"
              << "  --crop-border      crop the black border around the endoscope view\n"
              << "  --show             display the visualization full screen (q / Esc to quit)\n"
              << "  --windowed         with --show, use a resizable window instead of full screen\n"
              << "  --no-rsd           disable phase / RSD inference (preprocess_step + multimodal_rsd_step)\n"
              << "  --shm <name>       publish tables and per-frame status to this shared memory\n"
              << "                     (created by the parent process, see common/isurgical_shm.hpp)\n";
}

// Parse command-line arguments and populate the config
bool iSurgicalConfig::FromArgs(int argc, char* argv[], iSurgicalConfig& config) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
      if (arg == "--model" || arg == "--config" || arg == "--procedure" || arg == "--input" || arg == "--output" ||
          arg == "--raw-output" || arg == "--head" || arg == "--input-method") {
        const char* value = next();
        if (!value) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }
        if (arg == "--model") config.model_path = value;
        else if (arg == "--config") config.config_path = value;
        else if (arg == "--procedure") config.procedure = value;
        else if (arg == "--input-method") config.input_method = value;
        else if (arg == "--input") config.input = value;
        else if (arg == "--output") config.output = value;
        else if (arg == "--raw-output") config.raw_output = value;
        else config.head = std::atoi(value);
      } else if (arg == "--cpu") {
        config.use_cuda = false;
      } else if (arg == "--letterbox") {
        config.preproc_config.resize_mode = SegResizeMode::kLetterbox;
      } else if (arg == "--crop-border") {
        config.preproc_config.crop_black_border = true;
      } else if (arg == "--show") {
        config.show = true;
      } else if (arg == "--windowed") {
        config.windowed = true;
      } else if (arg == "--no-rsd") {
        config.rsd = false;
      } else if (arg == "--shm") {
        const char* value = next();
        if (!value) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }
        config.shm_name = value;
      } else {
        return false;
      }
    }
    return true;
}

// Move the config into the iSurgical instance; the input config is not used after this
iSurgical::iSurgical(iSurgicalConfig config) : config_(std::move(config)) {}

iSurgical::~iSurgical() {
    if (shared_) munmap(shared_, sizeof(isurgical_shm::SharedState));
}

// Run the iSurgical pipeline: load the config and models, then process the input until it ends or the user quits.
// Returns the process exit code: 0 on success, -1 on failure.
int iSurgical::Run() {
    if (!Init()) return -1;
    if (config_.input_method == "image") {
      return RunImage();
    } else if (config_.input_method == "video" || config_.input_method == "camera" || config_.input_method == "rtsp") {
      return RunStream();
    } else {
      std::cerr << "Unknown input method: " << config_.input_method << std::endl;
      return -1;
    }
}


// Initialisation: check parameters and output paths first, then load the config and models.
// Return false on any failure, true on success.
bool iSurgical::Init() {
    // 1. Check the head parameter is valid before loading the model, so a bad head fails fast
    // Valid values are -1 (all heads) or 0 ~ kNumHeads-1 (only one head)
    if (config_.head < -1 || config_.head >= SegResult::kNumHeads) {
        std::cerr << "head must be in [0, " << SegResult::kNumHeads - 1 << "] or -1 for all heads" << std::endl;
        return false;
    }

    // 2. Check the output paths before loading the model, so a bad path fails fast
    // Fail before loading the model if an output cannot be written
    // Check out output and raw_output
    for (const std::string& output : {config_.output, config_.raw_output}) {
        if (output.empty()) continue;                       
        // Get the parent directory of the output path
        const std::filesystem::path out_dir = std::filesystem::path(output).parent_path();
        // The specified output directory must exist; we do not create it automatically
        if (!out_dir.empty() && !std::filesystem::is_directory(out_dir)) {
            std::cerr << "Output directory does not exist: " << out_dir.string() << std::endl;
            return false;
        }
    }

    // 3. Read the yaml config and create the postprocessors; fail fast if the config is invalid
    // Load the config before the model so a bad config fails fast
    try {
        // Set up postprocessors for segmentation from config file.
        postprocessor_ = std::make_unique<SegPostProcessor>(SegPostConfig::FromYaml(config_.config_path, config_.procedure));
        // Set up postprocessor for phase / RSD from config file.
        rsd_postprocessor_ = std::make_unique<RsdPostProcessor>(RsdConfig::FromYaml(config_.config_path, config_.procedure));

    } catch (const std::exception& e) {
        // YAML parsing or validation failed; print the error and exit
        std::cerr << "Invalid config: " << e.what() << std::endl;
        return false;
    }

    // 4. Set up shared memory if requested; fail fast if it cannot be opened or mapped
    if (!config_.shm_name.empty()) {                        // specific shm name from parent process only
        if (!OpenSharedMemory()) return false;              // mount shm_open + mmap，and check magic / version
        PublishTables();                                    // write phase class name, colour and threshold tables to shared memory
                                                            // set tables_ready=1 inform parent process that the tables are ready for reading
    }

    // Load the segmentation model
    try {
        // Parameters: model path, whether to use CUDA, device ID 0, preprocessing config
        engine_ = std::make_unique<SegInferenceEngine>(config_.model_path, config_.use_cuda, 0,
                                                       config_.preproc_config);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model " << config_.model_path << ": " << e.what() << std::endl;
        return false;
    }

    // Load the phase / RSD model if requested.
    if (config_.rsd) {                                      
        try {
            rsd_engine_ = std::make_unique<RsdInferenceEngine>(config_.rsd_preprocess_path, config_.rsd_model_path,
                                                               config_.use_cuda);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load phase / RSD models: " << e.what() << std::endl;
            return false;
        }
    }

    // Start to RunImage() / RunStream()
    return true;                                            
}

int iSurgical::RunImage() {
    const SegPostProcessor& postprocessor = *postprocessor_;
    SegResult seg_result;
    SegPostResult post_result;
    cv::Mat vis;

    cv::Mat image = cv::imread(config_.input);
    if (image.empty()) {
        std::cerr << "Could not read the image " << config_.input << std::endl;
        return -1;
    }
    const auto start = Clock::now();
    if (!engine_->Run(image, seg_result)) return -1;
    const double infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    postprocessor.Process(seg_result, post_result);
    PrintResult(0, post_result, postprocessor, infer_ms);

    postprocessor.Draw(image, post_result, vis, config_.head);
    RsdResult rsd_result;
    if (rsd_engine_) {
        if (!rsd_engine_->Run(image, 0.0, rsd_result)) return -1;
        std::cout << rsd_postprocessor_->Describe(rsd_result) << std::endl;
        rsd_postprocessor_->Draw(vis, rsd_result);
    }
    PublishStatus(0, post_result, rsd_result.phase >= 0 ? &rsd_result : nullptr);
    if (!config_.output.empty() && !cv::imwrite(config_.output, vis)) {
        std::cerr << "Failed to write " << config_.output << std::endl;
        return -1;
    }
    if (!config_.raw_output.empty() && !cv::imwrite(config_.raw_output, image)) {
        std::cerr << "Failed to write " << config_.raw_output << std::endl;
        return -1;
    }
    if (config_.show) {
        OpenDisplay(config_.windowed);
        cv::imshow(kWindowName, vis);
        cv::waitKey(0);
    }
    return 0;
}

// State shared by the stream threads; it lives on RunStream()'s stack for the whole run
struct iSurgical::StreamPipeline {
    std::unique_ptr<CaptureImg> capture;
    double source_fps = 0.0;        // source frame rate, used for recording and file time
    bool live = false;              // camera or network stream (not a file)
    double rsd_interval_s = 1.0;    // phase / RSD and encoder sampling interval (s)

    // Cameras and real-time file replay both keep only the newest frame,
    // so inference never lags behind the feed; frames it cannot keep up with are skipped
    // Producer and consumer threads communicate through these queues, which are thread-safe and block when full.
    BlockingQueue<CapturedFrame> frame_queue{1, true};

    // Results wait here for the main thread; the queue blocks when full instead of dropping
    BlockingQueue<InferredFrame> result_queue{4};

    // Phase / RSD runs on its own thread at RSD_InferFPS (1 fps, the training rate) of source time.
    // The displayed result changes only every RSD_Period seconds and is drawn on every frame in between.
    // The queue absorbs short stalls; frames are only dropped if inference falls behind for 8 steps.
    BlockingQueue<RsdFrame> rsd_queue{8, true};

    std::atomic<bool> stop{false};                  // tells the capture thread to stop 
    std::atomic<bool> inference_failed{false};      // set true when segmentation inference fails 
    std::atomic<bool> raw_record_failed{false};     // set true when the raw video cannot be opened
    std::mutex rsd_mutex;                           // guards latest_rsd and rsd_postprocessor_ state 
    RsdResult latest_rsd;                           // newest phase / RSD result for the shared memory, under rsd_mutex
};

// Namespace for writeing video frames and displaying them
namespace {

  // Writes one frame to the video at path, opening it on the first frame once the frame size is known.
  // Returns false if the file cannot be opened
  bool WriteVideoFrame(const std::string& path, double fps, const cv::Mat& frame,
                       std::unique_ptr<VideoRecord>& recorder) {
      if (!recorder) {
          recorder = std::make_unique<VideoRecord>(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
          if (!recorder->IsOpened()) return false;
      }
      recorder->WriteFrame(frame);
      return true;
  }

  // Annotated output video. Holds the previous result over the skipped source frames so the recording
  // keeps the source timeline
  struct AnnotatedRecorder {
      std::unique_ptr<VideoRecord> recorder;
      int64_t last_index = -1;        // index of the last recorded frame 
      cv::Mat last_vis;               // previous drawn frame, used to fill skipped frames

      // Returns false if the file cannot be opened 
      bool Write(const std::string& path, double fps, int64_t index, const cv::Mat& vis) {
          for (int64_t i = last_index + 1; i < index && !last_vis.empty(); ++i) {
              recorder->WriteFrame(last_vis);
          }
          if (!WriteVideoFrame(path, fps, vis, recorder)) return false;
          last_index = index;
          vis.copyTo(last_vis);       // deep copy: vis is overwritten next frame
          return true;
      }
  };

  // Shows vis with the measured capture rate; returns false when the user presses Esc or q
  bool ShowFrame(cv::Mat& vis, double fps) {
      // Draw the FPS text twice: a thick black outline, then white on top, for readability
      const std::string fps_text = cv::format("%.1f FPS", fps);
      cv::putText(vis, fps_text, cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      cv::putText(vis, fps_text, cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::imshow(kWindowName, vis);
      const int key = cv::waitKey(1);         // also lets the window refresh 
      return key != 27 && key != 'q';         // Esc or q quits 
  }

}

// Camera or video stream: capture thread -> inference thread -> main thread (draw / record / display)
// Phase / RSD runs on its own side thread
int iSurgical::RunStream() {

    StreamPipeline pipeline;

    // Open the input source (camera, video file or network stream)
    try {
        pipeline.capture = std::make_unique<CaptureImg>(config_.input);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;                         // cannot open the source
        return -1;
    }

    pipeline.source_fps = pipeline.capture->GetSourceFPS();
    pipeline.live = pipeline.capture->IsLive();
    pipeline.rsd_interval_s = 1.0 / rsd_postprocessor_->Config().infer_fps;

    std::thread capture_thread([&] { CaptureLoop(pipeline); });
    std::thread inference_thread([&] { InferenceLoop(pipeline); });
    std::thread rsd_thread([&] { RsdLoop(pipeline); });
    const int exit_code = DisplayLoop(pipeline);                    // runs on the main thread

    // Shut down in pipeline order; closing the queues unblocks any waiting Push/Pop
    pipeline.stop = true;
    pipeline.frame_queue.Close();
    pipeline.result_queue.Close();
    pipeline.rsd_queue.Close();

    // Wait for every thread to finish
    capture_thread.join();
    inference_thread.join();
    rsd_thread.join();

    // Report errors in priority order 
    if (pipeline.inference_failed) {
        std::cerr << "Stopped: segmentation inference failed." << std::endl;
        return -1;
    }
    if (exit_code != 0) {
        std::cerr << "Stopped: could not record to " << config_.output << std::endl;
        return exit_code;
    }
    if (pipeline.raw_record_failed) {
        std::cerr << "Stopped: could not record to " << config_.raw_output << std::endl;
        return -1;
    }

    // Number of frames dropped from frame_queue because inference could not keep up
    std::cout << "frames skipped by inference: " << pipeline.frame_queue.Dropped() << std::endl;
    return 0;
}

// 1. Capture thread: read frames and feed every queue 
void iSurgical::CaptureLoop(StreamPipeline& pipeline) {
    const auto start = Clock::now();            // start of capture, base for live time
    double next_rsd_time = 0.0;                 // next source time to sample for RSD 

    // Recorded here rather than in the main thread (display) so the raw video keeps every source frame,
    // including those inference skips
    std::unique_ptr<VideoRecord> raw_recorder;
    for (int64_t index = 0; !pipeline.stop; ++index) {
        CapturedFrame item;                     // fresh Mat each time, queued frames must not share a buffer

        if (!pipeline.capture->Capture(item.frame)) break;          // end of source or read error
        if (!config_.raw_output.empty() &&
            !WriteVideoFrame(config_.raw_output, pipeline.source_fps, item.frame, raw_recorder)) {
            pipeline.raw_record_failed = true;  // reported after shutdown
            break;
        }
        item.index = index;                     // frame index in the source 
        item.timestamp = Clock::now();          // capture time, used for latency

        // Source time: frame position for files (matches the video timeline), wall clock for live feeds
        const double time_s = pipeline.live ? std::chrono::duration<double>(item.timestamp - start).count()
                                            : static_cast<double>(index) / pipeline.source_fps;


        // Small tolerance so a 25 fps file samples exactly every 25th frame, as in the evaluation
        if (rsd_engine_ && time_s + 1e-6 >= next_rsd_time) {
            pipeline.rsd_queue.Push(RsdFrame{item.frame, time_s});    // read-only share with seg

            // Advance to the next sampling point after this time (skips missed points)
            while (next_rsd_time <= time_s + 1e-6) next_rsd_time += pipeline.rsd_interval_s;
        }
        if (!pipeline.frame_queue.Push(std::move(item))) break;     // queue closed: stop capturing 
    }
    // Finish the raw video and close the queues so downstream threads can end
    if (raw_recorder) raw_recorder->Release();
    pipeline.frame_queue.Close();
    pipeline.rsd_queue.Close();
}

// Inference thread: segmentation + post-processing 
void iSurgical::InferenceLoop(StreamPipeline& pipeline) {
    CapturedFrame item;
    // take the newest frame; false once closed and empty 
    while (pipeline.frame_queue.Pop(item)) {    
        InferredFrame out;
        SegResult seg_result;
        const auto start = Clock::now();
        // run the segmentation model 
        if (!engine_->Run(item.frame, seg_result)) {                
            pipeline.inference_failed = true;
            break;
        }
        // inference time (ms) 
        out.infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();   
        // post-process into the final result 
        postprocessor_->Process(seg_result, out.result);            
        // Carry the frame info along for the main thread
        out.index = item.index;
        out.timestamp = item.timestamp;
        out.frame = std::move(item.frame);
        // queue closed: stop 
        if (!pipeline.result_queue.Push(std::move(out))) break;     
    }
    pipeline.frame_queue.Close();               // unblock the capture thread if we stopped early 
    pipeline.result_queue.Close();              // tell the main thread there are no more results 
}

// Phase / RSD thread
void iSurgical::RsdLoop(StreamPipeline& pipeline) {
    // rsd off: nothing to do
    if (!rsd_engine_) return;                   
    RsdFrame item;
    //When rsd queue is empty, Pop() will block until a new item is pushed or the queue is closed.
    while (pipeline.rsd_queue.Pop(item)) {
        RsdResult result;

        if (!rsd_engine_->Run(item.frame, item.time_s, result)) {
            // Only this side branch stops; segmentation keeps running
            std::cerr << "Phase / RSD inference stopped; segmentation continues." << std::endl;
            break;
        }
        bool displayed = false;
        {
            std::lock_guard<std::mutex> lock(pipeline.rsd_mutex);
            displayed = rsd_postprocessor_->Update(result);         // true when the displayed result changed 
            pipeline.latest_rsd = result;                           // for PublishStatus() 
        }
        if (displayed) std::cout << rsd_postprocessor_->Describe(result) << std::endl;    // print only on update 
    }
    pipeline.rsd_queue.Close();
}

// Main thread: print / publish / draw / record / display. Returns -1 if the output video cannot be opened
int iSurgical::DisplayLoop(StreamPipeline& pipeline) {
    if (config_.show) OpenDisplay(config_.windowed);                // open the display window 

    int exit_code = 0;
    AnnotatedRecorder recorder;
    InferredFrame item;
    // the current frame being drawn
    cv::Mat vis;                                 
    while (pipeline.result_queue.Pop(item)) {
        // End-to-end latency from capture to now
        const double latency_ms = std::chrono::duration<double, std::milli>(Clock::now() - item.timestamp).count();
        PrintResult(item.index, item.result, *postprocessor_, item.infer_ms);
        std::cout << "  latency " << latency_ms << " ms\n";
        // shared memory is attached
        if (shared_) {                          
            PublishStreamStatus(pipeline, item.index, item.result);
             // the parent asked us to stop
            if (StopRequested()) break;         
        }

        // Draw only when the frame will be recorded or shown
        if (!config_.output.empty() || config_.show) DrawStreamFrame(pipeline, item.frame, item.result, vis);
        if (!config_.output.empty() && !recorder.Write(config_.output, pipeline.source_fps, item.index, vis)) {
            exit_code = -1;
            break;
        }
        if (config_.show && !ShowFrame(vis, pipeline.capture->GetFPS())) break;
    }
    if (recorder.recorder) recorder.recorder->Release();            // finish the output video / 收尾輸出影片
    return exit_code;
}

// Publishes the frame's result with the newest phase / RSD result to the shared memory
void iSurgical::PublishStreamStatus(StreamPipeline& pipeline, int64_t frame_index, const SegPostResult& seg) {
    RsdResult rsd_result;
    {
        std::lock_guard<std::mutex> lock(pipeline.rsd_mutex);
        // copy under the lock
        rsd_result = pipeline.latest_rsd;        
    }
    // phase < 0 means no RSD result yet 
    PublishStatus(frame_index, seg, rsd_result.phase >= 0 ? &rsd_result : nullptr);
}

// Draws the segmentation, plus the displayed phase / RSD result when rsd is on, into vis
void iSurgical::DrawStreamFrame(StreamPipeline& pipeline, const cv::Mat& frame, const SegPostResult& seg,
                                cv::Mat& vis) {
    postprocessor_->Draw(frame, seg, vis, config_.head);
    if (!rsd_engine_) return;
    RsdResult rsd_result;
    {
        std::lock_guard<std::mutex> lock(pipeline.rsd_mutex);
        rsd_result = rsd_postprocessor_->Displayed();               // the result currently on display 
    }
    rsd_postprocessor_->Draw(vis, rsd_result);                      // overlay phase / RSD 
}

bool iSurgical::OpenSharedMemory() {
    using isurgical_shm::SharedState;
    // The parent creates and sizes the segment; isurgical only attaches to it
    const int fd = shm_open(config_.shm_name.c_str(), O_RDWR, 0);
    if (fd < 0) {
        std::cerr << "Cannot open shared memory " << config_.shm_name << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < sizeof(SharedState)) {
        std::cerr << "Shared memory " << config_.shm_name << " is too small" << std::endl;
        close(fd);
        return false;
    }
    void* addr = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::cerr << "Cannot map shared memory " << config_.shm_name << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    shared_ = static_cast<SharedState*>(addr);
    if (shared_->magic != isurgical_shm::kMagic || shared_->version != isurgical_shm::kVersion) {
        std::cerr << "Shared memory " << config_.shm_name << " has an unexpected layout (version "
                  << shared_->version << ", expected " << isurgical_shm::kVersion << ")" << std::endl;
        munmap(shared_, sizeof(SharedState));
        shared_ = nullptr;
        return false;
    }
    return true;
}

void iSurgical::PublishTables() {
    using namespace isurgical_shm;
    Tables& tables = shared_->tables;
    const auto& phases = rsd_postprocessor_->Config().phase_names;
    tables.num_phases = std::min<int>(static_cast<int>(phases.size()), kMaxPhases);
    for (int i = 0; i < tables.num_phases; ++i) CopyName(tables.phase_names[i], phases[i].c_str());

    const SegPostConfig& seg = postprocessor_->Config();
    tables.num_classes = std::min(seg.num_classes, kMaxClasses);
    for (int i = 0; i < tables.num_classes; ++i) {
        CopyName(tables.class_names[i], seg.label_names[i].c_str());
        const cv::Scalar& bgr = seg.class_colours[i];           // stored as BGR in [0, 255]
        tables.class_colours[i][0] = static_cast<float>(bgr[2] / 255.0);
        tables.class_colours[i][1] = static_cast<float>(bgr[1] / 255.0);
        tables.class_colours[i][2] = static_cast<float>(bgr[0] / 255.0);
    }
    shared_->tables_ready.store(1, std::memory_order_release);
}

// Writes the per-frame status to the shared memory, including the segmentation counts and the latest phase / RSD result
void iSurgical::PublishStatus(int64_t frame_index, const SegPostResult& seg, const RsdResult* rsd) {
    if (!shared_) return;
    {
      using namespace isurgical_shm;
      Status status{};
      status.timestamp = static_cast<int64_t>(std::time(nullptr));
      status.frame_index = frame_index;

      // Count the number of detected regions for each class in the segmentation result
      for (const auto& head : seg.heads) {
          for (const auto& region : head.regions) {
              if (region.class_id >= 0 && region.class_id < kMaxClasses) ++status.detected[region.class_id];
          }
      }

      // Copy the latest phase / RSD result if available; otherwise set progress and remaining_minutes to -1
      if (rsd && rsd->phase >= 0) {
          CopyName(status.phase, rsd_postprocessor_->PhaseName(rsd->phase).c_str());
          status.progress = std::round(rsd->progress * 100.0f) / 100.0f;
          status.remaining_minutes = static_cast<int32_t>(std::lround(rsd->rsd_s / 60.0));
      } else {
          status.progress = -1.0f;
          status.remaining_minutes = -1;
      }
      WriteStatus(*shared_, status);
    }
}

bool iSurgical::StopRequested() const {
    return shared_ && shared_->stop_requested.load(std::memory_order_acquire) != 0;
}
