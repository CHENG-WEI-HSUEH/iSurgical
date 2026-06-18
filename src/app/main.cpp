#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

#include "pipeline.hpp"

int main(int argc, char* argv[])
{
    std::string config_path;

    // Parse command line arguments to get the config file path
    if (argc > 1) {
        config_path = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> " << std::endl;
        return EXIT_FAILURE;
    }

    // Load the YAML configuration file
    YAML::Node config = YAML::LoadFile(config_path);
    
    // Create and run the inference pipeline application
    InferencePipeline app{config};

    // Run the application
    app.run();
    
    std::string rec_path = config["RecordingPath"].as<std::string>();
    std::string cmd1 = "./scripts/move_mp4.sh " + rec_path + "/raw.mp4";
    std::string cmd2 = "./scripts/move_mp4.sh " + rec_path + "/segmentation.mp4";

    if (std::system(cmd1.c_str()) != 0) {
        HOLOSCAN_LOG_ERROR("Failed to move {}/raw.mp4 to remote server", rec_path);
    }
    if (std::system(cmd2.c_str()) != 0) {
        HOLOSCAN_LOG_ERROR("Failed to move {}/segmentation.mp4 to remote server", rec_path);
    }

    return 0;
}
