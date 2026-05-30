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
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the YAML configuration file
    YAML::Node config = YAML::LoadFile(config_path);

    // Create and run the inference pipeline application
    InferencePipeline app{config};

    // Run the application
    app.run();

    return 0;
}
