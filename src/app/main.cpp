#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

#include "pipeline.hpp"

int main(int argc, char* argv[])
{
    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return EXIT_FAILURE;
    }

    YAML::Node config = YAML::LoadFile(config_path);

    InferencePipeline app{config};
    app.run();

    return 0;
}
