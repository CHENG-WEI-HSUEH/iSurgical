#include "isurgical/isurgical.hpp"

int main(int argc, char* argv[]) {
    // Defaults come from iSurgicalConfig; command-line options override them.
    // Fields can also be set directly here, e.g. config.input = "0"; config.show = true;
    iSurgicalConfig config;
    if (!iSurgicalConfig::FromArgs(argc, argv, config)) {
        iSurgicalConfig::PrintUsage(argv[0]);
        return -1;
    }

    iSurgical app(config);
    return app.Run();
}
