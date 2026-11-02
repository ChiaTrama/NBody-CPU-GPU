#include "cli.hpp"

#include <stdexcept>
#include <string>

namespace {

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    ++i;
    return argv[i];
}

int parse_positive_int(const std::string& value, const std::string& option) {
    try {
        std::size_t parsed_characters = 0;
        const int parsed = std::stoi(value, &parsed_characters);
        if (parsed_characters != value.size() || parsed <= 0) {
            throw std::runtime_error("invalid value for " + option + ": " + value +
                                     " (expected a positive integer)");
        }
        return parsed;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("invalid value for " + option + ": " + value +
                                 " (expected a positive integer)");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("invalid value for " + option + ": " + value +
                                 " (expected a positive integer)");
    }
}

} // namespace

Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--N") {
            cfg.N = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--steps") {
            cfg.steps = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--save-every") {
            cfg.save_every = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--block-size") {
            cfg.block_size = parse_positive_int(require_value(i, argc, argv), arg);
        } else if (arg == "--dt") {
            cfg.dt = std::stod(require_value(i, argc, argv));
        } else if (arg == "--epsilon") {
            cfg.epsilon = std::stod(require_value(i, argc, argv));
        } else if (arg == "--G") {
            cfg.G = std::stod(require_value(i, argc, argv));
        } else if (arg == "--box-size") {
            cfg.box_size = std::stod(require_value(i, argc, argv));
        } else if (arg == "--min-distance") {
            cfg.min_distance = std::stod(require_value(i, argc, argv));
        } else if (arg == "--seed") {
            cfg.seed = static_cast<std::uint64_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--init") {
            cfg.init = require_value(i, argc, argv);
        } else if (arg == "--integrator") {
            cfg.integrator = require_value(i, argc, argv);
        } else if (arg == "--output") {
            cfg.output = require_value(i, argc, argv);
        } else if (arg == "--energy-output") {
            cfg.energy_output = require_value(i, argc, argv);
        } else if (arg == "--benchmark") {
            cfg.benchmark = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (cfg.N == 0 || cfg.steps < 0 || cfg.save_every <= 0 ||
        cfg.dt <= 0.0 || cfg.epsilon < 0.0 ||
        cfg.G <= 0.0 || cfg.box_size <= 0.0 || cfg.min_distance < 0.0) {
        throw std::runtime_error("invalid configuration values");
    }

    if (cfg.integrator != "euler" && cfg.integrator != "verlet") {
        throw std::runtime_error("invalid integrator: " + cfg.integrator);
    }

    return cfg;
}
