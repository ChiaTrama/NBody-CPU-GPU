#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cli.hpp"
#include "cuda_tiled.hpp"
#include "initialization.hpp"
#include "io.hpp"

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        Particles particles = initialize_particles(cfg);

        if (cfg.benchmark) {
            if (!run_cuda_tiled_benchmark(particles, cfg)) {
                std::cerr << "error: CUDA tiled benchmark failed\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        auto trajectory = open_trajectory_file(cfg.output);
        auto energy_file = open_energy_file(cfg.energy_output);

        if (!run_cuda_tiled_simulation(particles, cfg, trajectory, energy_file)) {
            std::cerr << "error: CUDA tiled simulation failed\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
