#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "cuda_naive.hpp"
#include "initialization.hpp"
#include "io.hpp"

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        Particles particles = initialize_particles(cfg);

        if (cfg.benchmark) {
            if (!run_cuda_naive_benchmark(particles, cfg)) {
                std::cerr << "error: CUDA naive benchmark failed\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        auto trajectory = open_trajectory_file(cfg.output);
        auto energy_file = open_energy_file(cfg.energy_output);

        if (!run_cuda_naive_simulation(particles, cfg, trajectory, energy_file)) {
            std::cerr << "error: CUDA naive simulation failed\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
