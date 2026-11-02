#include "cpu_driver.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "initialization.hpp"
#include "integrators.hpp"
#include "io.hpp"
#include "physics.hpp"

namespace {

double seconds_between(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void advance_one_step(Particles& particles, const Config& cfg,
                      AccelerationSnapshot& old_accelerations,
                      AccelerationFunction compute_acceleration) {
    if (cfg.integrator == "euler") {
        step_euler(particles, cfg);
        compute_acceleration(particles, cfg);
    } else {
        save_accelerations(particles, old_accelerations);
        update_positions_verlet(particles, cfg);
        compute_acceleration(particles, cfg);
        update_velocities_verlet(particles, cfg, old_accelerations);
    }
}

int run_benchmark(Particles& particles, const Config& cfg,
                  AccelerationFunction compute_acceleration) {
    // Backend time includes temporary-buffer setup; compute time starts afterward.
    const auto backend_start = std::chrono::steady_clock::now();
    AccelerationSnapshot old_accelerations(
        cfg.integrator == "verlet" ? particles.size() : 0);

    // Compute runtime covers the initial force evaluation and the time loop.
    const auto compute_start = std::chrono::steady_clock::now();

    compute_acceleration(particles, cfg);

    for (int step = 1; step <= cfg.steps; ++step) {
        advance_one_step(particles, cfg, old_accelerations, compute_acceleration);
    }

    const auto compute_end = std::chrono::steady_clock::now();
    const auto backend_end = std::chrono::steady_clock::now();

    std::cout << std::setprecision(17);
    std::cout << "compute_runtime_s," << seconds_between(compute_start, compute_end) << '\n';
    std::cout << "backend_runtime_s," << seconds_between(backend_start, backend_end) << '\n';
    std::cout << "benchmark_steps," << cfg.steps << '\n';
    return EXIT_SUCCESS;
}

void run_normal(Particles& particles, const Config& cfg,
                AccelerationFunction compute_acceleration,
                const CpuDriverOptions& options) {
    auto trajectory = open_trajectory_file(cfg.output);
    auto energy_file = open_energy_file(cfg.energy_output);

    compute_acceleration(particles, cfg);
    require_particles_finite(particles, std::string(options.backend_name) + " initial acceleration");

    const double initial_kinetic = kinetic_energy(particles);
    const double initial_potential = potential_energy(particles, cfg);
    const double initial_energy = initial_kinetic + initial_potential;

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "initial_energy," << initial_energy << '\n';

    write_trajectory_frame(trajectory, 0, 0.0, particles);
    write_energy_frame(energy_file, 0, 0.0, initial_kinetic, initial_potential, initial_energy);

    AccelerationSnapshot old_accelerations(cfg.integrator == "verlet" ? particles.size() : 0);

    for (int step = 1; step <= cfg.steps; ++step) {
        advance_one_step(particles, cfg, old_accelerations, compute_acceleration);

        if (step % cfg.save_every == 0 || step == cfg.steps) {
            const double time = step * cfg.dt;
            require_particles_finite(
                particles, std::string(options.backend_name) + " output frame step " +
                               std::to_string(step));
            const double kinetic = kinetic_energy(particles);
            const double potential = potential_energy(particles, cfg);

            write_trajectory_frame(trajectory, step, time, particles);
            write_energy_frame(energy_file, step, time, kinetic, potential, initial_energy);
        }
    }

    require_particles_finite(particles, std::string(options.backend_name) + " final energy");
    const double final_energy = total_energy(particles, cfg);
    const double relative_error = relative_energy_error(initial_energy, final_energy);

    std::cout << "final_energy," << final_energy << '\n';
    std::cout << "relative_energy_error," << relative_error << '\n';
}

} // namespace

int run_cpu_driver(int argc, char** argv,
                   AccelerationFunction compute_acceleration,
                   const CpuDriverOptions& options) {
    try {
        const Config cfg = parse_args(argc, argv);
        if (cfg.benchmark && !options.supports_benchmark) {
            throw std::runtime_error(
                std::string("--benchmark is not supported by ") + options.backend_name);
        }

        Particles particles = initialize_particles(cfg);
        if (cfg.benchmark) {
            return run_benchmark(particles, cfg, compute_acceleration);
        }

        run_normal(particles, cfg, compute_acceleration, options);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
