#ifndef NBODY_CONFIG_HPP
#define NBODY_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>

struct Config {
    std::size_t N = 100;
    int steps = 100;
    int save_every = 1;
    int block_size = 256;
    double dt = 0.01;
    double epsilon = 0.01;
    double G = 1.0;
    double box_size = 1000.0;
    double min_distance = 1.0;
    std::uint64_t seed = 12345;
    bool benchmark = false;
    std::string init = "random";
    std::string integrator = "euler";
    std::string output = "results/trajectory.csv";
    std::string energy_output = "results/energy.csv";
};

#endif
