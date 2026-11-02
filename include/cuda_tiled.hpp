#ifndef NBODY_CUDA_TILED_HPP
#define NBODY_CUDA_TILED_HPP

#include <fstream>

#include "config.hpp"
#include "particles.hpp"

bool compute_acceleration_cuda_tiled_once(Particles& p, const Config& cfg);
bool run_cuda_tiled_simulation(Particles& p, const Config& cfg,
                               std::ofstream& trajectory, std::ofstream& energy_file);
bool run_cuda_tiled_benchmark(Particles& p, const Config& cfg);

#endif
