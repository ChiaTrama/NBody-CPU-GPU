#ifndef NBODY_CUDA_NAIVE_HPP
#define NBODY_CUDA_NAIVE_HPP

#include <fstream>

#include "config.hpp"
#include "particles.hpp"

bool cuda_naive_smoke_test();
bool compute_acceleration_cuda_naive_once(Particles& p, const Config& cfg);
bool run_cuda_naive_simulation(Particles& p, const Config& cfg,
                               std::ofstream& trajectory, std::ofstream& energy_file);
bool run_cuda_naive_benchmark(Particles& p, const Config& cfg);

#endif
