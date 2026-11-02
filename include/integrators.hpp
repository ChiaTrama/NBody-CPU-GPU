#ifndef NBODY_INTEGRATORS_HPP
#define NBODY_INTEGRATORS_HPP

#include "config.hpp"
#include "particles.hpp"

#include <cstddef>
#include <vector>

struct AccelerationSnapshot {
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> az;

    explicit AccelerationSnapshot(std::size_t n = 0);
};

void step_euler(Particles& p, const Config& cfg);
void update_positions_verlet(Particles& p, const Config& cfg);
void save_accelerations(const Particles& p, AccelerationSnapshot& snapshot);
void update_velocities_verlet(Particles& p, const Config& cfg, const AccelerationSnapshot& old_accelerations);

#endif
