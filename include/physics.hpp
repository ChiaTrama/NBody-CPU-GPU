#ifndef NBODY_PHYSICS_HPP
#define NBODY_PHYSICS_HPP

#include "config.hpp"
#include "particles.hpp"

#include <string>

void compute_acceleration_serial(Particles& p, const Config& cfg);
void compute_acceleration_openmp(Particles& p, const Config& cfg);
void compute_acceleration_openmp_simd(Particles& p, const Config& cfg);
double kinetic_energy(const Particles& p);
double potential_energy(const Particles& p, const Config& cfg);
double total_energy(const Particles& p, const Config& cfg);
double relative_energy_error(double initial_energy, double current_energy);
bool particles_are_finite(const Particles& p);
void require_particles_finite(const Particles& p, const std::string& context);

#endif
