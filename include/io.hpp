#ifndef NBODY_IO_HPP
#define NBODY_IO_HPP

#include <fstream>
#include <string>

#include "particles.hpp"

std::ofstream open_trajectory_file(const std::string& path);
void write_trajectory_frame(std::ofstream& out, int step, double time, const Particles& p);

std::ofstream open_energy_file(const std::string& path);
void write_energy_frame(std::ofstream& out, int step, double time,
                        double kinetic, double potential, double initial_total);

#endif
