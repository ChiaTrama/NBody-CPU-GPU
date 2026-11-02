#ifndef NBODY_PARTICLES_HPP
#define NBODY_PARTICLES_HPP

#include <cstddef>
#include <vector>

struct Particles {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> az;
    std::vector<double> mass;

    explicit Particles(std::size_t n = 0);
    std::size_t size() const;
};

#endif
