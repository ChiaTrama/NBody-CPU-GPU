#include "particles.hpp"

Particles::Particles(std::size_t n)
    : x(n), y(n), z(n),
      vx(n), vy(n), vz(n),
      ax(n), ay(n), az(n),
      mass(n) {}

std::size_t Particles::size() const {
    return mass.size();
}
