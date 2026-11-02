#include "integrators.hpp"

#include <algorithm>
#include <stdexcept>

AccelerationSnapshot::AccelerationSnapshot(std::size_t n)
    : ax(n), ay(n), az(n) {}

void step_euler(Particles& p, const Config& cfg) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.x[i] += p.vx[i] * cfg.dt;
        p.y[i] += p.vy[i] * cfg.dt;
        p.z[i] += p.vz[i] * cfg.dt;

        p.vx[i] += p.ax[i] * cfg.dt;
        p.vy[i] += p.ay[i] * cfg.dt;
        p.vz[i] += p.az[i] * cfg.dt;
    }
}

void update_positions_verlet(Particles& p, const Config& cfg) {
    const double dt2 = cfg.dt * cfg.dt;

    for (std::size_t i = 0; i < p.size(); ++i) {
        p.x[i] += p.vx[i] * cfg.dt + 0.5 * p.ax[i] * dt2;
        p.y[i] += p.vy[i] * cfg.dt + 0.5 * p.ay[i] * dt2;
        p.z[i] += p.vz[i] * cfg.dt + 0.5 * p.az[i] * dt2;
    }
}

void save_accelerations(const Particles& p, AccelerationSnapshot& snapshot) {
    if (snapshot.ax.size() != p.size() ||
        snapshot.ay.size() != p.size() ||
        snapshot.az.size() != p.size()) {
        throw std::runtime_error("acceleration snapshot size does not match particle count");
    }

    std::copy(p.ax.begin(), p.ax.end(), snapshot.ax.begin());
    std::copy(p.ay.begin(), p.ay.end(), snapshot.ay.begin());
    std::copy(p.az.begin(), p.az.end(), snapshot.az.begin());
}

void update_velocities_verlet(Particles& p, const Config& cfg, const AccelerationSnapshot& old_accelerations) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.vx[i] += 0.5 * (old_accelerations.ax[i] + p.ax[i]) * cfg.dt;
        p.vy[i] += 0.5 * (old_accelerations.ay[i] + p.ay[i]) * cfg.dt;
        p.vz[i] += 0.5 * (old_accelerations.az[i] + p.az[i]) * cfg.dt;
    }
}
