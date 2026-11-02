#include "physics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool finite_array(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
    });
}

} // namespace

void compute_acceleration_serial(Particles& p, const Config& cfg) {
    const std::size_t n = p.size();
    const double eps2 = cfg.epsilon * cfg.epsilon;

    // Directed all-pairs formulation: each i accumulates its own acceleration.
    // This avoids concurrent writes to a_j in future OpenMP/CUDA versions,
    // at the cost of computing both i<-j and j<-i contributions.
    for (std::size_t i = 0; i < n; ++i) {
        double axi = 0.0;
        double ayi = 0.0;
        double azi = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }

            const double dx = p.x[j] - p.x[i];
            const double dy = p.y[j] - p.y[i];
            const double dz = p.z[j] - p.z[i];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            // Compute 1 / (r^2 + eps^2)^(3/2) without std::pow for a cheaper inner loop.
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double factor = cfg.G * p.mass[j] * inv_r3;

            axi += factor * dx;
            ayi += factor * dy;
            azi += factor * dz;
        }

        p.ax[i] = axi;
        p.ay[i] = ayi;
        p.az[i] = azi;
    }
}

void compute_acceleration_openmp(Particles& p, const Config& cfg) {
    const std::size_t n = p.size();
    const double eps2 = cfg.epsilon * cfg.epsilon;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        double axi = 0.0;
        double ayi = 0.0;
        double azi = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }

            const double dx = p.x[j] - p.x[i];
            const double dy = p.y[j] - p.y[i];
            const double dz = p.z[j] - p.z[i];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double factor = cfg.G * p.mass[j] * inv_r3;

            axi += factor * dx;
            ayi += factor * dy;
            azi += factor * dz;
        }

        p.ax[i] = axi;
        p.ay[i] = ayi;
        p.az[i] = azi;
    }
}

void compute_acceleration_openmp_simd(Particles& p, const Config& cfg) {
    const std::size_t n = p.size();
    const double eps2 = cfg.epsilon * cfg.epsilon;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        double axi = 0.0;
        double ayi = 0.0;
        double azi = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:axi, ayi, azi)
#endif
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) {
                const double dx = p.x[j] - p.x[i];
                const double dy = p.y[j] - p.y[i];
                const double dz = p.z[j] - p.z[i];
                const double r2 = dx * dx + dy * dy + dz * dz + eps2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double factor = cfg.G * p.mass[j] * inv_r3;

                axi += factor * dx;
                ayi += factor * dy;
                azi += factor * dz;
            }
        }

        p.ax[i] = axi;
        p.ay[i] = ayi;
        p.az[i] = azi;
    }
}

double kinetic_energy(const Particles& p) {
    double energy = 0.0;

    for (std::size_t i = 0; i < p.size(); ++i) {
        const double v2 = p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i];
        energy += 0.5 * p.mass[i] * v2;
    }

    return energy;
}

double potential_energy(const Particles& p, const Config& cfg) {
    double energy = 0.0;
    const double eps2 = cfg.epsilon * cfg.epsilon;

    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = i + 1; j < p.size(); ++j) {
            const double dx = p.x[j] - p.x[i];
            const double dy = p.y[j] - p.y[i];
            const double dz = p.z[j] - p.z[i];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            energy -= cfg.G * p.mass[i] * p.mass[j] / std::sqrt(r2);
        }
    }

    return energy;
}

double total_energy(const Particles& p, const Config& cfg) {
    return kinetic_energy(p) + potential_energy(p, cfg);
}

double relative_energy_error(double initial_energy, double current_energy) {
    if (!std::isfinite(initial_energy) || !std::isfinite(current_energy)) {
        throw std::runtime_error("relative energy error requires finite energies");
    }

    const double scale = std::max({1.0, std::abs(initial_energy), std::abs(current_energy)});
    const double threshold = std::numeric_limits<double>::epsilon() * scale;

    if (std::abs(initial_energy) <= threshold) {
        throw std::runtime_error("relative energy error is undefined: initial energy is too close to zero");
    }

    return std::abs(current_energy - initial_energy) / std::abs(initial_energy);
}

bool particles_are_finite(const Particles& p) {
    return finite_array(p.x) &&
           finite_array(p.y) &&
           finite_array(p.z) &&
           finite_array(p.vx) &&
           finite_array(p.vy) &&
           finite_array(p.vz) &&
           finite_array(p.ax) &&
           finite_array(p.ay) &&
           finite_array(p.az) &&
           finite_array(p.mass);
}

void require_particles_finite(const Particles& p, const std::string& context) {
    if (!particles_are_finite(p)) {
        throw std::runtime_error("non-finite particle state detected: " + context);
    }
}
