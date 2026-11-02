#include "initialization.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace {

double squared_distance(const Particles& p, std::size_t i, double x, double y, double z) {
    const double dx = p.x[i] - x;
    const double dy = p.y[i] - y;
    const double dz = p.z[i] - z;
    return dx * dx + dy * dy + dz * dz;
}

// Rejection sampling avoids initially overlapping particles, which would
// produce very large softened forces and possible numerical instability.
void initialize_random(Particles& p, const Config& cfg) {
    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> pos_dist(0.0, cfg.box_size);
    std::uniform_real_distribution<double> vel_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> mass_dist(1.0, 10.0);

    const double min_dist2 = cfg.min_distance * cfg.min_distance;
    const int max_attempts = 10000;

    for (std::size_t i = 0; i < p.size(); ++i) {
        bool accepted = false;

        for (int attempt = 0; attempt < max_attempts && !accepted; ++attempt) {
            const double x = pos_dist(rng);
            const double y = pos_dist(rng);
            const double z = pos_dist(rng);

            accepted = true;
            for (std::size_t j = 0; j < i; ++j) {
                if (squared_distance(p, j, x, y, z) < min_dist2) {
                    accepted = false;
                    break;
                }
            }

            if (accepted) {
                p.x[i] = x;
                p.y[i] = y;
                p.z[i] = z;
            }
        }

        if (!accepted) {
            throw std::runtime_error("could not place particles with the requested minimum distance");
        }

        p.vx[i] = vel_dist(rng);
        p.vy[i] = vel_dist(rng);
        p.vz[i] = vel_dist(rng);
        p.mass[i] = mass_dist(rng);
    }
}

void initialize_two_body(Particles& p) {
    if (p.size() != 2) {
        throw std::runtime_error("--init two-body requires --N 2");
    }

    p.mass[0] = 1.0;
    p.x[0] = 0.0;
    p.y[0] = 0.0;
    p.z[0] = 0.0;
    p.vx[0] = 0.0;
    p.vy[0] = 0.0;
    p.vz[0] = 0.0;

    p.mass[1] = 1.0e-3;
    p.x[1] = 1.0;
    p.y[1] = 0.0;
    p.z[1] = 0.0;
    p.vx[1] = 0.0;
    p.vy[1] = 1.0;
    p.vz[1] = 0.0;
}

void initialize_two_body_barycentric(Particles& p, const Config& cfg) {
    if (p.size() != 2) {
        throw std::runtime_error("--init two-body-barycentric requires --N 2");
    }

    constexpr double m1 = 1.0;
    constexpr double m2 = 1.0e-3;
    constexpr double separation = 1.0;
    const double total_mass = m1 + m2;
    const double softened_radius2 = separation * separation + cfg.epsilon * cfg.epsilon;
    const double omega = std::sqrt(
        cfg.G * total_mass / std::pow(softened_radius2, 1.5));

    p.mass[0] = m1;
    p.x[0] = -m2 * separation / total_mass;
    p.y[0] = 0.0;
    p.z[0] = 0.0;
    p.vx[0] = 0.0;
    p.vy[0] = omega * p.x[0];
    p.vz[0] = 0.0;

    p.mass[1] = m2;
    p.x[1] = m1 * separation / total_mass;
    p.y[1] = 0.0;
    p.z[1] = 0.0;
    p.vx[1] = 0.0;
    p.vy[1] = omega * p.x[1];
    p.vz[1] = 0.0;
}

void initialize_two_body_elliptic(Particles& p, const Config& cfg) {
    if (p.size() != 2) {
        throw std::runtime_error("--init two-body-elliptic requires --N 2");
    }

    constexpr double m0 = 1.0;
    constexpr double m1 = 1.0;
    constexpr double semi_major_axis = 1.0;
    constexpr double eccentricity = 0.5;
    const double total_mass = m0 + m1;
    const double periapsis = semi_major_axis * (1.0 - eccentricity);
    const double periapsis_speed = std::sqrt(
        cfg.G * total_mass * (1.0 + eccentricity) /
        (semi_major_axis * (1.0 - eccentricity)));

    p.mass[0] = m0;
    p.x[0] = -m1 * periapsis / total_mass;
    p.y[0] = 0.0;
    p.z[0] = 0.0;
    p.vx[0] = 0.0;
    p.vy[0] = -m1 * periapsis_speed / total_mass;
    p.vz[0] = 0.0;

    p.mass[1] = m1;
    p.x[1] = m0 * periapsis / total_mass;
    p.y[1] = 0.0;
    p.z[1] = 0.0;
    p.vx[1] = 0.0;
    p.vy[1] = m0 * periapsis_speed / total_mass;
    p.vz[1] = 0.0;
}

} // namespace

Particles initialize_particles(const Config& cfg) {
    Particles p(cfg.N);

    if (cfg.init == "random") {
        initialize_random(p, cfg);
    } else if (cfg.init == "two-body") {
        initialize_two_body(p);
    } else if (cfg.init == "two-body-barycentric") {
        initialize_two_body_barycentric(p, cfg);
    } else if (cfg.init == "two-body-elliptic") {
        initialize_two_body_elliptic(p, cfg);
    } else {
        throw std::runtime_error("unknown initializer: " + cfg.init);
    }

    return p;
}
