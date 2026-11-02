#include "io.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "physics.hpp"

namespace {

void require_finite_value(double value, const std::string& name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite energy value: " + name);
    }
}

} // namespace

std::ofstream open_trajectory_file(const std::string& path) {
    const std::filesystem::path output_path(path);

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not open output file: " + path);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "step,time,particle,x,y,z,vx,vy,vz,mass\n";
    return out;
}

std::ofstream open_energy_file(const std::string& path) {
    const std::filesystem::path output_path(path);

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not open energy file: " + path);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "step,time,kinetic,potential,total,relative_error\n";
    return out;
}

void write_trajectory_frame(std::ofstream& out, int step, double time, const Particles& p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        out << step << ','
            << time << ','
            << i << ','
            << p.x[i] << ','
            << p.y[i] << ','
            << p.z[i] << ','
            << p.vx[i] << ','
            << p.vy[i] << ','
            << p.vz[i] << ','
            << p.mass[i] << '\n';
    }
}

void write_energy_frame(std::ofstream& out, int step, double time,
                        double kinetic, double potential, double initial_total) {
    const double total = kinetic + potential;

    require_finite_value(kinetic, "kinetic");
    require_finite_value(potential, "potential");
    require_finite_value(total, "total");
    require_finite_value(initial_total, "initial_total");

    const double relative_error = relative_energy_error(initial_total, total);

    out << step << ','
        << time << ','
        << kinetic << ','
        << potential << ','
        << total << ','
        << relative_error << '\n';
}
