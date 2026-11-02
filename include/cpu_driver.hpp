#ifndef NBODY_CPU_DRIVER_HPP
#define NBODY_CPU_DRIVER_HPP

#include "config.hpp"
#include "particles.hpp"

using AccelerationFunction = void (*)(Particles&, const Config&);

struct CpuDriverOptions {
    const char* backend_name;
    bool supports_benchmark;
};

int run_cpu_driver(int argc, char** argv,
                   AccelerationFunction compute_acceleration,
                   const CpuDriverOptions& options);

#endif
