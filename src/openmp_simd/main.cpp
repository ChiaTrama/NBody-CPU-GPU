#include "cpu_driver.hpp"
#include "physics.hpp"

int main(int argc, char** argv) {
    return run_cpu_driver(argc, argv, compute_acceleration_openmp_simd,
                          {"OpenMP SIMD", true});
}
