#include "cuda_naive.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "io.hpp"
#include "physics.hpp"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__        \
                      << ": " << cudaGetErrorString(error) << '\n';             \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

struct DeviceParticles {
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    double* ax = nullptr;
    double* ay = nullptr;
    double* az = nullptr;
    double* next_ax = nullptr;
    double* next_ay = nullptr;
    double* next_az = nullptr;
    double* mass = nullptr;
    std::size_t n = 0;
};

__global__ void smoke_kernel(int* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *value = 1;
    }
}

__global__ void compute_acceleration_kernel(const double* __restrict__ x,
                                            const double* __restrict__ y,
                                            const double* __restrict__ z,
                                            const double* __restrict__ mass,
                                            double* __restrict__ ax,
                                            double* __restrict__ ay,
                                            double* __restrict__ az,
                                            std::size_t n, double g, double eps2) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];

    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    for (std::size_t j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }

        const double dx = x[j] - xi;
        const double dy = y[j] - yi;
        const double dz = z[j] - zi;
        const double r2 = dx * dx + dy * dy + dz * dz + eps2;
        const double inv_r = 1.0 / sqrt(r2);
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double factor = g * mass[j] * inv_r3;

        axi += factor * dx;
        ayi += factor * dy;
        azi += factor * dz;
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

__global__ void step_euler_kernel(double* x, double* y, double* z,
                                  double* vx, double* vy, double* vz,
                                  const double* ax, const double* ay, const double* az,
                                  std::size_t n, double dt) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;
}

__global__ void update_positions_verlet_kernel(double* x, double* y, double* z,
                                               const double* vx, const double* vy, const double* vz,
                                               const double* ax, const double* ay, const double* az,
                                               std::size_t n, double dt, double dt2) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    x[i] += vx[i] * dt + 0.5 * ax[i] * dt2;
    y[i] += vy[i] * dt + 0.5 * ay[i] * dt2;
    z[i] += vz[i] * dt + 0.5 * az[i] * dt2;
}

__global__ void update_velocities_verlet_kernel(double* vx, double* vy, double* vz,
                                                const double* old_ax, const double* old_ay, const double* old_az,
                                                const double* ax, const double* ay, const double* az,
                                                std::size_t n, double dt) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    vx[i] += 0.5 * (old_ax[i] + ax[i]) * dt;
    vy[i] += 0.5 * (old_ay[i] + ay[i]) * dt;
    vz[i] += 0.5 * (old_az[i] + az[i]) * dt;
}

int grid_size(std::size_t n, int block_size) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

double seconds_between(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

bool validate_launch_configuration(const Config& cfg, std::size_t n) {
    if (cfg.block_size <= 0) {
        std::cerr << "CUDA naive error: block size must be a positive integer\n";
        return false;
    }
    if (n == 0) {
        std::cerr << "CUDA naive error: particle count must be positive\n";
        return false;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

    if (cfg.block_size > properties.maxThreadsPerBlock) {
        std::cerr << "CUDA naive error: block size " << cfg.block_size
                  << " exceeds device maxThreadsPerBlock "
                  << properties.maxThreadsPerBlock << '\n';
        return false;
    }

    const std::size_t blocks =
        (n - 1) / static_cast<std::size_t>(cfg.block_size) + 1;
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        std::cerr << "CUDA naive error: required grid size " << blocks
                  << " exceeds device limit " << properties.maxGridSize[0] << '\n';
        return false;
    }
    return true;
}

bool allocate_device_particles(DeviceParticles& device, std::size_t n, bool with_scratch_accelerations) {
    const std::size_t bytes = n * sizeof(double);
    device.n = n;

    CUDA_CHECK(cudaMalloc(&device.x, bytes));
    CUDA_CHECK(cudaMalloc(&device.y, bytes));
    CUDA_CHECK(cudaMalloc(&device.z, bytes));
    CUDA_CHECK(cudaMalloc(&device.vx, bytes));
    CUDA_CHECK(cudaMalloc(&device.vy, bytes));
    CUDA_CHECK(cudaMalloc(&device.vz, bytes));
    CUDA_CHECK(cudaMalloc(&device.ax, bytes));
    CUDA_CHECK(cudaMalloc(&device.ay, bytes));
    CUDA_CHECK(cudaMalloc(&device.az, bytes));
    CUDA_CHECK(cudaMalloc(&device.mass, bytes));

    if (with_scratch_accelerations) {
        CUDA_CHECK(cudaMalloc(&device.next_ax, bytes));
        CUDA_CHECK(cudaMalloc(&device.next_ay, bytes));
        CUDA_CHECK(cudaMalloc(&device.next_az, bytes));
    }

    return true;
}

bool free_device_pointer(double*& pointer) {
    if (pointer == nullptr) {
        return true;
    }

    const cudaError_t error = cudaFree(pointer);
    pointer = nullptr;

    if (error != cudaSuccess) {
        std::cerr << "CUDA error while freeing device memory: "
                  << cudaGetErrorString(error) << '\n';
        return false;
    }

    return true;
}

bool free_device_particles(DeviceParticles& device) {
    bool ok = true;

    ok = free_device_pointer(device.x) && ok;
    ok = free_device_pointer(device.y) && ok;
    ok = free_device_pointer(device.z) && ok;
    ok = free_device_pointer(device.vx) && ok;
    ok = free_device_pointer(device.vy) && ok;
    ok = free_device_pointer(device.vz) && ok;
    ok = free_device_pointer(device.ax) && ok;
    ok = free_device_pointer(device.ay) && ok;
    ok = free_device_pointer(device.az) && ok;
    ok = free_device_pointer(device.next_ax) && ok;
    ok = free_device_pointer(device.next_ay) && ok;
    ok = free_device_pointer(device.next_az) && ok;
    ok = free_device_pointer(device.mass) && ok;

    device.n = 0;
    return ok;
}

bool create_cuda_events(cudaEvent_t& start, cudaEvent_t& stop) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    return true;
}

bool destroy_cuda_event(cudaEvent_t& event) {
    if (event == nullptr) {
        return true;
    }

    const cudaError_t error = cudaEventDestroy(event);
    event = nullptr;

    if (error != cudaSuccess) {
        std::cerr << "CUDA error while destroying event: "
                  << cudaGetErrorString(error) << '\n';
        return false;
    }

    return true;
}

bool destroy_cuda_events(cudaEvent_t& start, cudaEvent_t& stop) {
    bool ok = true;
    ok = destroy_cuda_event(start) && ok;
    ok = destroy_cuda_event(stop) && ok;
    return ok;
}

bool copy_host_to_device(const Particles& p, DeviceParticles& device) {
    const std::size_t bytes = p.size() * sizeof(double);

    CUDA_CHECK(cudaMemcpy(device.x, p.x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.y, p.y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.z, p.z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.vx, p.vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.vy, p.vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.vz, p.vz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.mass, p.mass.data(), bytes, cudaMemcpyHostToDevice));

    return true;
}

bool copy_device_to_host(const DeviceParticles& device, Particles& p) {
    const std::size_t bytes = p.size() * sizeof(double);

    CUDA_CHECK(cudaMemcpy(p.x.data(), device.x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.y.data(), device.y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.z.data(), device.z, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.vx.data(), device.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.vy.data(), device.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.vz.data(), device.vz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.ax.data(), device.ax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.ay.data(), device.ay, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p.az.data(), device.az, bytes, cudaMemcpyDeviceToHost));

    return true;
}

bool enqueue_compute_acceleration_to(DeviceParticles& device, const Config& cfg,
                                     double* ax, double* ay, double* az) {
    const double eps2 = cfg.epsilon * cfg.epsilon;

    compute_acceleration_kernel<<<grid_size(device.n, cfg.block_size), cfg.block_size>>>(
        device.x, device.y, device.z, device.mass,
        ax, ay, az,
        device.n, cfg.G, eps2);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

bool enqueue_compute_acceleration(DeviceParticles& device, const Config& cfg) {
    return enqueue_compute_acceleration_to(device, cfg, device.ax, device.ay, device.az);
}

bool enqueue_step_euler(DeviceParticles& device, const Config& cfg) {
    step_euler_kernel<<<grid_size(device.n, cfg.block_size), cfg.block_size>>>(
        device.x, device.y, device.z,
        device.vx, device.vy, device.vz,
        device.ax, device.ay, device.az,
        device.n, cfg.dt);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

bool enqueue_step_verlet(DeviceParticles& device, const Config& cfg) {
    if (device.next_ax == nullptr || device.next_ay == nullptr || device.next_az == nullptr) {
        std::cerr << "CUDA error: Verlet acceleration scratch buffers are not allocated\n";
        return false;
    }

    const double dt2 = cfg.dt * cfg.dt;

    update_positions_verlet_kernel<<<grid_size(device.n, cfg.block_size), cfg.block_size>>>(
        device.x, device.y, device.z,
        device.vx, device.vy, device.vz,
        device.ax, device.ay, device.az,
        device.n, cfg.dt, dt2);
    CUDA_CHECK(cudaGetLastError());

    if (!enqueue_compute_acceleration_to(device, cfg, device.next_ax, device.next_ay, device.next_az)) {
        return false;
    }

    update_velocities_verlet_kernel<<<grid_size(device.n, cfg.block_size), cfg.block_size>>>(
        device.vx, device.vy, device.vz,
        device.ax, device.ay, device.az,
        device.next_ax, device.next_ay, device.next_az,
        device.n, cfg.dt);
    CUDA_CHECK(cudaGetLastError());

    // Preserve the invariant: device.ax holds the current acceleration.
    std::swap(device.ax, device.next_ax);
    std::swap(device.ay, device.next_ay);
    std::swap(device.az, device.next_az);

    return true;
}

bool write_cuda_frame(DeviceParticles& device, Particles& p, const Config& cfg,
                      std::ofstream& trajectory, std::ofstream& energy_file,
                      int step, double initial_energy) {
    if (!copy_device_to_host(device, p)) {
        return false;
    }

    const double time = step * cfg.dt;
    require_particles_finite(p, "CUDA output frame step " + std::to_string(step));
    const double kinetic = kinetic_energy(p);
    const double potential = potential_energy(p, cfg);

    write_trajectory_frame(trajectory, step, time, p);
    write_energy_frame(energy_file, step, time, kinetic, potential, initial_energy);

    return true;
}

} // namespace

bool cuda_naive_smoke_test() {
    int* device_value = nullptr;
    int host_value = 0;

    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMemset(device_value, 0, sizeof(int)));

    smoke_kernel<<<1, 1>>>(device_value);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_value));

    return host_value == 1;
}

bool compute_acceleration_cuda_naive_once(Particles& p, const Config& cfg) {
    if (!validate_launch_configuration(cfg, p.size())) {
        return false;
    }

    DeviceParticles device;

    bool ok = allocate_device_particles(device, p.size(), false);

    if (ok) {
        ok = copy_host_to_device(p, device) &&
             enqueue_compute_acceleration(device, cfg) &&
             copy_device_to_host(device, p);
    }

    const bool cleanup_ok = free_device_particles(device);
    return ok && cleanup_ok;
}

bool run_cuda_naive_simulation(Particles& p, const Config& cfg,
                               std::ofstream& trajectory, std::ofstream& energy_file) {
    if (!validate_launch_configuration(cfg, p.size())) {
        return false;
    }

    DeviceParticles device;

    bool ok = allocate_device_particles(device, p.size(), cfg.integrator == "verlet") &&
              copy_host_to_device(p, device) &&
              enqueue_compute_acceleration(device, cfg);

    double initial_energy = 0.0;

    if (ok) {
        ok = copy_device_to_host(device, p);
    }

    if (ok) {
        require_particles_finite(p, "CUDA initial acceleration");

        const double initial_kinetic = kinetic_energy(p);
        const double initial_potential = potential_energy(p, cfg);
        initial_energy = initial_kinetic + initial_potential;

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        std::cout << "initial_energy," << initial_energy << '\n';

        write_trajectory_frame(trajectory, 0, 0.0, p);
        write_energy_frame(energy_file, 0, 0.0, initial_kinetic, initial_potential, initial_energy);

        for (int step = 1; step <= cfg.steps; ++step) {
            if (cfg.integrator == "euler") {
                ok = enqueue_step_euler(device, cfg) &&
                     enqueue_compute_acceleration(device, cfg);
            } else {
                ok = enqueue_step_verlet(device, cfg);
            }

            if (!ok) {
                break;
            }

            if (step % cfg.save_every == 0 || step == cfg.steps) {
                ok = write_cuda_frame(device, p, cfg, trajectory, energy_file, step, initial_energy);
                if (!ok) {
                    break;
                }
            }
        }
    }

    if (ok && cfg.steps == 0) {
        ok = copy_device_to_host(device, p);
    }

    if (ok) {
        require_particles_finite(p, "CUDA final energy");
        const double final_energy = total_energy(p, cfg);
        const double relative_error = relative_energy_error(initial_energy, final_energy);

        std::cout << "final_energy," << final_energy << '\n';
        std::cout << "relative_energy_error," << relative_error << '\n';
    }

    const bool cleanup_ok = free_device_particles(device);
    return ok && cleanup_ok;
}

bool run_cuda_naive_benchmark(Particles& p, const Config& cfg) {
    // CUDA backend runtime also includes launch validation.
    const auto backend_start = std::chrono::steady_clock::now();
    if (!validate_launch_configuration(cfg, p.size())) {
        return false;
    }

    DeviceParticles device;
    cudaEvent_t compute_start = nullptr;
    cudaEvent_t compute_stop = nullptr;
    float compute_runtime_ms = 0.0f;

    // Compute runtime covers the initial force evaluation and the time loop.
    // CUDA backend runtime also includes allocation, H2D copy, events, and cleanup.

    bool ok = allocate_device_particles(device, p.size(), cfg.integrator == "verlet") &&
              copy_host_to_device(p, device) &&
              create_cuda_events(compute_start, compute_stop);

    if (ok) {
        const cudaError_t error = cudaEventRecord(compute_start);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error while recording benchmark start event: "
                      << cudaGetErrorString(error) << '\n';
            ok = false;
        }
    }

    if (ok) {
        ok = enqueue_compute_acceleration(device, cfg);
    }

    for (int step = 1; ok && step <= cfg.steps; ++step) {
        if (cfg.integrator == "euler") {
            ok = enqueue_step_euler(device, cfg) &&
                 enqueue_compute_acceleration(device, cfg);
        } else {
            ok = enqueue_step_verlet(device, cfg);
        }
    }

    if (ok) {
        cudaError_t error = cudaEventRecord(compute_stop);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error while recording benchmark stop event: "
                      << cudaGetErrorString(error) << '\n';
            ok = false;
        }

        if (ok) {
            error = cudaEventSynchronize(compute_stop);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error while synchronizing benchmark stop event: "
                          << cudaGetErrorString(error) << '\n';
                ok = false;
            }
        }

        if (ok) {
            error = cudaEventElapsedTime(&compute_runtime_ms, compute_start, compute_stop);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error while reading benchmark elapsed time: "
                          << cudaGetErrorString(error) << '\n';
                ok = false;
            }
        }
    }

    const bool cleanup_ok = free_device_particles(device);
    const bool event_cleanup_ok = destroy_cuda_events(compute_start, compute_stop);
    const auto backend_end = std::chrono::steady_clock::now();

    if (ok) {
        std::cout << std::setprecision(17);
        std::cout << "compute_runtime_s," << compute_runtime_ms / 1000.0 << '\n';
        std::cout << "backend_runtime_s,"
                  << seconds_between(backend_start, backend_end) << '\n';
        std::cout << "benchmark_steps," << cfg.steps << '\n';
    }

    return ok && cleanup_ok && event_cleanup_ok;
}
