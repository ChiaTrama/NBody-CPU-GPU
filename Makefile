CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
OMPFLAGS := -fopenmp
SANITIZE_FLAGS := -std=c++17 -O1 -g -Wall -Wextra -pedantic -fsanitize=address,undefined -fno-omit-frame-pointer
CUDA_ARCH ?= sm_61
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -Iinclude

COMMON_SRCS := \
	src/common/particles.cpp \
	src/common/initialization.cpp \
	src/common/physics.cpp \
	src/common/integrators.cpp \
	src/common/io.cpp \
	src/common/cli.cpp

CPU_DRIVER_SRC := src/common/cpu_driver.cpp

SERIAL_MAIN := src/serial/main.cpp
OPENMP_MAIN := src/openmp/main.cpp
OPENMP_SIMD_MAIN := src/openmp_simd/main.cpp
CUDA_NAIVE_SRCS := \
	src/cuda_naive/main.cu \
	src/cuda_naive/cuda_naive.cu
CUDA_TILED_SRCS := \
	src/cuda_tiled/main.cu \
	src/cuda_tiled/cuda_tiled.cu

SERIAL_SRCS := $(SERIAL_MAIN) $(CPU_DRIVER_SRC) $(COMMON_SRCS)
OPENMP_SRCS := $(OPENMP_MAIN) $(CPU_DRIVER_SRC) $(COMMON_SRCS)
OPENMP_SIMD_SRCS := $(OPENMP_SIMD_MAIN) $(CPU_DRIVER_SRC) $(COMMON_SRCS)

BIN_DIR := bin
BUILD_DIR := build
SERIAL_TARGET := $(BIN_DIR)/nbody_serial
OPENMP_TARGET := $(BIN_DIR)/nbody_openmp
OPENMP_SIMD_TARGET := $(BIN_DIR)/nbody_openmp_simd
CUDA_NAIVE_TARGET := $(BIN_DIR)/nbody_cuda_naive
CUDA_TILED_TARGET := $(BIN_DIR)/nbody_cuda_tiled
SERIAL_SANITIZE_TARGET := $(BIN_DIR)/nbody_serial_sanitize
OPENMP_SANITIZE_TARGET := $(BIN_DIR)/nbody_openmp_sanitize
OPENMP_SIMD_SANITIZE_TARGET := $(BIN_DIR)/nbody_openmp_simd_sanitize

SERIAL_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/serial/%.o,$(SERIAL_SRCS))
OPENMP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/openmp/%.o,$(OPENMP_SRCS))
OPENMP_SIMD_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/openmp_simd/%.o,$(OPENMP_SIMD_SRCS))
CUDA_NAIVE_OBJS := $(patsubst %.cu,$(BUILD_DIR)/cuda_naive/%.o,$(CUDA_NAIVE_SRCS))
CUDA_NAIVE_COMMON_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/cuda_naive/%.o,$(COMMON_SRCS))
CUDA_TILED_OBJS := $(patsubst %.cu,$(BUILD_DIR)/cuda_tiled/%.o,$(CUDA_TILED_SRCS))
CUDA_TILED_COMMON_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/cuda_tiled/%.o,$(COMMON_SRCS))
SERIAL_SANITIZE_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/sanitize/serial/%.o,$(SERIAL_SRCS))
OPENMP_SANITIZE_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/sanitize/openmp/%.o,$(OPENMP_SRCS))
OPENMP_SIMD_SANITIZE_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/sanitize/openmp_simd/%.o,$(OPENMP_SIMD_SRCS))
OLD_OBJS := $(SERIAL_SRCS:.cpp=.o) $(OPENMP_SRCS:.cpp=.o) $(OPENMP_SIMD_SRCS:.cpp=.o) $(CUDA_NAIVE_SRCS:.cu=.o) $(CUDA_TILED_SRCS:.cu=.o)

.PHONY: all serial openmp openmp-simd cuda-naive cuda-tiled sanitize clean run run-openmp run-openmp-simd run-cuda-naive run-cuda-tiled run-sanitize

all: serial openmp openmp-simd

serial: $(SERIAL_TARGET)

openmp: $(OPENMP_TARGET)

openmp-simd: $(OPENMP_SIMD_TARGET)

cuda-naive: $(CUDA_NAIVE_TARGET)

cuda-tiled: $(CUDA_TILED_TARGET)

sanitize: $(SERIAL_SANITIZE_TARGET) $(OPENMP_SANITIZE_TARGET) $(OPENMP_SIMD_SANITIZE_TARGET)

$(SERIAL_TARGET): $(SERIAL_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OPENMP_TARGET): $(OPENMP_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(OPENMP_SIMD_TARGET): $(OPENMP_SIMD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(CUDA_NAIVE_TARGET): $(CUDA_NAIVE_OBJS) $(CUDA_NAIVE_COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_TILED_TARGET): $(CUDA_TILED_OBJS) $(CUDA_TILED_COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(SERIAL_SANITIZE_TARGET): $(SERIAL_SANITIZE_OBJS) | $(BIN_DIR)
	$(CXX) $(SANITIZE_FLAGS) $^ -o $@

$(OPENMP_SANITIZE_TARGET): $(OPENMP_SANITIZE_OBJS) | $(BIN_DIR)
	$(CXX) $(SANITIZE_FLAGS) $(OMPFLAGS) $^ -o $@

$(OPENMP_SIMD_SANITIZE_TARGET): $(OPENMP_SIMD_SANITIZE_OBJS) | $(BIN_DIR)
	$(CXX) $(SANITIZE_FLAGS) $(OMPFLAGS) $^ -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/serial/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/openmp/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/openmp_simd/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cuda_naive/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cuda_naive/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cuda_tiled/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cuda_tiled/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sanitize/serial/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(SANITIZE_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sanitize/openmp/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(SANITIZE_FLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sanitize/openmp_simd/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(SANITIZE_FLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

run: $(SERIAL_TARGET)
	./$(SERIAL_TARGET) --N 2 --steps 100 --dt 0.01 --epsilon 0.01 --init two-body --save-every 10 --output results/two_body.csv --energy-output results/two_body_energy.csv

run-openmp: $(OPENMP_TARGET)
	OMP_NUM_THREADS=4 ./$(OPENMP_TARGET) --N 2 --steps 100 --dt 0.01 --epsilon 0.01 --init two-body --save-every 10 --output results/two_body_openmp.csv --energy-output results/two_body_openmp_energy.csv

run-openmp-simd: $(OPENMP_SIMD_TARGET)
	OMP_NUM_THREADS=4 ./$(OPENMP_SIMD_TARGET) --N 2 --steps 100 --dt 0.01 --epsilon 0.01 --init two-body --save-every 10 --output results/two_body_openmp_simd.csv --energy-output results/two_body_openmp_simd_energy.csv

run-cuda-naive: $(CUDA_NAIVE_TARGET)
	./$(CUDA_NAIVE_TARGET)

run-cuda-tiled: $(CUDA_TILED_TARGET)
	./$(CUDA_TILED_TARGET) --N 2 --steps 10 --dt 0.01 --epsilon 0.01 --init two-body --integrator verlet --save-every 10 --block-size 256 --output results/two_body_cuda_tiled.csv --energy-output results/two_body_cuda_tiled_energy.csv

run-sanitize: sanitize
	mkdir -p results/sanitizer
	./$(SERIAL_SANITIZE_TARGET) --N 2 --steps 20 --dt 0.01 --epsilon 0.01 --init two-body --integrator euler --save-every 20 --output results/sanitizer/serial_euler.csv --energy-output results/sanitizer/serial_euler_energy.csv
	./$(SERIAL_SANITIZE_TARGET) --N 32 --steps 20 --dt 0.001 --epsilon 0.01 --init random --seed 12345 --integrator verlet --save-every 20 --output results/sanitizer/serial_verlet.csv --energy-output results/sanitizer/serial_verlet_energy.csv
	OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./$(OPENMP_SANITIZE_TARGET) --N 2 --steps 20 --dt 0.01 --epsilon 0.01 --init two-body --integrator euler --save-every 20 --output results/sanitizer/openmp_euler.csv --energy-output results/sanitizer/openmp_euler_energy.csv
	OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./$(OPENMP_SANITIZE_TARGET) --N 32 --steps 20 --dt 0.001 --epsilon 0.01 --init random --seed 12345 --integrator verlet --save-every 20 --output results/sanitizer/openmp_verlet.csv --energy-output results/sanitizer/openmp_verlet_energy.csv
	OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./$(OPENMP_SIMD_SANITIZE_TARGET) --N 2 --steps 20 --dt 0.01 --epsilon 0.01 --init two-body --integrator euler --save-every 20 --output results/sanitizer/openmp_simd_euler.csv --energy-output results/sanitizer/openmp_simd_euler_energy.csv
	OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./$(OPENMP_SIMD_SANITIZE_TARGET) --N 32 --steps 20 --dt 0.001 --epsilon 0.01 --init random --seed 12345 --integrator verlet --save-every 20 --output results/sanitizer/openmp_simd_verlet.csv --energy-output results/sanitizer/openmp_simd_verlet_energy.csv

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR) $(OLD_OBJS)
