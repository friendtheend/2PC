cmake_minimum_required(VERSION 3.18)
project(BMT_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# CUDA architectures (A6000: sm_86, H200: sm_90)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# libOTe path
set(LIBOTE_PATH "/usr/local" CACHE PATH "Path to libOTe installation")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBOTE_PATH}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
)

# Link directories
link_directories(
    ${LIBOTE_PATH}/lib
    ${CUDAToolkit_LIBRARY_DIR}
)

# ============================================================
# Offline Phase: GPU-accelerated BMT generation with OT
# ============================================================

# GPU kernels for offline phase
add_library(gpu_kernels_offline STATIC gpu_kernels.cu)
target_include_directories(gpu_kernels_offline PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(gpu_kernels_offline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Offline phase executable (requires MPI, libOTe)
add_executable(pcg_offline_gpu PCG_gpu.cpp)
target_compile_options(pcg_offline_gpu PRIVATE -fcoroutines)
target_link_libraries(pcg_offline_gpu 
    gpu_kernels_offline
    libOTe
    cryptoTools
    coproto
    sodium
    CUDA::cudart
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    pthread
)

# ============================================================
# Online Phase: GPU PRG expansion (standalone, no MPI/libOTe)
# ============================================================

# Online phase executable (pure CUDA, no dependencies)
add_executable(pcg_online_gpu PRG_GPU_Gen_BMT.cu)
target_link_libraries(pcg_online_gpu
    CUDA::cudart
)

# ============================================================
# Original CPU version for comparison
# ============================================================

# CPU offline phase
add_executable(pcg_offline_cpu PCG_multibit_parallel.cpp)
target_compile_options(pcg_offline_cpu PRIVATE -fcoroutines)
target_link_libraries(pcg_offline_cpu
    libOTe
    cryptoTools
    coproto
    sodium
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    pthread
)

# ============================================================
# Installation
# ============================================================

install(TARGETS pcg_offline_gpu pcg_online_gpu pcg_offline_cpu
        RUNTIME DESTINATION bin)

# ============================================================
# Summary
# ============================================================

message(STATUS "")
message(STATUS "=== BMT GPU Build Configuration ===")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "libOTe path: ${LIBOTE_PATH}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  pcg_offline_gpu  - GPU-accelerated offline phase (with OT)")
message(STATUS "  pcg_online_gpu   - GPU PRG expansion (standalone)")
message(STATUS "  pcg_offline_cpu  - CPU offline phase (comparison)")
message(STATUS "")
