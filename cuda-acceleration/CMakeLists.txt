cmake_minimum_required(VERSION 3.18)
project(portalis_cuda_acceleration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compilation options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# CUDA kernel library
add_library(portalis_cuda_kernels SHARED
    kernels/ast_parser.cu
    kernels/embedding_generator.cu
    kernels/verification_kernels.cu
    kernels/memory_manager.cu
    kernels/utils.cu
)

target_link_libraries(portalis_cuda_kernels
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

# CPU fallback library
add_library(portalis_cpu_fallback SHARED
    fallbacks/cpu_ast_parser.cpp
    fallbacks/cpu_embedding_generator.cpp
    fallbacks/cpu_verification.cpp
)

# Python bindings
add_subdirectory(bindings/python)

# Rust bindings
add_subdirectory(bindings/rust)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Installation
install(TARGETS portalis_cuda_kernels portalis_cpu_fallback
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY kernels/ DESTINATION include/portalis
    FILES_MATCHING PATTERN "*.cuh"
)
