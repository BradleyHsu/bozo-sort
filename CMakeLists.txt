cmake_minimum_required(VERSION 3.18)
project(bozo_sort CUDA CXX)

# Set CUDA architecture based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust this for your GPU
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable debug symbols and profiling info
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep")

# Main executable
# add_executable(bozo_sort
    # src/kernels/bogo_sort_matv1.cu
    # src/main.cu
    # src/kernels/basic_kernel.cu
    # src/kernels/bogo_sort.cu
    # src/kernels/bogo_sort_matv2.cu
# )

# Tests executable
add_executable(kernel_tests
    tests/kernel_tests.cu
    src/kernels/bogo_sort_matv2.cu
    # src/kernels/basic_kernel.cu
    # src/kernels/bogo_sort.cu
    # src/kernels/bogo_sort_matv1.cu
)
