#pragma once
#include <cuda_runtime.h>

// Kernel declaration
__global__ void bogo_sort_matv1(int* data, int size, int* output);
__device__ void verify_sort_matv1(int* input, int size, bool* is_sorted);
__device__ void bogo_sort_basis_gen(int* data, int size, int* random_ints);
__device__ void bogo_sort_permutation_gen(int* data, int size, int* random_ints);

// Wrapper class for kernel management and timing
class KernelManagerBogoSortMatV1 {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock = 256);

    // Launch kernel with timing
    static float launchKernel(int* data, int* output);
};
