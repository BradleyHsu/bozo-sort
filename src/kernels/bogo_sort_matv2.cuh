#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Kernel declaration
__global__ void bogo_sort_matv2(int* data, int size, int* output, int* block_permutation_counts);
__device__ void verify_sort_matv2(uint8_t* input, int size, bool* is_sorted);
__device__ void bogo_sort_basis_gen(uint8_t* data, int size, int* random_ints);
__device__ void bogo_sort_permutation_gen(int* data, int size, int* random_ints);

// Wrapper class for kernel management and timing
class KernelManagerBogoSortMatV2 {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock);

    // Launch kernel with timing
    static float launchKernel(int* data, int* output);
};
