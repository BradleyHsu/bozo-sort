#pragma once
#include <cuda_runtime.h>

// Kernel declaration
__global__ void bogo_sort(int* data, int size, int* output);

// Wrapper class for kernel management and timing
class KernelManagerBogoSort {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock = 256);

    // Launch kernel with timing
    static float launchKernel(int* data, int size, int* output);
};