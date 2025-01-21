#include <cuda_runtime.h>
#include <stdio.h>
// #include "../src/kernels/basic_kernel.cuh"
// #include "../src/kernels/bogo_sort.cuh"
// #include "../src/kernels/bogo_sort_matv1.cuh"
#include "../src/kernels/bogo_sort_matv2.cuh"

#define COMPILE_ALL false

// Simple test framework
#define RUN_TEST(test_func) do { \
    printf("Running %s...\n", #test_func); \
    if (test_func()) { \
        printf("PASSED\n"); \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

#if COMPILE_ALL
bool test_vector_add() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate and initialize host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerVectorAdd::launchKernel(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success;
}

bool test_bogo_sort() {
    int N = 32;
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];

    int num_zeroes = 16;
    
    // Initialize input array
    for (int i = 0; i < 32; i++) {
        h_input[i] = i % 2;
        h_output[i] = 0;
    }
    h_input[31] = 2;
    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
    
    // Expected output array
    int expected[32];
    int counts[3] = {0, 0, 0};
    // Count occurrences of each value
    for (int i = 0; i < N; i++) {
        counts[h_input[i]]++;
    }
    // Fill expected array in sorted order
    int idx = 0;
    for (int val = 0; val < 3; val++) {
        for (int j = 0; j < counts[val]; j++) {
            expected[idx++] = val;
        }
    }
    // Print expected array
    printf("Expected sdfdfds array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", expected[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerBogoSort::launchKernel(d_input, d_output);
    cudaError_t err = cudaGetLastError();        // Get error code

    printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output array
    printf("sdfdOutput array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]); 
    }
    printf("\n");

    // Verify output matches expected array
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != expected[i]) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return success;

}

bool test_bogo_sort_3_symbol() {
    int N = 32;
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];

    int num_zeroes = 16;
    
    // Initialize input array
    for (int i = 0; i < 32; i++) {
        h_input[i] = i % 2;  // Alternates between 0 and 1
        h_output[i] = 0;
    }
    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Expected output array
    int expected[32];
    for (int i = 0; i < 32; i++) {
        // expected[i] = (i < num_zeroes) ? 0 : 1;
        
        expected[i] = (i < 32 - num_zeroes) ? 0 : 1;
    }
    // Print expected array
    printf("Expected array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", expected[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerBogoSort::launchKernel(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output array
    printf("Output array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]); 
    }
    printf("\n");

    // Verify output matches expected array
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != expected[i]) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return success;

}

bool test_bogo_sort_matv1_3_symbol() {
    int N = 32;
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];

    int num_zeroes = 16;
    int num_twos = 1;
    
    // Initialize input array
    // First fill with zeroes
    for (int i = 0; i < 32; i++) {
        h_input[i] = 0;
        h_output[i] = 0;
    }

    // Randomly scatter num_zeroes 1's throughout
    int ones_placed = 0;
    while (ones_placed + num_twos < num_zeroes) {
        int pos = rand() % 32 - num_twos;
        if (h_input[pos] == 0) {
            h_input[pos] = 1;
            ones_placed++;
        }
    }

    for (int i = N - num_twos; i < N; i++) {
        h_input[i] = 2;
    }

    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Expected output array
    int expected[32];
    for (int i = 0; i < 32; i++) {
        // expected[i] = (i < num_zeroes) ? 0 : 1;
        
        expected[i] = (i < 32 - num_zeroes) ? 0 : 1;
    }

    for (int i = N - num_twos; i < N; i++) {
        expected[i] = 2;
    }

    // Print expected array
    printf("Expected array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", expected[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerBogoSortMatV1::launchKernel(d_input, d_output);

    cudaError_t err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output array
    printf("Output array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]); 
    }
    printf("\n");

    // Verify output matches expected array
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != expected[i]) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return success;

}
#endif

bool test_bogo_sort_matv2_3_symbol() {
    int N = 32;
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];

    int num_zeroes = 16;
    int num_twos = 1;
    
    // Initialize input array
    // First fill with zeroes
    for (int i = 0; i < 32; i++) {
        h_input[i] = 0;
        h_output[i] = 0;
    }

    // Randomly scatter num_zeroes 1's throughout
    int ones_placed = 0;
    while (ones_placed + num_twos < num_zeroes) {
        int pos = rand() % 32 - num_twos;
        if (h_input[pos] == 0) {
            h_input[pos] = 1;
            ones_placed++;
        }
    }

    for (int i = N - num_twos; i < N; i++) {
        h_input[i] = 2;
    }

    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Expected output array
    int expected[32];
    for (int i = 0; i < 32; i++) {
        // expected[i] = (i < num_zeroes) ? 0 : 1;
        
        expected[i] = (i < 32 - num_zeroes) ? 0 : 1;
    }

    for (int i = N - num_twos; i < N; i++) {
        expected[i] = 2;
    }

    // Print expected array
    printf("Expected array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", expected[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerBogoSortMatV2::launchKernel(d_input, d_output);

    cudaError_t err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output array
    printf("Output array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]); 
    }
    printf("\n");

    // Verify output matches expected array
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != expected[i]) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return success;

}
int main() {
    RUN_TEST(test_bogo_sort_matv2_3_symbol);
    return 0;
}
