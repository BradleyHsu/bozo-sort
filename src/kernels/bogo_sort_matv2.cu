#include "bogo_sort_matv2.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <mma.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

// #define DEBUG_PERMUTE
#define DEBUG_PRINT 
// #define DEBUG_SORT
// #define DEBUG_RANDOM
#define PERMUTE_MATRIX_HEIGHT 64
#define PERMUTATION_LENGTH 32
#define PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024 1024
#define PERMUTATION_VECTORS_64x32_FLAT_LENGTH_2048 2048
#define LOWER_ROW 512
#define NEXT_BLOCK 16

#define OUTER_WIDTH 16
#define INNER_DIM 16
#define OUTER_HEIGHT 16

// #define TOTAL_PERMUTATIONS 1000000 
#define TOTAL_PERMUTATIONS 10000000000
// #define TOTAL_PERMUTATIONS 10
#define CHECK_DONE_PERMUTATIONS 1000000

using namespace nvcuda;
using namespace std;
using namespace cute;

__global__ void bogo_sort_matv2(int* data, int size, int* output, int* block_permutation_counts) {
    __align__(256) extern __shared__ uint8_t permutation_matrix[PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024]; // two 32x32 arrays
    __align__(256) extern __shared__ uint8_t formatted_permutation_matrix[PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024];
    __align__(256) extern __shared__ uint8_t permutation_vectors[PERMUTATION_VECTORS_64x32_FLAT_LENGTH_2048];
    extern __shared__ int temp_permutation[PERMUTATION_LENGTH];

    extern __device__ int done;
    extern __shared__ int local_done;
    extern __shared__ bool is_sorted;

    if (threadIdx.x == 0) {
        local_done = 0;
        is_sorted = false;
        if (blockIdx.x == 0) {
            done = 0;
        }
    }
    __syncthreads();
    
    
    // Initialize random states and generate random ints
    extern __shared__ curandStatePhilox4_32_10_t random_states[PERMUTATION_LENGTH];
    extern __shared__ int random_ints[PERMUTATION_LENGTH];
    extern __shared__ long permutations_tried;

    
        
        //curand_init(unsigned long long seed,
        //  unsigned long long subsequence,
        //  unsigned long long offset,
        //  curandStatePhilox4_32_10_t *state)

    if (threadIdx.x < 32) {
        curand_init(blockIdx.x, threadIdx.x, 0, &random_states[threadIdx.x]);
        random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    }
    bogo_sort_permutation_gen(temp_permutation, size, random_ints);

    for (int i = 0; i < PERMUTE_MATRIX_HEIGHT; i++) {
        if (threadIdx.x < 32) {
            random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
        }
        __syncthreads();
        bogo_sort_permutation_gen(temp_permutation, size, random_ints);
        if (threadIdx.x < 32) {
            permutation_vectors[i * PERMUTATION_LENGTH + threadIdx.x] = __float2half(data[temp_permutation[threadIdx.x]]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x < 32) {
        random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    }
    __syncthreads();
    bogo_sort_basis_gen(permutation_matrix, size, random_ints);
    __syncthreads();
    if (threadIdx.x < 32) {
        random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    }
    __syncthreads();
    bogo_sort_basis_gen(permutation_matrix + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, size, random_ints);
    __syncthreads();

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("Before Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_HEIGHT; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%3d ", permutation_vectors[i * PERMUTATION_LENGTH + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
    #endif

    extern __shared__ uint32_t switch_indexer;
    extern __shared__ uint32_t switch_multiplier;
    extern __shared__ uint32_t switch_incrementer;
    if (threadIdx.x == 0) {
        permutations_tried = 0;
        switch_indexer = curand(&random_states[threadIdx.x]);
        switch_incrementer = curand(&random_states[threadIdx.x]);
        switch_multiplier = switch_indexer;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        for (int i = 0; i < 32; i++) {
            permutation_matrix[threadIdx.x * 32 + i] = (threadIdx.x * 32 + i) % 256;
        }
    }
    __syncthreads();

    #define ROW_LENGTH 32
    #define CORE_MATRIX_HEIGHT 8
    #define CORE_MATRIX_ROW_LENGTH 16
    #define CORE_MATRICES_VERTICALLY_PER_WARP 2
    #define SHIFT_RIGHT 16
    #define SHIFT_DOWN 256
    #define WARP_SIZE 32

    #define LBO 128
    #define SBO 512

    // ((8, 2), (10, 4)):((10, 100), (1, 8))

    Layout trivial_layout   = make_layout(make_shape(Int<32>{}, Int<32>{}),
        make_stride(Int<32>{}, Int<1>{}));
    Layout formatted_layout = make_layout(make_shape(make_shape(Int<8>{}, Int<4>{}), make_shape(Int<16>{}, Int<2>{})),
        make_stride(make_stride(Int<16>{}, Int<LBO>{}), make_stride(Int<1>{}, Int<SBO>{})));
    Tensor trivial_tensor   = make_tensor(make_smem_ptr(permutation_matrix), trivial_layout);
    Tensor formatted_tensor = make_tensor(make_smem_ptr(formatted_permutation_matrix), formatted_layout);

    cute::copy(trivial_tensor, formatted_tensor);
    __syncthreads();



    // #ifdef DEBUG_PRINT
    // if (threadIdx.x == 0) {
    //     printf("Trivial layout : \n");
    //     print_layout(trivial_layout);
    //     printf("Formatted layout : \n");
    //     print_layout(formatted_layout);
    //     printf("Permutation Matrix:\n");
    //     for (int i = 0; i < 32; i++) {
    //         printf("  Row %2d: ", i);
    //         for (int j = 0; j < 32; j++) {
    //             printf("%3x ", permutation_matrix[i * 32 + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");

    //     printf("Formatted Permutation Matrix:\n");
    //     for (int i = 0; i < 32; i++) {
    //         printf("  Row %2d: ", i);
    //         for (int j = 0; j < 32; j++) {
    //             printf("%3x ", formatted_permutation_matrix[i * 32 + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     printf("\n");
    // }
    // __syncthreads();
    
    // __syncthreads();
    // #endif

    // #define ROW_LENGTH 8
    // #define CORE_MATRIX_HEIGHT 16
    // #define SHIFT_RIGHT 4
    // #define SHIFT_DOWN 64
    // #define WARP_SIZE 32
    //We have a 64 x 32 array of uint8s (num rows x bnum columns)
    //We split into 4 16 x 32 warps  
    //Each warp has 4  8 x 16 core matrices
    //
    //                                    32 columns
    //  +------------------------------------------------+
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+     Warp 0    | 16 rows
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+              |
    //  |------------------------------------------------|
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+     Warp 1    | 16 rows
    //  |+----------------+----------------+              | 64 rows
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+              |
    //  |------------------------------------------------|
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+     Warp 2    | 16 rows
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+              |
    //  |------------------------------------------------|
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+     Warp 3    | 16 rows
    //  |+----------------+----------------+              |
    //  ||    16x8       |     16x8       |              |
    //  |+----------------+----------------+              |
    //  +------------------------------------------------+

    //each thread gets a vector of 4 .b32 registers
    //each register is packed with 4 u8 ints
    //each thread loads 4 ints from each of the 4 core matrices, to load 16 ints in all
    //each register gets 4 ints from the same core matrix

    // in terms of 8-bit words
    

    // // in terms of 32-bit words
    // #define ROW_LENGTH 8
    // #define CORE_MATRIX_HEIGHT 16
    // #define SHIFT_RIGHT 4
    // #define SHIFT_DOWN 128
    // #define WARP_SIZE 32

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_address = warp_id * ROW_LENGTH * CORE_MATRIX_HEIGHT * 2; //two core matrixes vertically per warp

    // #ifdef DEBUG_PRINT
    // if (threadIdx.x == 0) {
    //     for (int i = 0; i < 4; i++) {
    //         printf("\nwarp address if warp_id =%d: %d \n", i, i * ROW_LENGTH * CORE_MATRIX_HEIGHT * 2);
    //     }
    // }
    // #endif 

    int global_idx = warp_id * ROW_LENGTH + (threadIdx.x % ROW_LENGTH);
    for (int i = 0; i < CORE_MATRIX_HEIGHT * 2; i++) {
        int offset = i * ROW_LENGTH;
        int val = (offset + global_idx) % 256;
        permutation_vectors[offset + global_idx] = __float2half(val);
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("In thread register access (thread 0):\n");
    //         for (int i = 0; i < PERMUTE_MATRIX_HEIGHT; i++) {
    //             printf("  Row %2d: ", i);
    //             for (int j = 0; j < PERMUTATION_LENGTH; j++) {
    //                 printf("%2x ", permutation_vectors[i * PERMUTATION_LENGTH + j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     printf("warp_id: %d\n", warp_id);
    //     printf("warp_address: %d\n", warp_address);
    // }
    // __syncthreads();

    // if (threadIdx.x == 127) {
    //     printf("In thread register access (thread 127):\n");
    //         for (int i = 0; i < PERMUTE_MATRIX_HEIGHT; i++) {
    //             printf("  Row %2d: ", i);
    //             for (int j = 0; j < PERMUTATION_LENGTH; j++) {
    //                 printf("%2x ", permutation_vectors[i * PERMUTATION_LENGTH + j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     printf("warp_id: %d\n", warp_id);
    //     printf("warp_address: %d\n", warp_address);
    // }

    //each thread loads one contiguous 32 bit segment containing 4 u8 ints into a register in array
    //the threads are distributed to addresses in the 8 x 16 u8 as follows (8 x 4 b32)
    // Thread distribution in 8x4 matrix (row major order):
    //
    //       C0   C1   C2   C3   C4   C5   C6   C7
    //      +----+----+----+----+----+----+----+----+
    //  R0  | T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3 |
    //      +----+----+----+----+----+----+----+----+
    //  R1  | T4 | T5 | T6 | T7 | T4 | T5 | T6 | T7 |
    //      +----+----+----+----+----+----+----+----+
    //  R2  | T8 | T9 |T10 |T11 | T8 | T9 |T10 |T11 |
    //      +----+----+----+----+----+----+----+----+
    //  R3  |T12 |T13 |T14 |T15 |T12 |T13 |T14 |T15 |
    //      +----+----+----+----+----+----+----+----+
    //  R4  |T16 |T17 |T18 |T19 |T16 |T17 |T18 |T19 |
    //      +----+----+----+----+----+----+----+----+
    //  R5  |T20 |T21 |T22 |T23 |T20 |T21 |T22 |T23 |
    //      +----+----+----+----+----+----+----+----+
    //  R6  |T24 |T25 |T26 |T27 |T24 |T25 |T26 |T27 |
    //      +----+----+----+----+----+----+----+----+
    //  R7  |T28 |T29 |T30 |T31 |T28 |T29 |T30 |T31 |
    //      +----+----+----+----+----+----+----+----+
    // Each of these rows is 4 b32s, but we want to index in u8s so we count 16 u8s per row
    int thread_core_matrix_row = (threadIdx.x % 32) / 4;
    int thread_core_matrix_col = (threadIdx.x % 32) % 4;
    int core_matrix_thread_location = thread_core_matrix_row * ROW_LENGTH + thread_core_matrix_col * 4;

    uint32_t bitf_0 = populate_bitf(permutation_vectors, warp_address + core_matrix_thread_location);
    uint32_t bitf_1 = populate_bitf(permutation_vectors, warp_address + core_matrix_thread_location + SHIFT_RIGHT);
    uint32_t bitf_2 = populate_bitf(permutation_vectors, warp_address + core_matrix_thread_location + SHIFT_DOWN);
    uint32_t bitf_3 = populate_bitf(permutation_vectors, warp_address + core_matrix_thread_location + SHIFT_DOWN + SHIFT_RIGHT);

    __align__(256) extern __shared__ uint8_t output_vector_alloc[128];
    long output_vector = __cvta_generic_to_shared(output_vector_alloc);
    output_vector_alloc[0] = 0;
    output_vector_alloc[1] = 0;
    output_vector_alloc[2] = 0;
    output_vector_alloc[3] = 0;

    uint64_t permutation_matrix_start_address = (uint64_t) formatted_permutation_matrix;
    uint64_t b_desc = 0;

    b_desc += matrix_descriptor_encode(permutation_matrix_start_address) << 0;
    b_desc += matrix_descriptor_encode(LBO) << 16;
    b_desc += matrix_descriptor_encode(SBO) << 32;

    // if (threadIdx.x == 0) {
    //     for (int i = sizeof(b_desc) * 8 - 1; i >= 0; i--) {
    //         printf("%d", (b_desc >> i) & 1);
    //         if (i % 8 == 0) {
    //             printf(" ");
    //         }
    //     }
    //     printf("\n");
    // }
    // printf("permutation_matrix_start_address: %p\n", (void*)permutation_matrix_start_address);

    int* output_array = 0;
    __syncthreads();
    asm volatile ("\n"
                ".reg    .s32 d<16>;\n"
                ".local  .s32 output[16];\n"
                // ".global.u64 ptr = generic(output);\n"
                "mov.u64 %0 ptr;\n"

                "wgmma.fence.sync.aligned;\n"
                "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.u8 {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15}, {%1, %2, %3, %4}, %5, 0;\n"
                "wgmma.commit_group.sync.aligned;\n"
                "wgmma.wait_group.sync.aligned 0;\n"

                "st.local.s32 [output +  0] d0 ;\n"
                "st.local.s32 [output +  4] d1 ;\n"
                "st.local.s32 [output +  8] d2 ;\n"
                "st.local.s32 [output + 12] d3 ;\n"
                "st.local.s32 [output + 16] d4 ;\n"
                "st.local.s32 [output + 20] d5 ;\n"
                "st.local.s32 [output + 24] d6 ;\n"
                "st.local.s32 [output + 28] d7 ;\n"
                "st.local.s32 [output + 32] d8 ;\n"
                "st.local.s32 [output + 36] d9 ;\n"
                "st.local.s32 [output + 40] d10;\n"
                "st.local.s32 [output + 44] d11;\n"
                "st.local.s32 [output + 48] d12;\n"
                "st.local.s32 [output + 52] d13;\n"
                "st.local.s32 [output + 56] d14;\n"
                "st.local.s32 [output + 60] d15;\n"

                :"=l"(output_array)
                :"r"(bitf_0), "r"(bitf_1), "r"(bitf_2), "r"(bitf_3), "l"(b_desc)
                : "memory");

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("Contents of output array:\n");
        for (int i = 0; i < 16; i++) {
            printf("%d ", output_array[i]);
        }
        printf("\n");
    }
    __syncthreads();
    #endif

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("After Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_HEIGHT; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%3d ", permutation_vectors[i * PERMUTATION_LENGTH + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
    #endif

    return;
}

__device__ static inline uint32_t populate_bitf(uint8_t* array, int address) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        int val = array[i + address];
        result += val << (i * 8);
    }
    return result;
}


__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }



__device__ void verify_sort_matv2(uint8_t* input, int size, bool* is_sorted) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *is_sorted = true;
    }
    __syncthreads();
    if (threadIdx.x < size - 1 && threadIdx.x < 32) {  // Don't check the last element since it has no right neighbor
        if (input[threadIdx.x] > input[threadIdx.x + 1]) {
            *is_sorted = false;
        }
    }
    __syncthreads();
}

__device__ void bogo_sort_basis_gen(uint8_t* data, int size, int* random_ints) {
    extern __shared__ int sorted_ints[PERMUTATION_LENGTH * 2];
    auto parity_shift = [](int p) { return p ? PERMUTATION_LENGTH : 0;};
    __syncthreads();
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", random_ints[i]);
        }
        printf("\n");
    }
    __syncthreads();
    #endif
    
    // Copy random ints to sorted_ints initial parity section
    if (threadIdx.x < 32) {
        sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
    }
    __syncthreads();

    extern __shared__ int step_size;
    extern __shared__ bool parity;
    extern __shared__ int merge_indices[PERMUTATION_LENGTH];
    if (threadIdx.x == 0) {
        step_size = 2;
        parity = false;
    }
    __syncthreads();

    while (step_size <= PERMUTATION_LENGTH) {
        if (threadIdx.x % step_size == 0 && threadIdx.x < 32) {
            int left_merge_counter = threadIdx.x;
            int right_merge_counter = threadIdx.x + 1;
            #ifdef DEBUG_SORT
            printf("Thread %d: left_merge_counter=%d, right_merge_counter=%d\n", 
                   threadIdx.x, left_merge_counter, right_merge_counter);
            #endif

            merge_indices[left_merge_counter] = 0;
            merge_indices[right_merge_counter] = 0;
            int print_thread_idx = 0;
            for (int i=0; i < step_size; i++) {
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("Thread %d, Step %d, Iteration %d:\n", threadIdx.x, step_size, i);
                }
                #endif
                
                int left_idx = threadIdx.x + merge_indices[left_merge_counter] + parity_shift(parity);
                int right_idx = threadIdx.x + merge_indices[right_merge_counter] + step_size/2 + parity_shift(parity);
                int dest_idx = threadIdx.x + i + parity_shift(!parity);
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  Left index: %d (value: %d)\n", left_idx, sorted_ints[left_idx]);
                    printf("  Right index: %d (value: %d)\n", right_idx, sorted_ints[right_idx]);
                    printf("  Destination index: %d\n", dest_idx);
                }
                #endif
                
                bool take_from_left = merge_indices[right_merge_counter] == step_size/2 ||
                    (sorted_ints[left_idx] < sorted_ints[right_idx] && 
                     merge_indices[left_merge_counter] < step_size/2);
                
                if (take_from_left) {
                    sorted_ints[dest_idx] = sorted_ints[left_idx];
                    merge_indices[left_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from left array\n");
                    }
                    #endif
                } else {
                    sorted_ints[dest_idx] = sorted_ints[right_idx]; 
                    merge_indices[right_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from right array\n");
                    }
                    #endif
                }
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  New value at destination: %d\n", sorted_ints[dest_idx]);
                    printf("  Left merge index: %d, Right merge index: %d\n\n", 
                           merge_indices[left_merge_counter], merge_indices[right_merge_counter]);
                    printf("  Current array state: ");
                    for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                        printf("%d ", sorted_ints[j + parity_shift(!parity)]);
                    }
                    printf("\n");
                }
                #endif
            }
        }
        if (threadIdx.x == 0) {
            step_size *= 2;
            parity = !parity;
        }
        __syncthreads();
        #ifdef DEBUG_SORT
        if (threadIdx.x == 0) {
            printf("Random ints after step %d: ", step_size);
            for (int i = 0; i < PERMUTATION_LENGTH; i++) {
                printf("%d ", sorted_ints[i + parity_shift(parity)]);
            }
            printf("\n");
        }
        __syncthreads();
        #endif
        
    }
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Sorted random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", sorted_ints[i + parity_shift(parity)]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
    int my_value = random_ints[threadIdx.x];
    int final_index = -1;
    if (threadIdx.x < 32) {
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            if (sorted_ints[i + parity_shift(parity)] == my_value) {
                final_index = i;
                break;
            }
        }
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            data[threadIdx.x * PERMUTATION_LENGTH + i] = (i == final_index) ? 1 : 0;
        }
    }
    __syncthreads();

    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Final indices matrix:\n");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("  ");
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%d ", data[i * PERMUTATION_LENGTH + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    #endif

    // data[threadIdx.x] = sorted_ints[threadIdx.x + parity_shift(!parity)];
    __syncthreads();
}

__device__ void bogo_sort_permutation_gen(int* data, int size, int* random_ints) {
    extern __shared__ int sorted_ints[64];
    auto parity_shift = [](int p) { return p ? PERMUTATION_LENGTH : 0;};
    
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", random_ints[i]);
        }
        printf("\n");
    }
    __syncthreads();
    #endif
    
    // Copy random ints to sorted_ints initial parity section
    sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
    extern __shared__ int step_size;
    extern __shared__ bool parity;
    extern __shared__ int merge_indices[PERMUTATION_LENGTH];
    if (threadIdx.x == 0) {
        step_size = 2;
        parity = false;
    }
    __syncthreads();
    while (step_size <= PERMUTATION_LENGTH) {
        if (threadIdx.x % step_size == 0 && threadIdx.x < 32) {
            int left_merge_counter = threadIdx.x;
            int right_merge_counter = threadIdx.x + 1;
            #ifdef DEBUG_SORT
            printf("Thread %d: left_merge_counter=%d, right_merge_counter=%d\n", 
                   threadIdx.x, left_merge_counter, right_merge_counter);
            #endif
            merge_indices[left_merge_counter] = 0;
            merge_indices[right_merge_counter] = 0;
            int print_thread_idx = 0;
            for (int i=0; i < step_size; i++) {
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("Thread %d, Step %d, Iteration %d:\n", threadIdx.x, step_size, i);
                }
                #endif
                
                int left_idx = threadIdx.x + merge_indices[left_merge_counter] + parity_shift(parity);
                int right_idx = threadIdx.x + merge_indices[right_merge_counter] + step_size/2 + parity_shift(parity);
                int dest_idx = threadIdx.x + i + parity_shift(!parity);
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  Left index: %d (value: %d)\n", left_idx, sorted_ints[left_idx]);
                    printf("  Right index: %d (value: %d)\n", right_idx, sorted_ints[right_idx]);
                    printf("  Destination index: %d\n", dest_idx);
                }
                #endif
                
                bool take_from_left = merge_indices[right_merge_counter] == step_size/2 ||
                    (sorted_ints[left_idx] < sorted_ints[right_idx] && 
                     merge_indices[left_merge_counter] < step_size/2);
                
                if (take_from_left) {
                    sorted_ints[dest_idx] = sorted_ints[left_idx];
                    merge_indices[left_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from left array\n");
                    }
                    #endif
                } else {
                    sorted_ints[dest_idx] = sorted_ints[right_idx]; 
                    merge_indices[right_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from right array\n");
                    }
                    #endif
                }
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  New value at destination: %d\n", sorted_ints[dest_idx]);
                    printf("  Left merge index: %d, Right merge index: %d\n\n", 
                           merge_indices[left_merge_counter], merge_indices[right_merge_counter]);
                    printf("  Current array state: ");
                    for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                        printf("%d ", sorted_ints[j + parity_shift(!parity)]);
                    }
                    printf("\n");
                }
                #endif
            }
        }
        if (threadIdx.x == 0) {
            step_size *= 2;
            parity = !parity;
        }
        __syncthreads();
        #ifdef DEBUG_SORT
        if (threadIdx.x == 0) {
            printf("Random ints after step %d: ", step_size);
            for (int i = 0; i < PERMUTATION_LENGTH; i++) {
                printf("%d ", sorted_ints[i + parity_shift(parity)]);
            }
            printf("\n");
        }
        __syncthreads();
        #endif
    }
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Sorted random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", sorted_ints[i + parity_shift(parity)]);
        }
        printf("\n");
    }
    __syncthreads();
    #endif
    int my_value = random_ints[threadIdx.x];
    int my_index = -1;
    if (threadIdx.x < 32) {
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            if (sorted_ints[i + parity_shift(parity)] == my_value) {
                my_index = i;
                break;
            }
        }
        data[threadIdx.x] = my_index;
    }
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Final sorted array: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
}


dim3 KernelManagerBogoSortMatV2::calculateGrid(int n, int threadsPerBlock) {
    // return dim3((INNER_DIM + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(n);
}

float KernelManagerBogoSortMatV2::launchKernel(int* data, int* output) {
    int size = 32;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int smCount = 0;
    for (int i = 0; i < deviceCount; i = i + 1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        smCount += prop.multiProcessorCount;
    }

    #ifdef DEBUG_PRINT
    printf("Number of SMs: %d\n", smCount);
    #endif

    int numBlocks = smCount * 64;
    printf("Number of blocks: %d\n", numBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = 128;
    dim3 grid = calculateGrid(numBlocks, threadsPerBlock);
    dim3 block(threadsPerBlock);
    #ifdef DEBUG_PRINT
    printf("Grid dimensions: %d x %d x %d\n", grid.x, grid.y, grid.z);
    #endif

    // Record start time
    cudaEventRecord(start);

    // Allocate device memory for block permutation counts
    int* block_permutation_counts;
    cudaMalloc(&block_permutation_counts, grid.x * sizeof(int));

    // Launch kernel
    // bogo_sort_matv2<<<grid, block>>>(data, size, output, block_permutation_counts);
    bogo_sort_matv2<<<1, block>>>(data, size, output, block_permutation_counts);
    // bogo_sort_matv2<<<1, block>>>(data, size, output);
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy block permutation counts back to host and print
    int* h_block_permutation_counts = new int[grid.x];
    cudaMemcpy(h_block_permutation_counts, block_permutation_counts, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nBlock permutation counts:\n");
    for (int i = 0; i < grid.x; i=i+1000) {
        if (h_block_permutation_counts[i] > 0) {
            printf("Block %d: %d permutations\n", i, h_block_permutation_counts[i]);
        }
    }

    // Calculate and print total permutations across all blocks
    long total_block_permutations = 0;
    for (int i = 0; i < grid.x; i++) {
        total_block_permutations += h_block_permutation_counts[i];
    }
    printf("\nTotal block cycles computed: %'ld\n", total_block_permutations);

    // Each block permutation generates 16 actual permutations
    long total_permutations = total_block_permutations * 16;
    printf("Total actual permutations tried: %'ld\n", total_permutations);

    // Calculate FLOPS:
    // For each permutation:
    // - 16x16 matrix multiplied by 16x1 vector requires:
    //   16 rows * 16 columns * 2 operations (multiply + add) = 512 FLOPs per permutation
    double total_tflops = (total_permutations * 512.0) / 1e12;
    printf("Total TFLOPs performed: %.2f teraflops\n", total_tflops);

    // Cleanup
    delete[] h_block_permutation_counts;
    cudaFree(block_permutation_counts);


    return milliseconds;
}