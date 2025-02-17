#include "bogo_sort_matv1.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <mma.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// #define DEBUG_PERMUTE
// #define DEBUG_PRINT 
// #define DEBUG_SORT
// #define DEBUG_RANDOM
#define PERMUTE_MATRIX_WIDTH 16
#define PERMUTATION_LENGTH 32
#define PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024 1024
#define PERMUTATION_VECTORS_32x16_FLAT_LENGTH_512 512
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

__global__ void bogo_sort_matv1(int* data, int size, int* output, int* block_permutation_counts) {

    __align__(256) extern __shared__ __half permutation_matrix[PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024 * 2]; // two 32x32 arrays
    __align__(256) extern __shared__ __half permutation_vectors[PERMUTATION_VECTORS_32x16_FLAT_LENGTH_512];
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
    #ifdef DEBUG_PERMUTE
    for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
        data[i * PERMUTATION_LENGTH + threadIdx.x] = __float2half(threadIdx.x);
    }
    __syncthreads();
    #endif
    
    // Initialize random states and generate random ints
    extern __shared__ curandStatePhilox4_32_10_t random_states[PERMUTATION_LENGTH];
    extern __shared__ int random_ints[PERMUTATION_LENGTH];

    //curand_init(unsigned long long seed,
    //  unsigned long long subsequence,
    //  unsigned long long offset,
    //  curandStatePhilox4_32_10_t *state)   
    curand_init(blockIdx.x, threadIdx.x, 0, &random_states[threadIdx.x]);
    __syncthreads();
    
    random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    bogo_sort_permutation_gen(temp_permutation, size, random_ints);

    for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
        random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
        __syncthreads();
        bogo_sort_permutation_gen(temp_permutation, size, random_ints);
        permutation_vectors[i * PERMUTATION_LENGTH + threadIdx.x] = __float2half(data[temp_permutation[threadIdx.x]]);
        __syncthreads();
    }

    random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    __syncthreads();
    bogo_sort_basis_gen(permutation_matrix, size, random_ints);
    __syncthreads();

    random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    __syncthreads();
    bogo_sort_basis_gen(permutation_matrix + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, size, random_ints);
    __syncthreads();

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("Before Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%.1f ", __half2float(permutation_vectors[i * PERMUTATION_LENGTH + j]));
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
    #endif

    extern __shared__ long permutations_tried;
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

    //we want to use wmma's native 16x16x16 matmul accumulate operations,
    //so each of our fragments needs to be 16x16x16
    //
    //we break down each 32x32 matrices into 16x16 tiles
    /*
                    32x32 Matrix Memory Layout
    
    +-------------------+-------------------+
    | ⬡                 | ⬢        ↑        |
    |<- NEXT_BLOCK=16 ->|          |        |
    |                   |  LOWER_ROW=16*32  |
    |                   |          |        |
    |                   |          ↓        |
    +-------------------+-------------------+
    | ⬣                 | ⬤                |
    |                   |                   |
    |                   |                   |
    |                   |                   |
    +-------------------+--------------------+

    Memory addresses for each quadrant:
    
    ⬡ ←── permutation_matrix
    ⬢ ←── permutation_matrix + NEXT_BLOCK
    ⬣ ←── permutation_matrix + LOWER_ROW  
    ⬤ ←── permutation_matrix + LOWER_ROW + NEXT_BLOCK

    Each 16x16 quadrant maps to tensor core fragments:
    ⬡ → mat_nw_frag     ⬢ → mat_ne_frag
    ⬣ → mat_sw_frag     ⬤ → mat_se_frag
    */

    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_ne_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_nw_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_se_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_sw_frag;

    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_ne_alt_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_nw_alt_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_se_alt_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_sw_alt_frag;

    wmma::fragment<wmma::matrix_b, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::col_major> vec_up_frag;
    wmma::fragment<wmma::matrix_b, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::col_major> vec_down_frag;

    wmma::fragment<wmma::accumulator, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half> prod_up_frag;
    wmma::fragment<wmma::accumulator, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half> prod_down_frag;

    wmma::load_matrix_sync(vec_up_frag, permutation_vectors, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(vec_down_frag, permutation_vectors + NEXT_BLOCK, PERMUTATION_LENGTH);

    wmma::load_matrix_sync(mat_nw_frag, permutation_matrix, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_ne_frag, permutation_matrix + NEXT_BLOCK, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_sw_frag, permutation_matrix + LOWER_ROW, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_se_frag, permutation_matrix + LOWER_ROW + NEXT_BLOCK, PERMUTATION_LENGTH);

    wmma::load_matrix_sync(mat_nw_alt_frag, permutation_matrix + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_ne_alt_frag, permutation_matrix + NEXT_BLOCK + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_sw_alt_frag, permutation_matrix + LOWER_ROW + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_se_alt_frag, permutation_matrix + LOWER_ROW + NEXT_BLOCK + PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024, PERMUTATION_LENGTH);
    __syncthreads();

    while (permutations_tried < TOTAL_PERMUTATIONS) {
        #ifdef DEBUG_RANDOM
        if (threadIdx.x == 0) {
            printf("Reached permutation %ld\n", permutations_tried);
            printf("switch_indexer:     ");
            for (int i = 31; i >= 0; i--) {
                printf("%d", (switch_indexer >> i) & 1);
                if (i % 8 == 0) printf(" ");
            }
            printf("\nswitch_incrementer: ");
            for (int i = 31; i >= 0; i--) {
                printf("%d", (switch_incrementer >> i) & 1);
                if (i % 8 == 0) printf(" ");
            }
            printf("\nswitch_multiplier:  ");
            for (int i = 31; i >= 0; i--) {
                printf("%d", (switch_multiplier >> i) & 1);
                if (i % 8 == 0) printf(" ");
            }
            printf("\nrandom_bit:         %s\n\n", random_bit ? "true" : "false");
        }
        #endif

            //get 16th-ish bit of switch_indexer, should be random bit
        if ((switch_indexer >> 16) & 1) {
            wmma::fill_fragment(prod_up_frag, 0.0f);
            wmma::mma_sync(prod_up_frag, mat_nw_frag, vec_up_frag, prod_up_frag);
            wmma::mma_sync(prod_up_frag, mat_ne_frag, vec_down_frag, prod_up_frag);
            wmma::store_matrix_sync(permutation_vectors, prod_up_frag, PERMUTATION_LENGTH, wmma::mem_col_major);

            wmma::fill_fragment(prod_down_frag, 0.0f);
            wmma::mma_sync(prod_down_frag, mat_sw_frag, vec_up_frag, prod_down_frag);
            wmma::mma_sync(prod_down_frag, mat_se_frag, vec_down_frag, prod_down_frag);
            wmma::store_matrix_sync(permutation_vectors + NEXT_BLOCK, prod_down_frag, PERMUTATION_LENGTH, wmma::mem_col_major);
        } else {
            wmma::fill_fragment(prod_up_frag, 0.0f);
            wmma::mma_sync(prod_up_frag, mat_nw_alt_frag, vec_up_frag, prod_up_frag);
            wmma::mma_sync(prod_up_frag, mat_ne_alt_frag, vec_down_frag, prod_up_frag);
            wmma::store_matrix_sync(permutation_vectors, prod_up_frag, PERMUTATION_LENGTH, wmma::mem_col_major);

            wmma::fill_fragment(prod_down_frag, 0.0f);
            wmma::mma_sync(prod_down_frag, mat_sw_alt_frag, vec_up_frag, prod_down_frag);
            wmma::mma_sync(prod_down_frag, mat_se_alt_frag, vec_down_frag, prod_down_frag);
            wmma::store_matrix_sync(permutation_vectors + NEXT_BLOCK, prod_down_frag, PERMUTATION_LENGTH, wmma::mem_col_major);
        }

        wmma::load_matrix_sync(vec_up_frag, permutation_vectors, PERMUTATION_LENGTH);
        wmma::load_matrix_sync(vec_down_frag, permutation_vectors + NEXT_BLOCK, PERMUTATION_LENGTH);

        if (threadIdx.x == 0) {
            // these operations are weird because they are intended to
            //      1) generate a mostly random new bit
            //      2) in as few clock cycles + hits to shared memory as possible
            // can definitely be improved
            // shift all bits in incrementer left by 1
            switch_incrementer = (switch_incrementer << 1) | (switch_incrementer >> 31);
            // do weird fucky-wucky operation to generate a psuedorandom new switch_indexer
            switch_indexer = switch_indexer * switch_multiplier + switch_incrementer;
            permutations_tried++;
            #ifdef DEBUG_RANDOM
            printf("Reached switch update at permutation %d with indexer=0x%08x, incrementer=0x%08x, multiplier=0x%08x\n", 
                   permutations_tried, switch_indexer, switch_incrementer, switch_multiplier);
            printf("indexer: ");
            for (int i = 31; i >= 0; i--) {
                printf("%d", (switch_indexer >> i) & 1);
                if (i % 8 == 0) printf(" ");
            }
            printf("\n");
            #endif
        }

        __syncthreads();

        for (int i = 0; i < 16; i++) {
            //address of i'th row of permutation_vectors
            //assuming row major layout, and zero based indexing
            verify_sort_matv1(permutation_vectors + i * 32, 32, &is_sorted);
            if (is_sorted) {
                output[threadIdx.x] = permutation_vectors[i * 32 + threadIdx.x];
                if (threadIdx.x == 0) {
                    printf("Block %d found sorted array after %ld iterations and %ld permutations\n", blockIdx.x, permutations_tried, permutations_tried * 16);
                    atomicCAS(&done, 0, 1);
                    block_permutation_counts[blockIdx.x] = permutations_tried;
                }
                return;
            }
        }

        if (permutations_tried % CHECK_DONE_PERMUTATIONS == 1) {
            if (threadIdx.x == 0) {
                local_done = atomicAnd(&done, 1);
            }
            __syncthreads();
            if (local_done) {
                if (blockIdx.x%1000 == 0 && threadIdx.x == 0) {
                    printf("Block %d: Loops: %ld; Permutations tried: %ld\n", blockIdx.x, permutations_tried, permutations_tried * 16);
                }
                if (threadIdx.x == 0) {
                    block_permutation_counts[blockIdx.x] = permutations_tried;
                }
                return;
            }
        }
    }

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("After Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%5.1f ", __half2float(permutation_vectors[i * PERMUTATION_LENGTH + j]));
            }
            printf("\n");
        }
        printf("\n");

        printf("Output data: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");

        printf("Total permutations tried: %ld\n", permutations_tried);
    }
    #endif



    return;
}

__device__ void verify_sort_matv1(__half* input, int size, bool* is_sorted) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *is_sorted = true;
    }
    __syncthreads();
    if (threadIdx.x < size - 1) {  // Don't check the last element since it has no right neighbor
        if (input[threadIdx.x] > input[threadIdx.x + 1]) {
            *is_sorted = false;
        }
    }
    __syncthreads();
}

__device__ void bogo_sort_basis_gen(__half* data, int size, int* random_ints) {
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
    #endif
    __syncthreads();

    // Copy random ints to sorted_ints initial parity section
    sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
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
        if (threadIdx.x % step_size == 0) {
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
        #endif
        __syncthreads();
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
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        if (sorted_ints[i + parity_shift(parity)] == my_value) {
            final_index = i;
            break;
        }
    }
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        data[threadIdx.x * PERMUTATION_LENGTH + i] = __float2half(i == final_index ? 1.0f : 0.0f);
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
    
    __syncthreads();
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", random_ints[i]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
    // Copy random ints to sorted_ints initial parity section
    sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
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
        if (threadIdx.x % step_size == 0) {
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
        #endif
        __syncthreads();
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
    int my_index = -1;
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        if (sorted_ints[i + parity_shift(parity)] == my_value) {
            my_index = i;
            break;
        }
    }
    data[threadIdx.x] = my_index;
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


dim3 KernelManagerBogoSortMatV1::calculateGrid(int n, int threadsPerBlock) {
    // return dim3((INNER_DIM + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(n);
}

float KernelManagerBogoSortMatV1::launchKernel(int* data, int* output) {
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

    const int threadsPerBlock = 32;
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
    bogo_sort_matv1<<<grid, block>>>(data, size, output, block_permutation_counts);
    // bogo_sort_matv1<<<1, block>>>(data, size, output);
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
