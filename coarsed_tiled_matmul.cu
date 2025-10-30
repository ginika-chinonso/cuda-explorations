// Tiled Matrix Multiplication with Coarsing
#include <stdio.h>
#include <assert.h>
#include "utils.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void coarseTiledMatMulKernel(float *A, float *B, float *C, int m, int n, int k) {

    // Output tiles share the same row
    // Get the said row for the thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // Declare shared memory for A and B
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    // Allocate memory for the C values
    float c_values[COARSE_FACTOR];

    // Overwrite c_values to zero
    for (int i = 0; i < COARSE_FACTOR; i++) {
        c_values[i] = 0.0f;
    }

    // Get the expected number of phases
    int no_of_phases = n / (float) TILE_WIDTH;

    // Get the column value for the corresponding coarsing index
    // For a particular output, we need to get the corresponding phase
    for (int phase = 0; phase < no_of_phases; phase++) {
        
        // Store the thread's corresponding row value of A in shared memory
        shared_A[threadIdx.y][threadIdx.x] = A[row * n + phase * TILE_WIDTH + threadIdx.x];

        // Each kernel performs coarsing factor number of dot products for the output
        for (int coarse_index = 0; coarse_index < COARSE_FACTOR; coarse_index++) {
            
            // Get the required column
            int col = (blockIdx.x * TILE_WIDTH * coarse_index) + threadIdx.x;

            // Store the threads corresponding B value in shared memory
            shared_B[threadIdx.y][threadIdx.x] = B[k * (phase * TILE_WIDTH + threadIdx.y) + col];

            __syncthreads();

            // Based on the phase, we need to perform the dot product of the necessary elements
            for (int i = 0; i < TILE_WIDTH; ++i) {                
                // Accumulate the result in the corresponding c_values
                c_values[coarse_index] += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
            }
            __syncthreads();

        }
        
    }

    // Store result in back in C
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int col = (blockIdx.x * TILE_WIDTH * i) + threadIdx.x;
        C[row * k + col] = c_values[i];
    }

}

void coarsedMatMulDevice(float *A_h, float *B_h, float *C_h, int m, int n, int k) {
   
    // Declare device matrices
    float *A_d, *B_d, *C_d;

    // Allocate memory for the A,B and C matrices
    cudaMalloc(&A_d, m * n * sizeof(float));
    cudaMalloc(&B_d, n * k * sizeof(float));
    cudaMalloc(&C_d, m * k * sizeof(float));

    // Copy matrices A and B to device
    cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Instantiate grid and block dimensions
    dim3 grid_dim(ceil(k/float(TILE_WIDTH * COARSE_FACTOR)), ceil(m/float(TILE_WIDTH * COARSE_FACTOR)), 1);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call coarsed tiled mat mut kernel
    coarseTiledMatMulKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, m, n, k);

    // Copy result from device back to host
    cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main() {

    // Instantiate matrix sizes
    int m = 10;
    int n = 10;
    int k = 10;

    // Declare A,B and C matrices
    float *A, *B, *host_C, *device_C;

    // Allocate memory for A,B and C matrices
    A = (float *) malloc(m * n * sizeof(float));
    B = (float *) malloc(n * k * sizeof(float));
    host_C = (float *) malloc(m * k * sizeof(float));
    device_C = (float *) malloc(m * k * sizeof(float));

    // Instantiate A,B and C matrices
    init_matrix(A, m, n); 
    init_matrix(B, n, k); 
    init_with_zeros(host_C, m, k); 
    init_with_zeros(device_C, m, k);
    
    // Calculate matrix on host
    matMulHost(A, B, host_C, m, n, k);

    // Calculate matrix on device
    coarsedMatMulDevice(A, B, device_C, m, n, k);

    // Check that device result is equal to host result
    for (int i = 0; i < 10; ++i) {

        printf("Device: %f\n", device_C[i]);
        printf("Host: %f\n", host_C[i]);
        
        // assert(device_C[i] == host_C[i]);
    }

    // Free A,B and C matrices
    free(A);
    free(B);
    free(host_C);
    free(device_C);
}