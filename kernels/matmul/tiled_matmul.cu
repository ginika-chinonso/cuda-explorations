#include <stdio.h>
#include "../../helpers/utils.h"

// Define tile width
#define Tile_width 16

// Tiled Matrix Multiplication Kernel
__global__ void tiledMatMulKernel(float *A, float *B, float *C, int m, int n, int k) {
    
    // Get row and column global index for C indexing
    // This assumes Tile_width == BlockDim.x == BlockDim.y
    int row = blockIdx.y * Tile_width + threadIdx.y;
    int col = blockIdx.x * Tile_width + threadIdx.x;

    // Load value to shared memory
    __shared__ float A_sh[Tile_width][Tile_width];
    __shared__ float B_sh[Tile_width][Tile_width];

    // Initialize result
    float res = 0;

    // Loop for the required phase
    for (int phase = 0; phase < n / Tile_width; ++phase) {

        // Load value to shared memory
        A_sh[threadIdx.y][threadIdx.x] = A[row * n + phase * Tile_width + threadIdx.x];
        B_sh[threadIdx.y][threadIdx.x] = B[(phase * Tile_width + threadIdx.y) * k + col];
        __syncthreads;

        // Perform dot product on tile elements for phase and accumulate the result
        for (int i = 0; i < Tile_width; ++i) {
            res += A_sh[threadIdx.y][i] * B_sh[i][threadIdx.x];
        }
        __syncthreads;

    }

    // Store result at the correct index in C
    C[row * k + col] = res;
}

void tiledMatMulDevice(float *A, float *B, float *C, int m, int n, int k) {
    
    // Declare A, B, C matrix variables for device
    float *A_d, *B_d, *C_d;

    // Allocate memory on device for A, B, C matrices
    cudaMalloc(&A_d, m * n * sizeof(float));
    cudaMalloc(&B_d, n * k * sizeof(float));
    cudaMalloc(&C_d, m * k * sizeof(float));

    // Move matrices A and B from host to device
    cudaMemcpy(A_d, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimension
    dim3 grid_dim(ceil(k/Tile_width), ceil(m/Tile_width), 1);
    dim3 block_dim(Tile_width, Tile_width, 1);

    // Call tiled matmul kernel
    tiledMatMulKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, m, n, k);

    // Copy result from device back to host
    cudaMemcpy(C, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free matrices A, B, C on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    // Declare variables for matrices A, B and C
    float *A_h, *B_h, *C_h;

    // Instantiate matrix sizes
    int m = 10;
    int n = 10;
    int k = 10;

    // Allocate memory for matrices A, B and C
    A_h = (float *) malloc(m * n * sizeof(float));
    B_h = (float *) malloc(n * k * sizeof(float));
    C_h = (float *) malloc(m * k * sizeof(float));
    
    // Populate matrices A and B
    init_matrix(A_h, m, n);
    init_matrix(B_h, n, k);

    // Call device matrix mul function
    tiledMatMulDevice(A_h, B_h, C_h, m, n, k);

    // Print result matrix
    print_matrix(C_h, m, k);


    return 0;
}