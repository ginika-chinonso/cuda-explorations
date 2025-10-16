#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

// Matrix multiplication kernel
__global__ void matMulKernel(float *A, float *B, float *C, int m, int n, int k) {
    
    // Get row and column for inner product
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Instantiate the result
    float res = 0;

    // Check if thread is in matrix range
    if (row < m && col < k) {

        // Perform inner product
        for (int i = 0; i < n; ++i) {
            res += A[row * n + i] * B[i * k + col];
        }

        // Calculate result index
        int res_index = row * k + col;
        C[res_index] = res;
    }
}

// Device Matrix Multiplication function
void matMulDevice(float *A_h, float *B_h, float *C_h, int m, int n, int k) {

    // Declare matrices on the GPU
    float *A_d, *B_d, *C_d;

    // Allocate memory for matrices on the device
    cudaMalloc(&A_d, m * n * sizeof(float));
    cudaMalloc(&B_d, n * k * sizeof(float));
    cudaMalloc(&C_d, m * k * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 grid_dim(ceil(k/5.0), ceil(m/5.0), 1);
    dim3 block_dim(5, 5, 1);

    // Call matrix mul kernel
    matMulKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, m, n, k);

    // Copy result from device to host
    cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free matrices A,B and C on the device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    // Declare matrices
    float *A, *B, *C;

    // Instantiate matrix sizes
    int m = 100;
    int n = 50;
    int k = 100;

    // Allocate memory for arrays
    A = (float *) malloc(m * n * sizeof(float));
    B = (float *) malloc(n * k * sizeof(float));
    C = (float *) malloc(m * k * sizeof(float));

    // Instantiate matrices A and B
    init_matrix(A, m , n);
    init_matrix(B, n , k);

    // Call Matmul device function
    matMulDevice(A, B, C, m, n, k);

    printf("Result matrix, C: \n");

    // Print result
    print_matrix(C, m, k);

    free(A);
    free(B);
    free(C);

    return 0;

}