#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../../helpers/utils.h"

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
    float *A, *B, *host_C, *device_C;

    // Instantiate matrix sizes
    int m = 5;
    int n = 5;
    int k = 5;

    // Allocate memory for arrays
    A = (float *) malloc(m * n * sizeof(float));
    B = (float *) malloc(n * k * sizeof(float));
    host_C = (float *) malloc(m * k * sizeof(float));
    device_C = (float *) malloc(m * k * sizeof(float));

    // Instantiate matrices A and B
    init_matrix(A, m , n);
    init_matrix(B, n , k);
    init_with_zeros(host_C, m, k);
    init_with_zeros(device_C, m, k);

    // Call Matmul device function
    matMulDevice(A, B, device_C, m, n, k);
    
    // Call Matmul host function
    matMulHost(A, B, host_C, m, n, k);

    // Verify correct result
    for (int i = 0; i < m * k; ++i) {
        assert(host_C[i] == device_C[i]);
    }

    printf("Host and Device results are correct");

    // Free A, B and C matrices memory
    free(A);
    free(B);
    free(host_C);
    free(device_C);

    return 0;

}