// Tiled Matrix Multiplication with Coarsing
#include <stdio.h>
#include <assert.h>
#include "utils.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void coarseTiledMatMulKernel(float *A, float *B, float *C, int m, int n, int k) {
    //
}

void coarsedMatMulDevice(float *A_h, float *B_h, float *C_h, int m, int n, int k) {
   
    // Declare device matrices
    float *A_d, *B_d, *C_d;

    // Allocate memory for the A,B and C matrices
    A_d = (float *) cudaMalloc(&A_d, m * n * sizeof(float));
    B_d = (float *) cudaMalloc(&B_d, n * k * sizeof(float));
    C_d = (float *) cudaMalloc(&A_d, m * k * sizeof(float));

    // Copy matrices A and B to device
    cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Instantiate grid and block dimensions
    dim3 grid_dim(ceil(m/float(TILE_WIDTH)), ceil(k/float(TILE_WIDTH)), 1);
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
    for (int i = 0; i < m * k; ++i) {
        assert(device_C[i] == host_C[i]);
    }

    // Free A,B and C matrices
    free(A);
    free(B);
    free(host_C);
    free(device_C);
}