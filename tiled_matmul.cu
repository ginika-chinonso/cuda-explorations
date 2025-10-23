#include <stdio.h>
#include "utils.h"

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

    // Call tiled matmul kernel
    tiledMatMulKernel();

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