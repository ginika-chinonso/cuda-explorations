// Tiled Matrix Multiplication with Coarsing
#include <stdio.h>
#include <assert.h>
#include "utils.h"

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