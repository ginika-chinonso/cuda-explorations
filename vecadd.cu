#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

// Adds two vectors on the host
void vecAdd_host(float *A_h, float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
   }
}


// Vector addition kernel
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {

    // Calculate global thread Id
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Verify thread is valid
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


// Adds two vectors on the device
// where A,B,C are pointers to the arrays on the host
void vecAdd_device(float *A, float *B, float *C, int n) {

    // Declare A, B and C matrices on the device
    float *d_A, *d_B, *d_C;
   
    // Allocate memory on deevice for vectors A, B and C
    cudaMalloc((void **) &d_A, n * sizeof(float));
    cudaMalloc((void **) &d_B, n * sizeof(float));
    cudaMalloc((void **) &d_C, n * sizeof(float));

    // Copy vectors A and B from Host to Device
    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);
   
    // Declare timers
    cudaEvent_t start, stop;

    // Create start and stop events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel to perform vector addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // Record stop event
    cudaEventRecord(stop);

    // Synchronize stop event to be sure it has been recorded
    cudaEventSynchronize(stop);

    // Declare elapsed time variable
    float elapsed_time;

    // Calculate time elapsed
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Print time elapsed
    printf("Vector add kernel took %f ms\n", elapsed_time);

   // Copy result from device to host 
    cudaMemcpy(C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory of A, B and C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy start and stop events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


int main(int argc, char **argv) {

    // Size of vector
    int n = 5;

    // Allocate memory for the A, B and C matrices on the host
    float *A_h = (float *) malloc(n * sizeof(float));
    float *B_h = (float *) malloc(n * sizeof(float));
    float *host_C = (float *) malloc(n * sizeof(float));
    float *device_C = (float *) malloc(n * sizeof(float));

    // Initialize the A and B matrices
    init_matrix(A_h, 1, n);
    init_matrix(B_h, 1, n);

    // Call host vector add function
    vecAdd_host(A_h, B_h, host_C, n);

    // Call device vector add function
    vecAdd_device(A_h, B_h, device_C, n);

    // Assert that host_C and device_C are the same
    for (int i = 0; i < n; ++i) {
        assert(host_C[i] == device_C[i]);
    }

    printf("Host and device results match \n");

    // Free A, B and C matrices
    free(A_h);
    free(B_h);
    free(host_C);
    free(device_C);
    
    return 0;
}