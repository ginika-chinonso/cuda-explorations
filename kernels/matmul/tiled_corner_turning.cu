// Tiled Corner turning
#include <stdio.h>
#include "../../helpers/utils.h"

#define TILE_WIDTH 32

// Similar to tiled matmul but the B matrix is transposed by flipping the x and y coordinates
__global__ void tiledCornerTurningMatrixMulKernel(float *A, float *B, float *C, int m, int n, int k) {

    // Each thread loads a piece of a tile to shared memory based on its global thread position
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Instantiate shared memory
    __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH];

    int phase = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int i = 0; i < phase; ++i) {
        
        // For each phase, load the required tile to shared memory
        // Since A is in row major order, the coordinates remain unchanged
        shared_a[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];

        // Since B is in column major order, the coordinates are swapped(transposed)
        shared_b[threadIdx.y][threadIdx.x] = B[(row * k) + (phase * TILE_WIDTH) + threadIdx.x];

        // Sync all threads to prevent race conditions
        __syncthreads();

        // With all needed values loaded to shared memory perform dot product
        float res = 0.0f;

        for (int i = 0; i < TILE_WIDTH; ++i) {
            res += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }

        __syncthreads();

        // Add partial sum from phase to the necessary output
        C[row * k + col] += res;
    }
}


void tiledCornerTurningMatMulDevice(float *A_h, float *B_h, float *C_h, int m, int n, int k) {

    // Declare variables for A, B and C
    float *A_d, *B_d, *C_d;

    // Allocate memory for A, B and C on the device
    cudaMalloc(&A_d, m * n * sizeof(float));
    cudaMalloc(&B_d, n * k * sizeof(float));
    cudaMalloc(&C_d, m * k * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Instantiate grid and block dimensions
    dim3 gridDim((k + TILE_WIDTH - 1)/TILE_WIDTH, (m + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call tiledCornerTurningMatMulKernel
    tiledCornerTurningMatrixMulKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, m, n, k);

    // Copy result from device back to host
    cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free A, B and C
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main() {

    // Instantiate matrix sizes
    int m = 10;
    int n = 10;
    int k = 10;

    // Declare variables for A,B and C matrices on the host
    float *A_h, *B_h, *host_C, *device_C;

    // Allocate memory for A, B and C matrices
    A_h = (float *) malloc(m * n * sizeof(float));
    B_h = (float *) malloc(n * k * sizeof(float));
    host_C = (float *) malloc(m * k * sizeof(float));
    device_C = (float *) malloc(m * k * sizeof(float));

    // Instantiate matrices A and B
    init_matrix(A_h, m, n);
    init_matrix(B_h, n, k);
    init_with_zeros(host_C, m, k);
    init_with_zeros(device_C, m, k);

    // Call tileddCornerTurningMatMulDevice function
    tiledCornerTurningMatMulDevice(A_h, B_h, device_C, m, n, k);

    // TODO: impl host transpose
    // TODO: call host matmul
    // TODO: assert host and device results are the same

    // print result
    print_matrix(device_C, m, k);

    // Free variables
    free(A_h);
    free(B_h);
    free(host_C);
    free(device_C);
}