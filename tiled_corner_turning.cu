// Tiled Corner turning

#define TILE_WIDTH 32

// Similar to tiled matmul but the B matrix is transposed by flipping the x and y coordinates
__global__ void tiledCornerTurningMatrixMulKernel(float *A, float *B, float *C, int m, int n, int k) {

    // Each thread loads a piece of a tile to shared memory based on its global thread position
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Instantiate shared memory
    __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH] = {0.0f};
    __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH] = {0.0f};

    int phase = n + TILE_WIDTH - 1 / TILE_WIDTH;

    for (int i = 0; i < phase; ++i) {
        
        // For each phase, load the required tile to shared memory
        // Since A is in row major order, the coordinates remain unchanged
        shared_a[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];

        // Since B is in column major order, the coordinates are swapped(transposed)
        shared_b[][] = B[];

        // Sync all threads to prevent race conditions
        __syncthreads();

        // With all needed values loaded to shared memory perform dot product
        float res = 0.0f;

        for (int i = 0; i < TILE_WIDTH; ++i) {
            res += shared_a[][] * shared_b[][];
        }

        // Add partial sum from phase to the necessary output
        C[row * k + col] += res;
    }
}