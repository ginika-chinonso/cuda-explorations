// Convolution kernel with input tile size number of threads
#include <stdio.h>
#include "../../helpers/utils.h"

#define FILTER_RADIUS 2
#define IN_TILE_WIDTH 32
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - (2 * FILTER_RADIUS))
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

#define MATRIX_HEIGHT 10
#define MATRIX_WIDTH 10

// Define constant memory variable for the filter
__constant__ float ConstantFilter[FILTER_SIZE][FILTER_SIZE];

// Convolution kernel that uses input tiles number of blocks
// Threads collaborate to loads input tile to shared memory
__global__ void inputHaloCellConvolutionKernel(float *In, float *Out) {
  
  // row is gotten by subtracting FILTER_RADIUS from output row
  // This takes it FILTER_RADIUS number of rows upwards
  int row = (blockIdx.y * OUT_TILE_WIDTH + threadIdx.y) - FILTER_RADIUS;
  // col is gotten by subtracting FILTER_RADIUS from output col
  // This takes it FILTER_RADIUS number of cols to the left
  int col = (blockIdx.x * OUT_TILE_WIDTH + threadIdx.x) - FILTER_RADIUS;

  // Declare shared memory
  __shared__ float shared_in[IN_TILE_WIDTH][IN_TILE_WIDTH];

  // Load necessary input values to shared memory
  // This loads halo cells but write 0 to ghost cells
  if (row >= 0 && row < MATRIX_HEIGHT && col >= 0 && col < MATRIX_WIDTH) {
    shared_in[threadIdx.y][threadIdx.x] = In[row * MATRIX_WIDTH + col]; 
  } else {
    shared_in[threadIdx.y][threadIdx.x] = 0.0f;
  }

  // Make sure all threads have loaded their values to shared memory
  __syncthreads();

  // Get the necessary shared memory row and col
  int shared_mem_row = threadIdx.y - FILTER_RADIUS;
  int shared_mem_col = threadIdx.x - FILTER_RADIUS;

  if (row >= 0 && row < MATRIX_HEIGHT && col >= 0 && col < MATRIX_WIDTH) {
    if (shared_mem_row >= 0 && shared_mem_row < OUT_TILE_WIDTH && shared_mem_col >= 0 && shared_mem_col < OUT_TILE_WIDTH) {
      
      float res = 0.0f;

      for (int fRow = 0 ; fRow < FILTER_SIZE; ++fRow) {
        for (int fCol = 0 ; fCol < FILTER_SIZE; ++fCol) {
          res += ConstantFilter[fRow][fCol] * shared_in[shared_mem_row + fRow][shared_mem_col + fCol];
        }   
      }
      Out[row * MATRIX_WIDTH + col] = res;
    }
  }
}

void inputHaloCellConvolutionDevice (float *In, float *Out, float *Filter) {
  
  // Declare variables for the Input and Output matrices
  float *In_d, *Out_d;

  // Allocate memory for the Input and Output matrices on the device
  cudaMalloc(&In_d, MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float));
  cudaMalloc(&Out_d, MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float));

  // Copy input matrix from the host to device
  cudaMemcpy(In_d, In, MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

  // Copy filter from host to the device constant memory
  cudaMemcpyToSymbol(ConstantFilter, Filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

  // Instantiate grid and block dimensions
  dim3 grid_dim((MATRIX_WIDTH + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH, (MATRIX_HEIGHT + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH, 1);
  dim3 block_dim(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);

  // call the constant memory convolution kernel
  inputHaloCellConvolutionKernel<<<grid_dim, block_dim>>>(In_d, Out_d);

  // Copy result from device to host
  cudaMemcpy(Out, Out_d, MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device variables
  cudaFree(In_d);
  cudaFree(Out_d);
}


int main() {

  // Declare variable for the input, output and filter
  float *In;
  float *Out;
  float *Filter;

  // Allocate memory for the input, output and filter
  In = (float *) malloc(MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float));
  Out = (float *) malloc(MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float));
  Filter = (float *) malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));

  // Initialize input and filter array
  init_matrix(In, MATRIX_HEIGHT, MATRIX_WIDTH);
  init_matrix(Filter, FILTER_SIZE, FILTER_SIZE);

  // Call device fuction
  inputHaloCellConvolutionDevice(In, Out, Filter);

  printf("Output matrix \n");
  print_matrix(Out, MATRIX_HEIGHT, MATRIX_WIDTH);

  // TODO: 
  // Implement host convolution
  // Assert Out correctness

  free(In);
  free(Out);
  free(Filter);

}
