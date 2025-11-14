// A 3d convolution kernel using constant memory to store the filter
#include <stdio.h>
#include <cuda_runtime.h>
#include "../helpers/utils.h"

#define FILTER_RADIUS 2
#define TILE_WIDTH 2
#define MATRIX_DIM 5
#define FILTER_DIM ((2 * FILTER_RADIUS) + 1)

__constant__ float ConstantFilter[FILTER_DIM][FILTER_DIM][FILTER_DIM];

__global__ void constantMemory3DConvolutionKernel(float *In, float *Out) {
  
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int depth = blockIdx.z * TILE_WIDTH + threadIdx.z;

  float res = 0;

  for(int fDepth = 0; fDepth < FILTER_DIM; ++fDepth) {
    for (int fRow = 0 ; fRow < FILTER_DIM; ++fRow) {
      for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {

        int in_y = row - FILTER_RADIUS + fRow;
        int in_x = col - FILTER_RADIUS + fCol;
        int in_z = depth - FILTER_RADIUS + fDepth;

        // Checks if cell is a ghost cell
        if (in_x >= 0 && in_x < MATRIX_DIM && in_y >= 0 && in_y < MATRIX_DIM && in_z >= 0 && in_z < MATRIX_DIM) {
          res += ConstantFilter[fDepth][fRow][fCol] * In[(in_z * MATRIX_DIM * MATRIX_DIM) + (in_y * MATRIX_DIM) + in_x];
        } 

      }
    }
    
  }

  if (row >= 0 && row < MATRIX_DIM && col >= 0 && col < MATRIX_DIM && depth >= 0 && depth < MATRIX_DIM) {
    Out[(depth * MATRIX_DIM * MATRIX_DIM) + (row * MATRIX_DIM) + col] = res;
  }
}

void constantMemoryConvolutionDevice(float *In, float *Out, float *Filter) {
  
  // Declare variables for the Input and Output matrices
  float *In_d, *Out_d;

  // Allocate memory for the Input and Output matrices on the device
  cudaMalloc(&In_d, MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float));
  cudaMalloc(&Out_d, MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float));

  // Copy input matrix from the host to device
  cudaMemcpy(In_d, In, MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);

  // Copy the filter to constant memory
  cudaMemcpyToSymbol(ConstantFilter, Filter, FILTER_DIM * FILTER_DIM * FILTER_DIM * sizeof(float));

  // Instantiate grid and block dimensions
  dim3 grid_dim((MATRIX_DIM + TILE_WIDTH - 1) / TILE_WIDTH, (MATRIX_DIM + TILE_WIDTH - 1) / TILE_WIDTH,  (MATRIX_DIM + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  // call the constant memory convolution kernel
  constantMemory3DConvolutionKernel<<<grid_dim, block_dim>>>(In_d, Out_d);

  // Copy result from device to host
  cudaMemcpy(Out, Out_d, MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyDeviceToHost);

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
  In = (float *) malloc(MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float));
  Out = (float *) malloc(MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * sizeof(float));
  Filter = (float *) malloc(FILTER_DIM * FILTER_DIM * FILTER_DIM * sizeof(float));

  // Initialize input array
  init_cube(In, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM);
  init_cube(Filter, FILTER_DIM, FILTER_DIM, FILTER_DIM);
  
  // Call device fuction
  constantMemoryConvolutionDevice(In, Out, Filter);

  printf("Result matrix: \n");
  print_cube(Out, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM);

  // TODO
  // write host convolution and assert result

  // Free host variables
  free(In);
  free(Out);
  free(Filter);

  return 0;
}
