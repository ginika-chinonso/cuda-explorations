// Constant memory convolution kernel
#include <stdio.h>
#include "utils.h"

#define filter_radius 2
#define TILE_WIDTH 32

// Constant memory variable for filter
__constant__ float ConstantFilter[2 * filter_radius + 1][2 * filter_radius + 1];


__global__ void constantMemoryConvolutionKernel(float *In, float *Out, int height, int width) {
  
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float res = 0;

  for (int fRow = 0 ; fRow < 2 * filter_radius + 1; ++fRow) {
    for (int fCol = 0; fCol < 2 * filter_radius + 1; ++fCol) {

      int in_y = row - filter_radius + fRow;
      int in_x = col - filter_radius + fCol;

      // Checks if cell is a ghost cell
      if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
        res += ConstantFilter[fRow][fCol] * In[in_y * width + in_x];
      } 

    }
    
  }

  if (row >= 0 && row < height && col >= 0 && col < width) {
    Out[row * width + col] = res;
  }
}

void constantMemoryConvolutionDevice(float *In, float *Out, float *Filter, int height, int width, int filter_size) {
  
  // Declare variables for the Input and Output matrices
  float *In_d, *Out_d;

  // Allocate memory for the Input and Output matrices on the device
  cudaMalloc(&In_d, height * width * sizeof(float));
  cudaMalloc(&Out_d, height * width * sizeof(float));

  // Copy input matrix from the host to device
  cudaMemcpy(In_d, In, height * width * sizeof(float), cudaMemcpyHostToDevice);

  // Copy filter to constant memory
  cudaMemcpyToSymbol(ConstantFilter, Filter, filter_size * filter_size * sizeof(float));

  // Instantiate grid and block dimensions
  dim3 grid_dim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH, 1);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);

  // call the constant memory convolution kernel
  constantMemoryConvolutionKernel<<<grid_dim, block_dim>>>(In_d, Out_d, height, width);

  // Copy result from device to host
  cudaMemcpy(Out, Out_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device variables
  cudaFree(In_d);
  cudaFree(Out_d);
}


int main() {

  // Instantiate matrix size
  int height = 10;
  int width = 10;

  // Instantiate filter size
  int filter_size = 2 * filter_radius + 1;

  // Declare variable for the input, output and filter
  float *In;
  float *Out;
  float *Filter;

  // Allocate memory for the input, output and filter
  In = (float *) malloc(height * width * sizeof(float));
  Out = (float *) malloc(height * width * sizeof(float));
  Filter = (float *) malloc(filter_size * filter_size * sizeof(float));

  // Initialize input array
  init_matrix(In, height, width);
  init_matrix(Filter, filter_size, filter_size);
  
  // Call device fuction
  constantMemoryConvolutionDevice(In, Out, Filter, height, width, filter_size);

  printf("Result matrix: \n");
  print_matrix(Out, height, width);

  // TODO
  // write host convolution and assert result

  // Free host variables
  free(In);
  free(Out);
  free(Filter);

  return 0;
}
