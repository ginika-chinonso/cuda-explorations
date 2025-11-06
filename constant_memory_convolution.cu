// Constant memory convolution kernel
#include <stdio.h>
#include "utils.h";

#define filter_width 5
#define filter_height 5
#define TILE_WIDTH 32

__constant__ float Filter[filter_height][filter_width];


__global__ void constantMemoryConvolutionKernel() {
  //
}

void constantMemoryConvolutionDevice(float *In, float *Out, int height, int width) {
  
  // Declare variables for the Input and Output matrices
  float *In_d, *Out_d;

  // Allocate memory for the Input and Output matrices on the device
  cudaMalloc(&In_d, height * width * sizeof(float));
  cudaMalloc(&Out_d, height * width * sizeof(float));

  // Copy input matrix from the host to device
  cudaMemcpy(In_d, In, height * width * sizeof(float), cudaMemcpyHostToDevice);

  // Instantiate grid and block dimensions
  dim3 grid_dim(ceil(width + TILE_WIDTH - 1 / TILE_WIDTH), ceil(height + TILE_WIDTH - 1 / TILE_WIDTH), 1);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);

  // call the constant memory convolution kernel
  constantMemoryConvolutionKernel<<<grid_dim, block_dim>>>();

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

  // Declare variable for the input, output and filter
  float *In;
  float *Out;

  // Allocate memory for the input, output and filter
  In = (float *) malloc(height * width * sizeof(float));
  Out = (float *) malloc(height * width * sizeof(float));

  // Initialize input array
  init_matrix(In, height, width);
  init_matrix(Out, height, width);
  init_matrix((float *) Filter, filter_height, filter_width);

  // Call device fuction
  constantMemoryConvolutionDevice(In, Out, height, width);

  print_matrix(Out, height, width);

}
