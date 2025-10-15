#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


// Grey scale kernel

// Given an image, converts it to grey scale
__global__ void imageBlurKernel(unsigned char *Image, unsigned char *Res, int width, int height, int channels) {

   int col = (blockIdx.x * blockDim.x) + threadIdx.x;

   int row = (blockIdx.y * blockDim.y) + threadIdx.y;

   if (col < width && row < height) {

   int grey_scale_index = (row * width) + col;
   int rgb_index = grey_scale_index * channels;

    int r = Image[rgb_index];
    int g = Image[rgb_index + 1];
    int b = Image[rgb_index + 2];

    Res[grey_scale_index] = 0.21*r + 0.72*g + 0.07*b;
   }
   
}

void imageBlurDevice(unsigned char *Image_h, unsigned char *Res_h, int width, int height, int channels) {
    
    unsigned char *Image_d, *Res_d;

    int size = width * height * sizeof(unsigned char);

    cudaMalloc(&Image_d, size * channels);
    cudaMalloc(&Res_d, size);

    cudaMemcpy(Image_d, Image_h, size * channels, cudaMemcpyHostToDevice);

    dim3 dim_grid(ceil(width/16.0),ceil(height / 16.0), 1);

    dim3 dim_block(16, 16, 1);

    imageBlurKernel<<<dim_grid, dim_block>>>(Image_d, Res_d, width, height, channels);

    cudaMemcpy(Res_h, Res_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Res_d);
    cudaFree(Image_d);
}

int main() {

    const char *filename = "test_pic.png";

    // Image, Width, Height and Channel
    int width, height, channels, output_channels;
    
    output_channels = 3;

    unsigned char *Image = stbi_load(filename, &width, &height, &channels, output_channels);

    int size = width * height;

    printf("Width: %d, Height: %d, Channels: %d, Size: %d", width, height, channels, size);

    unsigned char *Res = (unsigned char *) malloc(size * sizeof(unsigned char));

    imageBlurDevice(Image, Res, width, height, output_channels);

    *Res = stbi_write_png("grey_scale_output.png", width, height, 1, Res, width);

    stbi_image_free(Image);
    free(Res);

    return 0;

}