#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


// Grey scale image blur kernel
// Generates a blured grey scale version of an image
__global__ void greyScaleImageBlurKernel(unsigned char *Image, unsigned char *Res, int width, int height, int channel) {

    // Gets the pixel row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Instantiate image blur radius
    int blur_size = 20;

    // Instantiate pixel value
    float pixel_value = 0;

    // Instantiate number of valid pixels
    float valid_pixels = 0;

    // Verify that pixel is within image range
    if (row < height && col < width) {

        // Get associated pixels for blur radius
        // Get blur row
        for (int i = -blur_size; i < blur_size + 1; ++i) {
            // Get blur column
            for (int j = -blur_size; j < blur_size + 1; ++j) {
                int curr_row = row + i;
                int curr_col = col + j;

                // Verify if current pixel is a valid pixel
                if (curr_col < width && curr_row < height && curr_col >= 0 && curr_row >= 0) {
                    valid_pixels++;

                    // Calculate current index
                    int curr_index = (curr_row * width + curr_col) * channel;
                    
                    // Find the grey scale value off that pixel and add it to pixel value
                    pixel_value += 0.21*Image[curr_index] + 0.72*Image[curr_index + 1] + 0.07*Image[curr_index + 2];
                }
            }

        }

        // Calculate pixel index for result
        int res_index = row * width + col;

        // Store new pixel
        Res[res_index] = (unsigned char) (pixel_value / valid_pixels);
    }
}


void greyScaleImageBlurDevice(unsigned char *Image_h, unsigned char *Res_h, int width, int height, int channels) {
    
    // Declare the image and result variables on the device
    unsigned char *Image_d, *Res_d;

    // Calculate the amount of bytes to be allocated for the image
    int size = width * height * sizeof(unsigned char);

    // Allocate memory on the device for the image
    cudaMalloc(&Image_d, size * channels);
    
    // Allocate memory on the device for the result
    cudaMalloc(&Res_d, size);

    // Copy the image to be processed from host to device
    cudaMemcpy(Image_d, Image_h, size * channels, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);

    dim3 block_dim(16, 16, 1);

    // Call the image blur kernel
    greyScaleImageBlurKernel<<<grid_dim, block_dim>>>(Image_d, Res_d, width, height, channels);

    // Copy the result from device back to host
    cudaMemcpy(Res_h, Res_d, size, cudaMemcpyDeviceToHost);

    // Free image and result variables
    cudaFree(Image_d);
    cudaFree(Res_d);
}


int main () {

    // Instantiate the file name variable
    const char *file_name = "test_pic.png";

    // Declare variables for the width, height and channels
    int width, height, channels;

    // Load Image using stb
    unsigned char *Image = stbi_load(file_name, &width, &height, &channels, 3);

    printf("Image loaded...: Width: %d, Height: %d, Channels: %d", width, height, channels);

    // Calculate result image size
    int size = width * height;

    // Allocate memory for the result
    unsigned char *Res = (unsigned char *) malloc(size * sizeof(unsigned char));

    // Call image blur device function
    greyScaleImageBlurDevice(Image, Res, width, height, 3);

    // Write result to output file
    *Res = stbi_write_png("grey_scale_image_blur_output.png", width, height, 1, Res, width);

    // Free image and result variables
    stbi_image_free(Image);
    free(Res);

    return 0;
}