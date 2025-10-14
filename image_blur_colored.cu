#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


// Image blur kernel
__global__ void imageBlurKernel(unsigned char *Image, unsigned char *Res, int width, int height, int channel) {
    
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    // instantiate the blur size
    // TODO: make it a variable
    int blur_size = 5;

    // Pixel should be a valid pixel
    if (col < width && row < height) {
        // number of valid pixels (especially for edge pixels)
        float valid_pixels = 0;


        // Get pixel RGB values
        int r = 0;
        int g = 0;
        int b = 0;

        //  Get the rgb values of all pixels around pixel of interest within the blur radius
        // For the row
        for (int i = -blur_size; i < blur_size + 1; ++i) {
            // For the column
            for (int j = -blur_size; j < blur_size + 1; ++j) {

                // Get row and column of current pixel in question
                int curr_row = row + i;
                int curr_col = col + j;

                // check if pixels are valid pixels
                if (curr_col >= 0 && curr_row >= 0 && curr_col < width && curr_row < height) {
                    // Increment valid pixels count
                    valid_pixels++;

                    // Get current image index
                    int curr_img_index = (curr_row * width + curr_col) * channel;

                    // Add RGB values to pixel RGB values
                    r = r + Image[curr_img_index];
                    g = g + Image[curr_img_index + 1];
                    b = b + Image[curr_img_index + 2];

                }
            }
        }
        // // Get average RGB values
        unsigned char new_r_value = r / valid_pixels;
        unsigned char new_g_value = g / valid_pixels;
        unsigned char new_b_value = b / valid_pixels;

        int image_index = (row * width + col) * channel;

        // Write new RGB values to result vector
        Res[image_index] = new_r_value;
        Res[image_index + 1] = new_g_value;
        Res[image_index + 2] = new_b_value;

    }
}

void imageBlurDevice(unsigned char *Image_h, unsigned char *Res_h, int width, int height, int channels) {
    
    // Declare the image and result variables on the device
    unsigned char *Image_d, *Res_d;

    // Calculate the amount of bytes to be allocated for the image
    int size = width * height * channels * sizeof(unsigned char);

    // Allocate memory on the device for the image
    cudaMalloc(&Image_d, size);
    
    // Allocate memory on the device for the result
    cudaMalloc(&Res_d, size);

    // Copy the image to be processed from host to device
    cudaMemcpy(Image_d, Image_h, size, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);

    dim3 block_dim(16, 16, 1);

    // Call the image blur kernel
    imageBlurKernel<<<grid_dim, block_dim>>>(Image_d, Res_d, width, height, channels);

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

    // Instantiate output channels to 3 for rgb
    int output_channels = 3;

    // Load Image using stb
    unsigned char *Image = stbi_load(file_name, &width, &height, &channels, output_channels);

    printf("Image loaded...: Width: %d, Height: %d, Channels: %d", width, height, channels);

    // Calculate result image size
    int size = width * height * output_channels;

    // Allocate memory for the result
    unsigned char *Res = (unsigned char *) malloc(size * sizeof(unsigned char));

    // Call image blur device function
    imageBlurDevice(Image, Res, width, height, output_channels);

    // Write result to output file
    *Res = stbi_write_png("image_blur_output.png", width, height, output_channels, Res, width * output_channels);

    // Free image and result variables
    stbi_image_free(Image);
    free(Res);

    return 0;
}