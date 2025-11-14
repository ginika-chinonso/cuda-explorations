#include <stdio.h>

// Instantiate a matrix given the required number of rows and columns
void init_matrix(float *A, int rows, int cols) {
    // Iterate over every row
    for (int i = 0; i < rows; ++i) {
        // Iterate over every column
        for (int j = 0; j < cols; ++j) {
            // Instantiate cell value with global cell id
            A[i * cols + j] = (float) i * cols + j;
        }
    }
}

// Instantiate a cube given the x, y, z dimension sizes
void init_cube(float *A, int x, int y, int z) {
    // Iterate over the z dimension
    for (int i = 0; i < z; ++i) {
        // Iterate over the y dimension
        for (int j = 0; j < y; ++j) {
            // Iterate over the x dimension
            for (int k = 0; k < x; ++k) {
                // Instantiate cell value with global cell id
                A[(i * y * x) + (j * x) + k] = (float) (i * y * x) + (j * x) + k;
            }
        }
    }
}

// Initialize a matrix with zeros given number of rows and columns
void init_with_zeros(float *A, int rows, int cols) {
    // Iterate over every row
    for (int i = 0; i < rows; ++i) {
        // Iterate over every column
        for (int j = 0; j < cols; ++j) {
            // Instantiate cell value with 0
            A[i * cols + j] = 0.0;
        }
    }
}

// Print a matrix given its data, number of rows and columns
void print_matrix(float *A, int rows, int cols) {

    // Iterate over every row
    for (int i = 0; i < rows; ++i) {
        // Iterate over every column
        for (int j = 0; j < cols; ++j) {
            // Print every column in the row
            printf("%f ", A[cols * i + j]);
        }
        // Print new line
        printf("\n");
    }
    
}

// TODO: Make this function generic over the dimension
// Print a cube given its data, and dimensions
void print_cube(float *A, int x, int y, int z) {

    // Iterate over the z dimension
    for (int i = 0; i < z; ++i) {
        printf("z = %d\n", i);
        // Iterate over the y dimension
        for (int j = 0; j < y; ++j) {
            // Iterate over the x dimension
            for (int k = 0; k < x; ++k) {
                printf("%f ", A[(i * y * x) + (j * x) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


// Calculate the product of two matrices on the host
void matMulHost(float *A, float *B, float *C, int m, int n, int k) {

    // loop through every row in A
    for (int i = 0; i < m; ++i) {
        
        // loop through every column of B
        for (int j = 0; j < k; ++j) {

            // loop through every columns value
            for (int index = 0; index < n; ++index) {
                C[i * k + j] += A[i * n + index] * B[index * k + j];
            }
            
        }
        
    }
}