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