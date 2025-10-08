#include <stdio.h>
#include <stdlib.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
   }
}


void init_array(float *A_h, int array_size) {
    for (int i = 0; i < array_size; ++i) {
        A_h[i] = float(i);
    }
}


int main(int argc, int *argv[]) {

    int n = 5;

    float *A_h = (float *) malloc(n * sizeof(float));
    float *B_h = (float *) malloc(n * sizeof(float));
    float *C_h = (float *) malloc(n * sizeof(float));

    init_array(A_h, n);
    init_array(B_h, n);

    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; ++i) {
       printf("%f\n", C_h[i]);        
    }
    free(A_h);
    free(B_h);
    free(C_h);
    
    return 0;
}