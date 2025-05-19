#include "matrix_ops.h"

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols) 
{
    // Check if the dimensions are compatible for matrix multiplication
    if (A_rows == 0 || A_cols == 0 || B_rows == 0 || B_cols == 0) {
        printf("Matrix dimensions are zero\n");
        return NULL;
    }
    if (A_cols != B_rows) {
        printf("Matrix dimensions are not compatible for multiplication\n");
        return NULL;
    }
    float **result = (float **)malloc(A_rows * sizeof(float *));
    if (result == NULL) {
        printf("Could not allocate result\n");
        return NULL;
    }


    
    for (int i = 0; i < A_rows; i++) {
        result[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            result[i][j] = 0;
            for (int k = 0; k < A_cols; k++)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}