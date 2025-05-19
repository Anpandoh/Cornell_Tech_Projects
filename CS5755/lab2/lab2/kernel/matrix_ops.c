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


// Matmul with blocking optimization
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
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

    // Define block size
    int blockSize = 16;

    float **result = (float **)malloc(A_rows * sizeof(float *));
    if (result == NULL) {
        printf("Could not allocate result\n");
        return NULL;
    }

    for (int i = 0; i < A_rows; i++) {
        result[i] = (float *)malloc(B_cols * sizeof(float));
        if (result[i] == NULL) {
            printf("Could not allocate result row\n");
            return NULL;
        }
        for (int j = 0; j < B_cols; j++) {
            result[i][j] = 0;
        }
    }

    // Loop over blocks of the result matrix
    for (int ii = 0; ii < A_rows; ii += blockSize) {
        for (int jj = 0; jj < B_cols; jj += blockSize) {
            for (int kk = 0; kk < A_cols; kk += blockSize) {

                // Loop within a block
                for (int i = ii; i < ii + blockSize && i < A_rows; i++) {
                    for (int j = jj; j < jj + blockSize && j < B_cols; j++) {
                        for (int k = kk; k < kk + blockSize && k < A_cols; k++) {
                            result[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }

            }
        }
    }

    return result;
}