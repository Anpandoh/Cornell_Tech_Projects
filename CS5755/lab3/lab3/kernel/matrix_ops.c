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
    int blockSize = 64;

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
    for (int jj = 0; jj < B_cols; jj += blockSize) {
        for (int ii = 0; ii < A_rows; ii += blockSize) {
            for (int kk = 0; kk < A_cols; kk += blockSize) {

                // Loop within a block
                for (int j = jj; j < jj + blockSize && j < B_cols; j++) {
                    for (int i = ii; i < ii + blockSize && i < A_rows; i++) {
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
float **convert_to_csr(float **A, int A_rows, int A_cols) {
    // Allocate memory for CSR format
    int nnz = 0; // Number of non-zero elements
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            if (A[i][j] != 0) {
                nnz++;
            }
        }
    }

    float **csr = (float **)malloc(3 * sizeof(float *));
    csr[0] = (float *)malloc((A_rows + 1) * sizeof(float)); // Row pointers
    csr[1] = (float *)malloc(nnz * sizeof(float)); // Column indices
    csr[2] = (float *)malloc(nnz * sizeof(float)); // Non-zero values

    int k = 0;
    csr[0][0] = 0;
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            if (A[i][j] != 0) {
                csr[1][k] = j;
                csr[2][k] = A[i][j];
                k++;
            }
        }
        csr[0][i + 1] = k;
    }

    return csr;
}


// float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols) {
//     // Check if the dimensions are compatible for matrix multiplication
//     if (A_rows == 0 || A_cols == 0 || B_rows == 0 || B_cols == 0) {
//         printf("Matrix dimensions are zero\n");
//         return NULL;
//     }
//     if (A_cols != B_rows) {
//         printf("Matrix dimensions are not compatible for multiplication\n");
//         return NULL;
//     }

//     // Convert A to CSR format
//     float **csrA = convert_to_csr(A, A_rows, A_cols);

//     // Allocate memory for the result matrix
//     float **result = (float **)malloc(A_rows * sizeof(float *));
//     if (result == NULL) {
//         printf("Could not allocate result\n");
//         return NULL;
//     }
//     for (int i = 0; i < A_rows; i++) {
//         result[i] = (float *)malloc(B_cols * sizeof(float));
//         if (result[i] == NULL) {
//             printf("Could not allocate result row\n");
//             return NULL;
//         }
//         for (int j = 0; j < B_cols; j++) {
//             result[i][j] = 0;
//         }
//     }

//     // Perform matrix multiplication using CSR format
//     //Repeat 10 times

//     for (int i = 0; i < A_rows; i++) {
//         for (int k = csrA[0][i]; k < csrA[0][i + 1]; k++) {
//             int colA = csrA[1][k];
//             float valA = csrA[2][k];
//             for (int j = 0; j < B_cols; j++) {
//                 result[i][j] += valA * B[colA][j];
//             }
//         }
//     }

//     // Free CSR format memory
//     free(csrA[0]);
//     free(csrA[1]);
//     free(csrA[2]);
//     free(csrA);

//     return result;
// }
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols) {
    // Check if the dimensions are compatible for matrix multiplication
    if (A_rows == 0 || A_cols == 0 || B_rows == 0 || B_cols == 0) {
        printf("Matrix dimensions are zero\n");
        return NULL;
    }
    if (A_cols != B_rows) {
        printf("Matrix dimensions are not compatible for multiplication\n");
        return NULL;
    }

    // Convert A and B to CSR format
    float **csrA = convert_to_csr(A, A_rows, A_cols);
    float **csrB = convert_to_csr(B, B_rows, B_cols);

    // Allocate memory for the result matrix
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

    // Perform matrix multiplication using CSR format
    // for (int t = 0; t < 10; t++) { // Loop 10 times for testing purposes
    //     for (int i = 0; i < A_rows; i++) {
    //         for (int j = 0; j < B_cols; j++) {
    //             result[i][j] = 0;
    //         }
    //     }
    for (int i = 0; i < A_rows; i++) {
        for (int k = csrA[0][i]; k < csrA[0][i + 1]; k++) {
            int colA = csrA[1][k];
            float valA = csrA[2][k];
            for (int l = csrB[0][colA]; l < csrB[0][colA + 1]; l++) {
                int colB = csrB[1][l];
                float valB = csrB[2][l];
                result[i][colB] += valA * valB;
            }
        }
        }
    // }

    // Free CSR format memory
    free(csrA[0]);
    free(csrA[1]);
    free(csrA[2]);
    free(csrA);
    free(csrB[0]);
    free(csrB[1]);
    free(csrB[2]);
    free(csrB);

    return result;
}