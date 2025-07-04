#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_multithread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);


#endif /* MATRIX_OPS_H */