#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_matrix_ops.h"


#define EPSILON 0.000001f

void assert_float_array_equal_matmul(float **expected, float **actual, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j], actual[i][j], __LINE__, "Arrays Not Equal!");
        }
    }
}

void test_matmul_square_matrices(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(2 * sizeof(float));
    }
    A[0][0] = 1.0f;
    A[0][1] = 2.0f;
    A[1][0] = 3.0f;
    A[1][1] = 4.0f;

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }
    B[0][0] = 2.0f;
    B[0][1] = 0.0f;
    B[1][0] = 1.0f;
    B[1][1] = 2.0f;

    float **expected = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = 4.0f;
    expected[0][1] = 4.0f;
    expected[1][0] = 10.0f;
    expected[1][1] = 8.0f;

    // Run function under test
    float **C = matmul(A, B, 2, 2, 2, 2);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 2, 2);

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_3x1_by_1x3(void)
{
    // Setup 3x1 matrix A
    float **A = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        A[i] = (float *)malloc(1 * sizeof(float));
    }
    A[0][0] = 1.0f; // First row
    A[1][0] = 2.0f; // Second row
    A[2][0] = 3.0f; // Third row

    // Setup 1x3 matrix B
    float **B = (float **)malloc(1 * sizeof(float *));
    B[0] = (float *)malloc(3 * sizeof(float));
    B[0][0] = 4.0f; // First column
    B[0][1] = 5.0f; // Second column
    B[0][2] = 6.0f; // Third column

    // Expected result is a 3x3 matrix
    float **expected = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        expected[i] = (float *)malloc(3 * sizeof(float));
    }
    expected[0][0] = 4.0f; expected[0][1] = 5.0f; expected[0][2] = 6.0f; // 1.0 * B
    expected[1][0] = 8.0f; expected[1][1] = 10.0f; expected[1][2] = 12.0f; // 2.0 * B
    expected[2][0] = 12.0f; expected[2][1] = 15.0f; expected[2][2] = 18.0f; // 3.0 * B

    // Run function under test
    float **C = matmul(A, B, 3, 1, 1, 3);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 3, 3);

    // Cleanup
    for (int i = 0; i < 3; i++)
    {
        free(A[i]);
        free(expected[i]);
    }
    free(A);
    free(expected);

    // Free B and C (1x3 and 3x3)
    free(B[0]);
    free(B);
    for (int i = 0; i < 3; i++)
    {
        free(C[i]);
    }
    free(C);
}

void test_matmul_1024x1024(void)
{
    // Setup 1024x1024 matrix A (all elements are 2)
    float **A = (float **)malloc(1024 * sizeof(float *));
    for (int i = 0; i < 1024; i++)
    {
        A[i] = (float *)malloc(1024 * sizeof(float));
        for (int j = 0; j < 1024; j++)
        {
            A[i][j] = 2.0f;
        }
    }

    // Setup 1024x1024 matrix B (all elements are 3)
    float **B = (float **)malloc(1024 * sizeof(float *));
    for (int i = 0; i < 1024; i++)
    {
        B[i] = (float *)malloc(1024 * sizeof(float));
        for (int j = 0; j < 1024; j++)
        {
            B[i][j] = 3.0f;
        }
    }

    // Expected result is a 1024x1024 matrix with every element as 6144 (2 * 3 * 1024)
    float **expected = (float **)malloc(1024 * sizeof(float *));
    for (int i = 0; i < 1024; i++)
    {
        expected[i] = (float *)malloc(1024 * sizeof(float));
        for (int j = 0; j < 1024; j++)
        {
            expected[i][j] = 6144.0f;
        }
    }

    // Run the function under test
    float **C = matmul(A, B, 1024, 1024, 1024, 1024);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 1024, 1024);

    // Cleanup
    for (int i = 0; i < 1024; i++)
    {
        free(A[i]);
        free(B[i]);
        free(expected[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(expected);
    free(C);
}


void test_matmul_incompatible_dimensions(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(3 * sizeof(float));
    }

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }

    // Run function under test
    float **C = matmul(A, B, 2, 3, 2, 2);

    // Check expectations
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
}