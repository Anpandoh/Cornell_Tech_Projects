#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_linear.h"


void test_linear_basic(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_basic2(void)
{
    // Define the input array of size 512 (for simplicity, fill with some values, e.g., 0.5)
    float input[512];
    for (int i = 0; i < 512; i++) {
        input[i] = 0.5;  // Example value for all elements
    }

    // Define the weights array of size [2][512] (2 outputs, 512 inputs)
    float *weights[2];
    for (int i = 0; i < 2; i++) {
        weights[i] = (float*)malloc(1024 * sizeof(float));
        for (int j = 0; j < 1024; j++) {
            weights[i][j] = 1.0;  // Example weight of 1.0 for simplicity
        }
    }

    // Define the biases array of size 2
    float biases[2] = {0.0, -1.5};  // Example biases

    // Call the linear function
    float *output = linear(input, weights, biases, 512, 2);
    
    // Check the expected outputs
    // For this example, with input 0.5 and weights 1.0 for all, output[0] should be 0.5 * 512 + 0.0
    // and output[1] should be 0.5 * 512 - 1.5
    TEST_ASSERT_EQUAL_FLOAT(256.0, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(254.5, output[1]);

    // Cleanup
    free(output);
    for (int i = 0; i < 2; i++) {
        free(weights[i]);
    }
}


// Add more test cases as needed
void test_linear_basic3(void)
{
    // Define the input array of size 1024 (for simplicity, fill with some values, e.g., 0.5)
    float input[1024];
    for (int i = 0; i < 1024; i++) {
        input[i] = 0.5;  // Example value for all elements
    }

    // Define the weights array of size [2][1024] (2 outputs, 1024 inputs)
    float *weights[2];
    for (int i = 0; i < 2; i++) {
        weights[i] = (float*)malloc(1024 * sizeof(float));
        for (int j = 0; j < 1024; j++) {
            weights[i][j] = 1.0;  // Example weight of 1.0 for simplicity
        }
    }

    // Define the biases array of size 2
    float biases[2] = {0.0, -1.5};  // Example biases

    // Call the linear function
    float *output = linear(input, weights, biases, 1024, 2);
    
    // Check the expected outputs
    // For this example, with input 0.5 and weights 1.0 for all, output[0] should be 0.5 * 1024 + 0.0
    // and output[1] should be 0.5 * 1024 - 1.5
    TEST_ASSERT_EQUAL_FLOAT(512.0, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(510.5, output[1]);

    // Cleanup
    free(output);
    for (int i = 0; i < 2; i++) {
        free(weights[i]);
    }
}

void test_linear_empty_input(void)
{
    float input_empty[] = {};
    float *weights_empty[] = {};
    float biases_empty[] = {};
    float *output_empty = linear(input_empty, weights_empty, biases_empty, 0, 0);

    TEST_ASSERT_NULL(output_empty); // Expect NULL since there are no inputs or weights

    // Cleanup
    free(output_empty); // Optional, in case NULL is not returned but memory is allocated
}

void test_linear_zero_weights_biases(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights_zero[] = {(float[]){0.0, 0.0, 0.0}, (float[]){0.0, 0.0, 0.0}};
    float biases_zero[] = {0.0, 0.0};
    float *output_zero = linear(input, weights_zero, biases_zero, 3, 2);

    TEST_ASSERT_EQUAL_FLOAT(0.0, output_zero[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0, output_zero[1]);

    // Cleanup
    free(output_zero);
}
