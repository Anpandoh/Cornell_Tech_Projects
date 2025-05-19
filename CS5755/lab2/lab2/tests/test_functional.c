#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_functional.h"
#include <float.h>
#include <time.h>


void test_softmax_basic(void) {
    float input[] = {1.0, 2.0, 3.0};
    float *output = softmax(input, 3);
    float sum = 0.0;

    // Check that the sum of the output is 0 (because the output is log softmax)
    for (int i = 0; i < 3; i++) {
        sum += expf(output[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, sum);

    // Check that the maximum input corresponds to the maximum output
    int maxInputIndex = 0;
    int maxOutputIndex = 0;

    for (int i = 1; i < 3; i++) {
        if (input[i] > input[maxInputIndex]) {
            maxInputIndex = i;
        }
        if (output[i] > output[maxOutputIndex]) {
            maxOutputIndex = i;
        }
    }

    TEST_ASSERT_EQUAL_INT(maxInputIndex, maxOutputIndex);

    // Cleanup
    free(output);
}

void test_softmax_edge_cases(void) 
{
    float input[] = {0.0, 0.0, 0.0}; // All zeros
    float *output = softmax(input, 3);
    float sum = 0.0;

    // Check that the sum of the output is 1.0
    for (int i = 0; i < 3; i++) {
        sum += expf(output[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, sum);
    
    // Check that all outputs are equal (1/3 in this case)
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0 / 3.0, expf(output[i]));
    }

    // Cleanup
    free(output);
}

void test_softmax_large_values(void) 
{
    float input[] = {1000.0, 1001.0, 1002.0}; // Large values
    float *output = softmax(input, 3);
    float sum = 0.0;

    // Check that the sum of the output is 1.0
    for (int i = 0; i < 3; i++) {
        sum += expf(output[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, sum);

    // Check that the maximum input corresponds to the maximum output
    int maxInputIndex = 0;
    int maxOutputIndex = 0;

    for (int i = 1; i < 3; i++) {
        if (input[i] > input[maxInputIndex]) {
            maxInputIndex = i;
        }
        if (output[i] > output[maxOutputIndex]) {
            maxOutputIndex = i;
        }
    }

    TEST_ASSERT_EQUAL_INT(maxInputIndex, maxOutputIndex);

    // Cleanup
    free(output);
}

void test_relu(void) {
    float inputs[] = {3.0f, 0.0f, -3.0f};
    float expected_outputs[] = {3.0f, 0.0f, 0.0f};
    int test_cases = sizeof(inputs) / sizeof(inputs[0]);

    applyRelu(inputs, test_cases);

    for (int i = 0; i < test_cases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_outputs[i], inputs[i]);
    }
}

void test_relu2(void) {
    float inputs[] = {-1e5f, -100.0f, -0.1f};
    float expected_outputs[] = {0.0f, 0.0f, 0.0f};
    int test_cases = sizeof(inputs) / sizeof(inputs[0]);

    applyRelu(inputs, test_cases);

    for (int i = 0; i < test_cases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_outputs[i], inputs[i]);
    }
}

void test_relu3(void) {
    int input_size = 1024;
    float inputs[input_size];
    float expected_outputs[input_size];

    // Seed random number generator
    srand(time(NULL));

    // Generate random inputs and expected outputs
    for (int i = 0; i < input_size; i++) {
        inputs[i] = (float)(rand() % 200000 - 100000) / 1000.0f; // Random values between -100.0 and 100.0
        expected_outputs[i] = (inputs[i] < 0) ? 0 : inputs[i]; // Expected output for ReLU
    }

    // Apply ReLU
    applyRelu(inputs, input_size);

    // Run tests
    for (int i = 0; i < input_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_outputs[i], inputs[i]);
    }

    printf("All tests passed!\n");
}


// Add more test cases as needed