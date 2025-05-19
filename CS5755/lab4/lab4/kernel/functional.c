#include "functional.h"

float relu(float x)
{
    return x > 0 ? x : 0; //If x is greater than 0, return x, else return 0
}

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax(float *input, int inputSize)
{
    // Check for zero-length input
    if (inputSize == 0 || input == NULL) {
        return NULL;  // Return NULL for zero-length input
    }

    // Allocate output array
    float *output = (float *)malloc(inputSize * sizeof(float));
    if (output == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Find maximum of input vector
    float maxInput = input[0];
    for (size_t i = 1; i < inputSize; i++) {
        if (input[i] > maxInput) {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid underflow
    float sum = 0.0;
    for (size_t i = 0; i < inputSize; i++) {
        output[i] = expf(input[i] - maxInput);
        sum += output[i];
    }
    // Normalise and apply log

    for (size_t i = 0; i < inputSize; i++) {
        output[i] = logf(output[i] / sum);
    }

    return output;

}