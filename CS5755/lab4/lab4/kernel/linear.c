#include "linear.h"

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
	// Check for zero-length input or output sizes
    if (inputSize == 0 || outputSize == 0 || input == NULL || weights == NULL || biases == NULL) {
        return NULL;  // Return NULL for zero-length input or output sizes
    }

	float *output = (float*) malloc(sizeof(float) * outputSize);
	
	if (output == NULL){
		printf("output alloc failed\n");
		return NULL;
	}
	for (int i = 0; i < outputSize; i++) {
		//Init output with bias
		output[i] = biases[i];
		//Dot product of the input vec and the ith col of weight matrix
		for (int j = 0; j < inputSize; j++) {
			output[i] += input[j] * weights[i][j];
		}
	}

	return output;
}
