#include "conv.h"

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    // Check for zero-length input
    if (inputSize == 0 || kernelSize == 0 || numChannels == 0 || numFilters == 0 || biasData == NULL || kernel == NULL || image == NULL) {
        return NULL;  // Return NULL for zero-length input
    }
    //Check if the dimensions are compatible for convolution
    if (inputSize < kernelSize) {
        printf("Input size is less than kernel size\n");
        return NULL;
    }

    int outputSize = inputSize - kernelSize + 1;
    float ***output = (float ***)malloc(numFilters * sizeof(float **));

    for (int f = 0; f < numFilters; f++) {
        output[f] = (float **)malloc(outputSize * sizeof(float *));
        for (int i = 0; i < outputSize; i++) {
            output[f][i] = (float *)malloc(outputSize * sizeof(float));
            for (int j = 0; j < outputSize; j++) {
                output[f][i][j] = biasData[f]; 
                for (int c = 0; c < numChannels; c++) {
                    for (int ki = 0; ki < kernelSize; ki++) {
                        for (int kj = 0; kj < kernelSize; kj++) {
                            output[f][i][j] += image[c][i + ki][j + kj] * kernel[f][c][ki][kj];
                        }
                    }
                }
            }
        }
    }



    return output;
}

// Convolution with im2col algorithm
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    // Check for zero-length input
    if (inputSize == 0 || kernelSize == 0 || numChannels == 0 || numFilters == 0 || biasData == NULL || kernel == NULL || image == NULL) {
        return NULL;  // Return NULL for zero-length input
    }
    //Check if the dimensions are compatible for convolution
    if (inputSize < kernelSize) {
        printf("Input size is less than kernel size\n");
        return NULL;
    }

    // Flatten kernel


    // Apply im2col

    // Apply matmul

    // Apply col2im

    // Add bias and apply ReLU

    // Cleanup
}
