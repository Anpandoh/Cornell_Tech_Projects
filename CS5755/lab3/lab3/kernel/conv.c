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

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    // Calculate the output dimension
    int outputDim = (imageSize - kernelSize) / stride + 1;
    *outputSize = outputDim;

    // Calculate the dimensions of the column matrix
    int colHeight = numChannels * kernelSize * kernelSize;
    int colWidth = outputDim * outputDim;

    // Allocate memory for the column matrix
    float **col = (float **)malloc(colHeight * sizeof(float *));
    for (int i = 0; i < colHeight; i++) {
        col[i] = (float *)malloc(colWidth * sizeof(float));
    }

    // Fill the column matrix with image patches
    for (int c = 0; c < numChannels; c++) {
        for (int kRow = 0; kRow < kernelSize; kRow++) {
            for (int kCol = 0; kCol < kernelSize; kCol++) {
                int colIndex = c * kernelSize * kernelSize + kRow * kernelSize + kCol;
                for (int outRow = 0; outRow < outputDim; outRow++) {
                    for (int outCol = 0; outCol < outputDim; outCol++) {
                        int imgRow = outRow * stride + kRow;
                        int imgCol = outCol * stride + kCol;
                        col[colIndex][outRow * outputDim + outCol] = image[c][imgRow][imgCol];
                    }
                }
            }
        }
    }

    return col;
}

// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // Allocate memory for the output image
    float ***output = (float ***)malloc(num_kernels * sizeof(float **));
    for (int k = 0; k < num_kernels; k++) {
        output[k] = (float **)malloc(conv_rows * sizeof(float *));
        for (int i = 0; i < conv_rows; i++) {
            output[k][i] = (float *)malloc(conv_cols * sizeof(float));
            for (int j = 0; j < conv_cols; j++) {
                output[k][i][j] = 0.0f; // Initialize to zero
            }
        }
    }

    // Fill the output image with the values from the column matrix
    int colHeight = num_kernels * conv_rows * conv_cols;
    int colWidth = conv_rows * conv_cols;
    for (int k = 0; k < num_kernels; k++) {
        for (int i = 0; i < conv_rows; i++) {
            for (int j = 0; j < conv_cols; j++) {
                int colIndex = k * conv_rows * conv_cols + i * conv_cols + j;
                output[k][i][j] = result[colIndex / colWidth][colIndex % colWidth];
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

    int outputSize;
    float **colImage = im2col(image, numChannels, inputSize, kernelSize, 1, &outputSize);

    // Flatten kernel
    int kernelSizeSquared = kernelSize * kernelSize;
    float **flattenedKernel = (float **)malloc(numFilters * sizeof(float *));
    for (int f = 0; f < numFilters; f++) {
        flattenedKernel[f] = flatten(kernel[f], kernelSize, numChannels);
    }

    // Apply matmul based on matmul_type
    float **result;
    switch (matmul_type) {
        case MATMUL_BASE:
            result = matmul_blocking(flattenedKernel, colImage, numFilters, numChannels * kernelSizeSquared, numChannels * kernelSizeSquared, outputSize * outputSize);
            break;
        case MATMUL_SPARSE:
            result = matmul_sparse(flattenedKernel, colImage, numFilters, numChannels * kernelSizeSquared, numChannels * kernelSizeSquared, outputSize * outputSize);
            break;
    }

    // Apply col2im
    float ***output = col2im(result, numFilters, outputSize, outputSize);

    // Add bias and apply ReLU
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                output[f][i][j] += biasData[f];
                output[f][i][j] = relu(output[f][i][j]);
            }
        }
    }

    // Cleanup
    for (int i = 0; i < numChannels * kernelSizeSquared; i++) {
        free(colImage[i]);
    }
    free(colImage);

    for (int f = 0; f < numFilters; f++) {
        free(flattenedKernel[f]);
    }
    free(flattenedKernel);

    for (int i = 0; i < numFilters; i++) {
        free(result[i]);
    }
    free(result);

    return output;
}
