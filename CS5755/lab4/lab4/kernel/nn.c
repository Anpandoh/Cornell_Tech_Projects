#include "nn.h"

float *flatten(float ***input, int inputSize, int depth)
{
    if (inputSize == 0 || depth == 0 || input == NULL) {
        printf("Input is NULL\n");
        return NULL;
    }
  	float* flattenedOutput = (float*)malloc(inputSize * inputSize * depth * sizeof(float));
	if (flattenedOutput == NULL) {
		printf("Could not allocated flattenedOutput\n");
		return NULL;
	}

    // Index for the flattened array
    int index = 0;

    // Iterate over depth and slice size to copy values into the flattened array
    for (int d = 0; d < depth; ++d) {
        for(int i = 0; i < inputSize; ++i) {
            for(int j = 0; j < inputSize; ++j) {
                flattenedOutput[index++] = input[d][i][j];
            }
        }
    }

    return flattenedOutput;
}

void destroyConvOutput(float ***convOutput, int convOutputSize)
{
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < convOutputSize; j++)
        {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}

int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData)
{
    // 1. Perform the convolution operation
    int numFilters = 32;
    int kernelSize = 5;
    int inputSize = 28;
    float ***convOutput = convolution(image, numChannels, conv1WeightsData, conv1BiasData, numFilters, inputSize, kernelSize);
    if (convOutput == NULL) {
        printf("Convolution operation failed\n");
        return -1;
    }
    int convOutputSize = inputSize - kernelSize + 1;
    // 2. Flatten the output
    float *flattenedOutput = flatten(convOutput, convOutputSize, numFilters);

    // 3. Perform the fully connected operations


    //first layer
    int fc1OutputSize = 128;
    float *fc1Output = linear(flattenedOutput, fc1WeightsData, fc1BiasData, convOutputSize * convOutputSize * numFilters, fc1OutputSize);
    //Apply the ReLU activation
    applyRelu(fc1Output, 128);
    
    //second layer
    int fc2OutputSize = 10;
    float *fc2Output = linear(fc1Output, fc2WeightsData, fc2BiasData, fc1OutputSize, fc2OutputSize);



    // 4. Apply the final softmax activation
    float *probabilityVector = softmax(fc2Output, fc2OutputSize);
    // 5. Make predictions
    int predictedClass = predict(probabilityVector, 10);

    // Clean up the memory usage
    destroyConvOutput(convOutput, convOutputSize);
    free(flattenedOutput);
    free(fc1Output);
    free(fc2Output);
    free(probabilityVector);
}



int predict(float *probabilityVector, int numClasses)
{
    if (numClasses == 0 || probabilityVector == NULL) {
        printf("Input is NULL\n");
        return -1;
    }
    int predictedClass = 0;
    float maxProb = probabilityVector[0];
    for (int i = 1; i < numClasses; i++)
    {
        if (probabilityVector[i] > maxProb)
        {
            maxProb = probabilityVector[i];
            predictedClass = i;
        }
    }
    return predictedClass;
}
