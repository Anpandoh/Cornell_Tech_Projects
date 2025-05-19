#include "attention.h"
#include <stdio.h>
#include <math.h>
#include "functional.h" 

// Scaled dot-product attention
#include <math.h>
#include <stdlib.h>

float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    if (Q == NULL || K == NULL || V == NULL || seqLength <= 0 || depth <= 0) {
        return NULL;
    }

    // Allocate memory for attention matrix
    float **attention_log = (float **)malloc(seqLength * sizeof(float *));
    if (attention_log == NULL) {
        return NULL;
    }

    for (int i = 0; i < seqLength; i++) {
        attention_log[i] = (float *)malloc(seqLength * sizeof(float));
        if (attention_log[i] == NULL) {
            return NULL;
        }
    }

    // Compute scaled dot product: Q * K^T / sqrt(depth)
    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < seqLength; j++) {
            float dot_product = 0.0;
            for (int k = 0; k < depth; k++) {
                dot_product += Q[i][k] * K[j][k];
            }
            attention_log[i][j] = dot_product / sqrtf((float)depth);
        }
    }

    // Apply log-normalized softmax to each row of attention
    for (int i = 0; i < seqLength; i++) {
        // Find max element for log-sum-exp trick
        float max_val = attention_log[i][0];
        for (int j = 1; j < seqLength; j++) {
            if (attention_log[i][j] > max_val) {
                max_val = attention_log[i][j];
            }
        }

        // Compute log-sum-exp and normalize each row
        float log_sum_exp = 0.0;
        for (int j = 0; j < seqLength; j++) {
            log_sum_exp += expf(attention_log[i][j] - max_val);
        }
        log_sum_exp = logf(log_sum_exp) + max_val;

        // Subtract log-sum-exp to get log-softmax
        for (int j = 0; j < seqLength; j++) {
            attention_log[i][j] -= log_sum_exp;
        }
    }

    // Allocate memory for output
    float **output = (float **)malloc(seqLength * sizeof(float *));
    if (output == NULL) {
        return NULL;
    }

    for (int i = 0; i < seqLength; i++) {
        output[i] = (float *)malloc(depth * sizeof(float));
        if (output[i] == NULL) {
            return NULL;
        }
        for (int j = 0; j < depth; j++) {
            output[i][j] = 0.0;
            for (int k = 0; k < seqLength; k++) {
                output[i][j] += expf(attention_log[i][k]) * V[k][j];
            }
        }
    }

    // Free attention_log matrix
    for (int i = 0; i < seqLength; i++) {
        free(attention_log[i]);
    }
    free(attention_log);

    return output;
}
