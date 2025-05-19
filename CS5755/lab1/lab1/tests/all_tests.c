#include "unity/unity.h"
#include "test_conv.h"
#include "test_nn.h"
#include "test_functional.h"
#include "test_linear.h"
#include "test_matrix_ops.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();

    // Test conv
    RUN_TEST(test_conv);
    RUN_TEST(test_conv2);
    RUN_TEST(test_conv3);
    // RUN_TEST(test_conv_multi_channel);
    RUN_TEST(test_conv_kernel_larger_than_image);
    // RUN_TEST(test_conv_with_1x1_kernel);

    // Test nn
    RUN_TEST(test_flatten_basic);
    RUN_TEST(test_flatten_empty);
    RUN_TEST(test_flatten_basic2);
    RUN_TEST(test_flatten_basic3);
    RUN_TEST(test_predict_simple_array);
    RUN_TEST(test_predict_all_same_values);
    RUN_TEST(test_predict_mix_of_negatives_and_positives);

    // Test functional
    RUN_TEST(test_softmax_basic);
    RUN_TEST(test_softmax_edge_cases);
    RUN_TEST(test_softmax_large_values);
    RUN_TEST(test_relu);
    RUN_TEST(test_relu2);
    RUN_TEST(test_relu3);

    // Test linear
    RUN_TEST(test_linear_basic);
    RUN_TEST(test_linear_basic2);
    RUN_TEST(test_linear_basic3);
    RUN_TEST(test_linear_empty_input);
    RUN_TEST(test_linear_zero_weights_biases);

    // Test matrix_ops
    RUN_TEST(test_matmul_square_matrices);
    RUN_TEST(test_matmul_3x1_by_1x3);
    RUN_TEST(test_matmul_1024x1024);
    RUN_TEST(test_matmul_incompatible_dimensions);

    return UNITY_END();
}
