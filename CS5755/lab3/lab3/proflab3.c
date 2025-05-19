#include "proflab3.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();
    // RUN_TEST(test_conv);
    RUN_TEST(test_matmul_1024x1024);

    // RUN_TEST(test_matmul_2048x2048);


    //Test matrix_blocing
    // RUN_TEST(test_sparse_matmul_1024x1024);
    // RUN_TEST(test_sparse_matmul_blocking_1024x1024);
    // RUN_TEST(test_CSR_sparse_matmul_1024x1024);

}