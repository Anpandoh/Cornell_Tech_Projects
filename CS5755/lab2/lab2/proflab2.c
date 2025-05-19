#include "proflab2.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();

    //Test matrix_blocing
    for (int i = 0; i < 10; i++) {
        // RUN_TEST(test_matmul_512x512);
        // RUN_TEST(test_matmul_blocking_512x512);
        // RUN_TEST(test_matmul_1024x1024);
        RUN_TEST(test_matmul_blocking_1024x1024);
        // RUN_TEST(test_matmul_2048x2048);
        // RUN_TEST(test_matmul_blocking_2048x2048);
        // RUN_TEST(test_sparse_matmul_1024x1024);
        // RUN_TEST(test_sparse_matmul_blocking_1024x1024);

    }
}