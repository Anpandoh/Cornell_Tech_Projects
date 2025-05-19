#include "proflab4.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();
    // for (int i = 0; i < 10; i++)
    // {
    RUN_TEST(test_matmul_512x512);
    // }
}