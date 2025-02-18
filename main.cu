#include "SequentialMorphology.h"
#include "ParallelMorphology.cuh"

int main() {
    // sequentialTest("../imgs/lena_4k.jpg");
    parallelTest("../imgs/lena_4k.jpg");
    return 0;
}
