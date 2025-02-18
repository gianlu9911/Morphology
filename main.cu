#include "ParallelMorphology.cuh"

int main() {
    std::string imagePath = "../imgs/lena.jpg";
    parallelTest(imagePath);
    // sequentialTest(imagePath);
    return 0;
}
