#include "SequentialMorphology.h"
#include "ParallelMorphology.cuh"

int main() {
    
    std::string imagePath = "../imgs/lena_4k.jpg";
    parallelTest(imagePath);
    return 0;
}
