#include "SequentialMorphology.h"

int main(){
    my_kernel<<<1,1>>>();  // Correct kernel launch syntax
    cudaDeviceSynchronize();  // Ensure kernel execution completes

    std::string input_image_path = "imgs/lena.jpg";
    sequentialTest(input_image_path);
    return 0;
}

