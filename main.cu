#include "SequentialMorphology.h"
#include "ParallelMorphology.cu"

int main() {
    cv::Mat inputImage = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    erosionCUDA(inputImage, outputImage);

    cv::imwrite("../output.png", outputImage);
    return 0;
}
