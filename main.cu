#include "Utility.h"
#include <opencv2/opencv.hpp>

int main(){
    my_kernel<<<1,1>>>();  // Correct kernel launch syntax
    cudaDeviceSynchronize();  // Ensure kernel execution completes

    cv::Mat image;
    image = cv::imread("imgs/lena.jpg");
    cv::imshow("Image", image);
    cv::waitKey(0);
    return 0;
}
