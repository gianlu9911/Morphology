#pragma once
#include "MorphologicalOperators.h"
#include "Utility.h"
#include <chrono>

// The function just add text for displying the images after applining the morphological operators
void addLabel(cv::Mat& img, const std::string& label) {
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    cv::putText(img, label, cv::Point(10, 20), font, scale, cv::Scalar(255), thickness);
}

int sequentialMorphology(std::string img_path, int kernel_size, cv::Size res) {

    cv::Mat image_to_resize = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (image_to_resize.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::Mat image;
    cv::resize(image_to_resize, image, res);

    int width = image.cols;
    int height = image.rows;
    cv::Size size(200, 200);  // Resize all images to a fixed size for visualization

    cv::Mat eroded_image(height, width, CV_8UC1);
    cv::Mat dilated_image(height, width, CV_8UC1);
    cv::Mat opened_image(height, width, CV_8UC1);
    cv::Mat closed_image(height, width, CV_8UC1);
    cv::Mat gradient_image(height, width, CV_8UC1);
    cv::Mat tophat_image(height, width, CV_8UC1);
    cv::Mat blackhat_image(height, width, CV_8UC1);

    erosion(image.data, eroded_image.data, width, height, kernel_size);
    dilation(image.data, dilated_image.data, width, height, kernel_size);
    opening(image.data, opened_image.data, width, height, kernel_size);
    closure(image.data, closed_image.data, width, height, kernel_size);
    gradient(image.data, gradient_image.data, width, height, kernel_size);
    top_hat(image.data, tophat_image.data, width, height, kernel_size);
    black_hat(image.data, blackhat_image.data, width, height, kernel_size);

    // Resize images for grid display
    cv::resize(image, image, size);
    cv::resize(eroded_image, eroded_image, size);
    cv::resize(dilated_image, dilated_image, size);
    cv::resize(opened_image, opened_image, size);
    cv::resize(closed_image, closed_image, size);
    cv::resize(gradient_image, gradient_image, size);
    cv::resize(tophat_image, tophat_image, size);
    cv::resize(blackhat_image, blackhat_image, size);

    // Add labels
    addLabel(image, "Original");
    addLabel(eroded_image, "Erosion");
    addLabel(dilated_image, "Dilation");
    addLabel(opened_image, "Opening");
    addLabel(closed_image, "Closing");
    addLabel(gradient_image, "Gradient");
    addLabel(tophat_image, "Tophat");
    addLabel(blackhat_image, "Blackhat");

    // Create grid (2x4)
    cv::Mat row1, row2, grid;
    std::vector<cv::Mat> row1_images = {image, eroded_image, dilated_image, opened_image};
    cv::hconcat(row1_images, row1);
    std::vector<cv::Mat> row2_images = {closed_image, gradient_image, tophat_image, blackhat_image};
    cv::hconcat(row2_images, row2);
    cv::vconcat(row1, row2, grid);

    cv::imshow("Morphological Operations", grid);
    cv::waitKey(0);

    return 0;
}

void sequentialTest(std::string img_path) {
    std::cout << "Sequential Morphology Test" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Image: " << img_path << std::endl;

    std::vector<int> kernel_sizes = {3, 5, 7};
    std::vector<cv::Size> resolutions = { 
        cv::Size(256, 256), 
        cv::Size(512, 512), 
        cv::Size(1024, 1024) 
    };
    
    std::vector<std::string> operations = {"Erosion", "Dilation", "Opening", 
                                           "Closing", "Gradient", "Tophat", "Blackhat"};

   
    for (const auto& res : resolutions) {
        
        std::cout << "Testing at Resolution: " << res.width << "x" << res.height << std::endl;

        for (int kernel_size : kernel_sizes) {
            std::cout << "Kernel Size: " << kernel_size << std::endl;

            auto start = std::chrono::high_resolution_clock::now();
            sequentialMorphology(img_path, kernel_size, res);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            for (const auto& op : operations) {
                saveExecutionTimeCSV("results.csv", std::to_string(res.width) + "x" + std::to_string(res.height),
                                     duration.count(), op, "Sequential", kernel_size);
            }

            std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;
            std::cout << "--------------------------" << std::endl;
        }
    }
}
