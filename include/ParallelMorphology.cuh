#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include "KernelsAndStructs.cuh"
#include "SequentialMorphology.h"


#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define SE_SIZE 3  // Structuring Element size (change to 3, 5, 7, etc.)
#define SE_RADIUS (SE_SIZE / 2)  // Radius of structuring element

//-----------------------------------------------------------------------
// Main processing function implementing all morphological operations.
int parallelTest(std::string path) {
    // Load the input image in grayscale.
    cv::Mat binaryImg = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (binaryImg.empty()) {
        std::cerr << "Error: Could not open image " << path << std::endl;
        return -1;
    }

    int width = binaryImg.cols;
    int height = binaryImg.rows;
    size_t rowBytes = width * sizeof(unsigned char);
    std::string resolution = std::to_string(width) + "x" + std::to_string(height) + "_SE" + std::to_string(SE_SIZE);

    // Allocate pinned host memory for the input and for all outputs.
    unsigned char *h_input, *h_erosion, *h_dilation, *h_opening, *h_closing, *h_tophat, *h_blackhat;
    cudaMallocHost((void**)&h_input, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_erosion, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_dilation, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_opening, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_closing, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_tophat, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_blackhat, width * height * sizeof(unsigned char));

    memcpy(h_input, binaryImg.data, width * height * sizeof(unsigned char));

    // Allocate device images.
    DeviceImage d_input, d_temp;
    // We'll reuse d_temp for intermediate steps.
    allocateDeviceImage(d_input, width, height);
    allocateDeviceImage(d_temp, width, height);

    // Additional device images for storing specific operation results.
    DeviceImage d_erosion, d_dilation, d_opening, d_closing, d_tophat, d_blackhat;
    allocateDeviceImage(d_erosion, width, height);
    allocateDeviceImage(d_dilation, width, height);
    allocateDeviceImage(d_opening, width, height);
    allocateDeviceImage(d_closing, width, height);
    allocateDeviceImage(d_tophat, width, height);
    allocateDeviceImage(d_blackhat, width, height);

    // Copy input image to device.
    cudaMemcpy2D(d_input.data, d_input.pitch, h_input, rowBytes, rowBytes, height, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // -----------------------
    // 1. Erosion: erosion = erosion(original)
    // -----------------------
    {
        MinOp minOp;
        cudaEventRecord(start);
        // Horizontal pass.
        morphOperationKernel<<<grid, block>>>(d_input, d_temp, true, minOp);
        // Vertical pass.
        morphOperationKernel<<<grid, block>>>(d_temp, d_erosion, false, minOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Erosion time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Erosion", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_erosion, rowBytes, d_erosion.data, d_erosion.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat erosionResult(height, width, CV_8UC1, h_erosion);
        cv::imwrite("../erosion_result.jpg", erosionResult);
    }

    // -----------------------
    // 2. Dilation: dilation = dilation(original)
    // -----------------------
    {
        MaxOp maxOp;
        cudaEventRecord(start);
        // Horizontal pass.
        morphOperationKernel<<<grid, block>>>(d_input, d_temp, true, maxOp);
        // Vertical pass.
        morphOperationKernel<<<grid, block>>>(d_temp, d_dilation, false, maxOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Dilation time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Dilation", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_dilation, rowBytes, d_dilation.data, d_dilation.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat dilationResult(height, width, CV_8UC1, h_dilation);
        cv::imwrite("../dilation_result.jpg", dilationResult);
    }

    // -----------------------
    // 3. Opening: opening = dilation(erosion(original))
    // -----------------------
    {
        MinOp minOp;
        MaxOp maxOp;
        cudaEventRecord(start);
        // First, compute erosion.
        morphOperationKernel<<<grid, block>>>(d_input, d_temp, true, minOp);
        morphOperationKernel<<<grid, block>>>(d_temp, d_opening, false, minOp);
        // Then, apply dilation on the erosion result.
        morphOperationKernel<<<grid, block>>>(d_opening, d_temp, true, maxOp);
        morphOperationKernel<<<grid, block>>>(d_temp, d_opening, false, maxOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Opening time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Opening", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_opening, rowBytes, d_opening.data, d_opening.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat openingResult(height, width, CV_8UC1, h_opening);
        cv::imwrite("../opening_result.jpg", openingResult);
        std::cout << "Opening computed." << std::endl;
    }

    // -----------------------
    // 4. Closing: closing = erosion(dilation(original))
    // -----------------------
    {
        MaxOp maxOp;
        MinOp minOp;
        cudaEventRecord(start);
        // First, compute dilation.
        morphOperationKernel<<<grid, block>>>(d_input, d_temp, true, maxOp);
        morphOperationKernel<<<grid, block>>>(d_temp, d_closing, false, maxOp);
        // Then, apply erosion on the dilation result.
        morphOperationKernel<<<grid, block>>>(d_closing, d_temp, true, minOp);
        morphOperationKernel<<<grid, block>>>(d_temp, d_closing, false, minOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Closing time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Closing", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_closing, rowBytes, d_closing.data, d_closing.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat closingResult(height, width, CV_8UC1, h_closing);
        cv::imwrite("../closing_result.jpg", closingResult);
        std::cout << "Closing computed." << std::endl;
    }

    // -----------------------
    // 5. Top-hat: top-hat = original - opening
    // -----------------------
    {
        cudaEventRecord(start);
        subtractKernel<<<grid, block>>>(d_input, d_opening, d_tophat);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Top-hat time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Top-hat", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_tophat, rowBytes, d_tophat.data, d_tophat.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat tophatResult(height, width, CV_8UC1, h_tophat);
        cv::imwrite("../tophat_result.jpg", tophatResult);
        std::cout << "Top-hat computed." << std::endl;
    }

    // -----------------------
    // 6. Black-hat: black-hat = closing - original
    // -----------------------
    {
        cudaEventRecord(start);
        subtractKernel<<<grid, block>>>(d_closing, d_input, d_blackhat);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Black-hat time: " << elapsedTime / 1000.0 << " sec" << std::endl;
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Black-hat", "CUDA", SE_SIZE);
        cudaMemcpy2D(h_blackhat, rowBytes, d_blackhat.data, d_blackhat.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cv::Mat blackhatResult(height, width, CV_8UC1, h_blackhat);
        cv::imwrite("../blackhat_result.jpg", blackhatResult);
        std::cout << "Black-hat computed." << std::endl;
    }

    // Cleanup: free device and host memory.
    cudaFree(d_input.data);
    cudaFree(d_temp.data);
    cudaFree(d_erosion.data);
    cudaFree(d_dilation.data);
    cudaFree(d_opening.data);
    cudaFree(d_closing.data);
    cudaFree(d_tophat.data);
    cudaFree(d_blackhat.data);
    cudaFreeHost(h_input);
    cudaFreeHost(h_erosion);
    cudaFreeHost(h_dilation);
    cudaFreeHost(h_opening);
    cudaFreeHost(h_closing);
    cudaFreeHost(h_tophat);
    cudaFreeHost(h_blackhat);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}