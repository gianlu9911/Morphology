#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

// Struct to encapsulate device image information.
struct DeviceImage {
    unsigned char* data;
    int width;
    int height;
    size_t pitch;
};

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define SE_SIZE 3  // Structuring Element size (change this to 3, 5, 7, etc.)

// Morphological operators
struct MinOp {
    __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return (a < b) ? a : b;
    }
};

struct MaxOp {
    __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return (a > b) ? a : b;
    }
};

// Kernel function supporting configurable structuring element size
template <typename Op>
__global__ void morphOperationKernel(DeviceImage d_img, DeviceImage d_out, bool horizontal, Op op) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < d_img.height && x < d_img.width) {
        unsigned char result = d_img.data[y * d_img.pitch + x];

        if (horizontal) {
            for (int dx = -SE_SIZE / 2; dx <= SE_SIZE / 2; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < d_img.width) {
                    result = op(result, d_img.data[y * d_img.pitch + nx]);
                }
            }
        } else {
            for (int dy = -SE_SIZE / 2; dy <= SE_SIZE / 2; dy++) {
                int ny = y + dy;
                if (ny >= 0 && ny < d_img.height) {
                    result = op(result, d_img.data[ny * d_img.pitch + x]);
                }
            }
        }

        d_out.data[y * d_out.pitch + x] = result;
    }
}

int parallelTest(std::string path){
    cv::Mat binaryImg = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (binaryImg.empty()) {
        std::cerr << "Error: Could not open image ../imgs/lena_4k.jpg" << std::endl;
        return -1;
    }

    int width = binaryImg.cols;
    int height = binaryImg.rows;
    size_t rowBytes = width * sizeof(unsigned char);
    std::string resolution = std::to_string(width) + "x" + std::to_string(height) +  std::to_string(SE_SIZE);

    // Allocate pinned memory for host input and output images
    unsigned char *h_input, *h_erosion, *h_dilation;
    cudaMallocHost((void**)&h_input, width * height * sizeof(unsigned char)); 
    cudaMallocHost((void**)&h_erosion, width * height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_dilation, width * height * sizeof(unsigned char));

    memcpy(h_input, binaryImg.data, width * height * sizeof(unsigned char));

    // Allocate device memory using cudaMallocPitch
    DeviceImage d_input, d_intermediate, d_output;
    cudaMallocPitch(&d_input.data, &d_input.pitch, rowBytes, height);
    cudaMallocPitch(&d_intermediate.data, &d_intermediate.pitch, rowBytes, height);
    cudaMallocPitch(&d_output.data, &d_output.pitch, rowBytes, height);

    d_input.width = width; d_input.height = height;
    d_intermediate.width = width; d_intermediate.height = height;
    d_output.width = width; d_output.height = height;

    // Use cudaMemcpyAsync for faster transfers
    cudaMemcpy2DAsync(d_input.data, d_input.pitch, h_input, rowBytes, width, height, cudaMemcpyHostToDevice);

    // Define CUDA grid/block dimensions
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    // CUDA Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // -----------------------
    // Erosion
    // -----------------------
    {
        MinOp minOp;
        cudaEventRecord(start);
        morphOperationKernel<<<grid, block>>>(d_input, d_intermediate, true, minOp);
        morphOperationKernel<<<grid, block>>>(d_intermediate, d_output, false, minOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        // Copy result back to host asynchronously
        cudaMemcpy2DAsync(h_erosion, rowBytes, d_output.data, d_output.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Save result
        cv::Mat erosionResult(height, width, CV_8UC1, h_erosion);
        cv::imwrite("../erosion_result.jpg", erosionResult);
        std::cout << "Erosion result saved as erosion_result.jpg" << std::endl;

        // Save execution time
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Erosion", "CUDA", SE_SIZE);
    }

    // -----------------------
    // Dilation
    // -----------------------
    {
        MaxOp maxOp;
        cudaEventRecord(start);
        morphOperationKernel<<<grid, block>>>(d_input, d_intermediate, true, maxOp);
        morphOperationKernel<<<grid, block>>>(d_intermediate, d_output, false, maxOp);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        // Copy result back to host asynchronously
        cudaMemcpy2DAsync(h_dilation, rowBytes, d_output.data, d_output.pitch, rowBytes, height, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Save result
        cv::Mat dilationResult(height, width, CV_8UC1, h_dilation);
        cv::imwrite("../dilation_result.jpg", dilationResult);
        std::cout << "Dilation result saved as dilation_result.jpg" << std::endl;

        // Save execution time
        saveExecutionTimeCSV("../execution_times.csv", resolution, elapsedTime / 1000.0, "Dilation", "CUDA", SE_SIZE);
    }

    // Cleanup
    cudaFree(d_input.data);
    cudaFree(d_intermediate.data);
    cudaFree(d_output.data);
    cudaFreeHost(h_input);
    cudaFreeHost(h_erosion);
    cudaFreeHost(h_dilation);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}