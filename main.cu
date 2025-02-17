#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// Struct to encapsulate device image information.
struct DeviceImage {
    unsigned char* data;
    int width;
    int height;
    size_t pitch; // pitch (in bytes) for each row
};

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

//--------------------------------------------------
// Horizontal erosion kernel without shared memory
// Each thread loads its own left, center, and right pixels.
// No synchronization is needed since each thread works independently.
//--------------------------------------------------
__global__ void erosionHorizontalKernelNoSync(DeviceImage d_img, DeviceImage d_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (y < d_img.height && x < d_img.width) {
        const unsigned char* row = d_img.data + y * d_img.pitch;
        // Load neighbors from global memory
        unsigned char left   = (x > 0) ? row[x - 1] : 0;
        unsigned char center = row[x];
        unsigned char right  = (x < d_img.width - 1) ? row[x + 1] : 0;
        unsigned char minVal = min(min(left, center), right);

        unsigned char* outRow = d_out.data + y * d_out.pitch;
        outRow[x] = minVal;
    }
}

//--------------------------------------------------
// Vertical erosion kernel without shared memory
// Each thread loads its own top, center, and bottom pixels.
// No synchronization is needed since each thread works independently.
//--------------------------------------------------
__global__ void erosionVerticalKernelNoSync(DeviceImage d_img, DeviceImage d_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (y < d_img.height && x < d_img.width) {
        // Get pointer for current row
        const unsigned char* row = d_img.data + y * d_img.pitch;
        unsigned char top    = (y > 0) ? *(d_img.data + (y - 1) * d_img.pitch + x) : 0;
        unsigned char center = row[x];
        unsigned char bottom = (y < d_img.height - 1) ? *(d_img.data + (y + 1) * d_img.pitch + x) : 0;
        unsigned char minVal = min(min(top, center), bottom);

        unsigned char* outRow = d_out.data + y * d_out.pitch;
        outRow[x] = minVal;
    }
}

int main()
{
    // Load the binary (grayscale) image.
    cv::Mat binaryImg = cv::imread("../imgs/lena_binary_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (binaryImg.empty()) {
        std::cerr << "Error: Could not open image ../imgs/lena_binary_4k.jpg" << std::endl;
        return -1;
    }

    int width  = binaryImg.cols;
    int height = binaryImg.rows;
    size_t rowBytes = width * sizeof(unsigned char);

    // Allocate pitched device memory for input, intermediate, and output images.
    DeviceImage d_input, d_intermediate, d_output;
    cudaMallocPitch(&d_input.data, &d_input.pitch, rowBytes, height);
    cudaMallocPitch(&d_intermediate.data, &d_intermediate.pitch, rowBytes, height);
    cudaMallocPitch(&d_output.data, &d_output.pitch, rowBytes, height);

    d_input.width  = width; d_input.height  = height;
    d_intermediate.width = width; d_intermediate.height = height;
    d_output.width = width; d_output.height = height;

    // Copy host image data to device using cudaMemcpy2D.
    cudaMemcpy2D(d_input.data, d_input.pitch,
                 binaryImg.data, rowBytes,
                 rowBytes, height, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
              (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    // Launch the horizontal erosion kernel (no synchronization inside).
    erosionHorizontalKernelNoSync<<<grid, block>>>(d_input, d_intermediate);
    // cudaDeviceSynchronize();

    // Launch the vertical erosion kernel (no synchronization inside).
    erosionVerticalKernelNoSync<<<grid, block>>>(d_intermediate, d_output);
    // cudaDeviceSynchronize();

    // Copy the final result back to the host.
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy2D(result.data, rowBytes,
                 d_output.data, d_output.pitch,
                 rowBytes, height, cudaMemcpyDeviceToHost);

    // Save the result.
    cv::imwrite("../erosion_nosync_result.jpg", result);
    std::cout << "Erosion result saved as erosion_nosync_result.jpg" << std::endl;

    // Free device memory.
    cudaFree(d_input.data);
    cudaFree(d_intermediate.data);
    cudaFree(d_output.data);

    return 0;
}
