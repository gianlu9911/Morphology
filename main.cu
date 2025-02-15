#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vhgw_dilation_row(float *d_input, float *d_output, int width, int height, int radius) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y;

    if (col >= width || row >= height) return;

    // Load data into shared memory with boundary checks
    int sharedIdx = tx + radius;
    shared_mem[sharedIdx] = d_input[row * width + col];

    if (tx < radius) {
        shared_mem[sharedIdx - radius] = (col >= radius) ? d_input[row * width + col - radius] : d_input[row * width];
        shared_mem[sharedIdx + blockDim.x] = (col + blockDim.x < width) ? d_input[row * width + col + blockDim.x] : d_input[row * width + width - 1];
    }

    __syncthreads();

    // Compute max using boundary checks
    float max_value = shared_mem[sharedIdx];
    for (int i = 1; i <= radius; i++) {
        if (sharedIdx - i >= 0)
            max_value = max(max_value, shared_mem[sharedIdx - i]);
        if (sharedIdx + i < blockDim.x + 2 * radius)
            max_value = max(max_value, shared_mem[sharedIdx + i]);
    }

    d_output[row * width + col] = max_value;
}

__global__ void vhgw_dilation_col(float *d_input, float *d_output, int width, int height, int radius) {
    extern __shared__ float shared_mem[];
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x;

    if (row >= height || col >= width) return;

    int sharedIdx = ty + radius;
    shared_mem[sharedIdx] = d_input[row * width + col];

    if (ty < radius) {
        shared_mem[sharedIdx - radius] = (row >= radius) ? d_input[(row - radius) * width + col] : d_input[col];
        shared_mem[sharedIdx + blockDim.y] = (row + blockDim.y < height) ? d_input[(row + blockDim.y) * width + col] : d_input[(height - 1) * width + col];
    }

    __syncthreads();

    float max_value = shared_mem[sharedIdx];
    for (int i = 1; i <= radius; i++) {
        if (sharedIdx - i >= 0)
            max_value = max(max_value, shared_mem[sharedIdx - i]);
        if (sharedIdx + i < blockDim.y + 2 * radius)
            max_value = max(max_value, shared_mem[sharedIdx + i]);
    }

    d_output[row * width + col] = max_value;
}

void run_vhgw_dilation(float *h_input, float *h_output, int width, int height, int radius) {
    float *d_input, *d_intermediate, *d_output;
    size_t size = width * height * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_intermediate, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 blockSize(32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, height);
    size_t shared_mem_size = (blockSize.x + 2 * radius) * sizeof(float);

    vhgw_dilation_row<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_intermediate, width, height, radius);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 blockSizeCol(1, 32);
    dim3 gridSizeCol(width, (height + blockSizeCol.y - 1) / blockSizeCol.y);
    shared_mem_size = (blockSizeCol.y + 2 * radius) * sizeof(float);

    vhgw_dilation_col<<<gridSizeCol, blockSizeCol, shared_mem_size>>>(d_intermediate, d_output, width, height, radius);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_output));
}


int main() {
    std::string input_path = "../imgs/lena.jpg";
    std::string output_path = "../lena_dilated.jpg";
    

    int radius = 1;
    
    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }
    
    int width = image.cols;
    int height = image.rows;
    
    cv::Mat output_image(height, width, CV_8UC1);
    
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = static_cast<float>(image.at<uchar>(i, j));
        }
    }
    
    run_vhgw_dilation(h_input, h_output, width, height, radius);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output_image.at<uchar>(i, j) = static_cast<uchar>(h_output[i * width + j]);
        }
    }
    
    cv::imwrite(output_path, output_image);
    
    delete[] h_input;
    delete[] h_output;
    
    std::cout << "Dilation complete. Output saved to " << output_path << std::endl;
    
    return 0;
}
