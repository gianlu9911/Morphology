#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

template <typename T>
__device__ void DilationStep1K(T* input, T* output, int* image_width, int* image_height, int* filter_size, int* filter_half_size, int row, int col) {
    int width = *image_width;
    int height = *image_height;
    int filter_size_ = *filter_size;
    int filter_half_size_ = *filter_half_size;

    T max_val = 0;
    for (int filter_row = -filter_half_size_; filter_row <= filter_half_size_; ++filter_row) {
        for (int filter_col = -filter_half_size_; filter_col <= filter_half_size_; ++filter_col) {
            int img_row = row + filter_row;
            int img_col = col + filter_col;
            if (img_row >= 0 && img_row < height && img_col >= 0 && img_col < width) {
                max_val = max(max_val, input[img_row * width + img_col]);
            }
        }
    }
    output[row * width + col] = max_val;
}

template <typename T>
__global__ void DilationKernel(T* input, T* output, int* image_width, int* image_height, int* filter_size, int* filter_half_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < *image_height && col < *image_width) {
        DilationStep1K(input, output, image_width, image_height, filter_size, filter_half_size, row, col);
    }
}

template <typename T>
void Dilation(T* input, T* output, int width, int height, int filter_size, int blocksize) {
    int* d_image_width;
    int* d_image_height;
    int* d_filter_size;
    int* d_filter_half_size;
    T* d_input;
    T* d_output;

    int filter_half_size = filter_size / 2;

    cudaMalloc(&d_image_width, sizeof(int));
    cudaMalloc(&d_image_height, sizeof(int));
    cudaMalloc(&d_filter_size, sizeof(int));
    cudaMalloc(&d_filter_half_size, sizeof(int));
    cudaMalloc(&d_input, width * height * sizeof(T));
    cudaMalloc(&d_output, width * height * sizeof(T));

    cudaMemcpy(d_image_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_height, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_size, &filter_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_half_size, &filter_half_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, width * height * sizeof(T), cudaMemcpyHostToDevice);

    dim3 blockSize(blocksize, blocksize);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    DilationKernel<<<gridSize, blockSize>>>(d_input, d_output, d_image_width, d_image_height, d_filter_size, d_filter_half_size);

    cudaMemcpy(output, d_output, width * height * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_image_width);
    cudaFree(d_image_height);
    cudaFree(d_filter_size);
    cudaFree(d_filter_half_size);
    cudaFree(d_input);
    cudaFree(d_output);
}

void apply_dilation(const std::string& image_path, const int filter_size, int block_size) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE); 
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return;
    }

    int width = img.cols;
    int height = img.rows;

    unsigned char* input_image = img.data;
    unsigned char* output_image = new unsigned char[width * height];

    Dilation(input_image, output_image, width, height, filter_size, block_size);

    cv::Mat result(height, width, CV_8UC1, output_image);  
    cv::imwrite("../imgs/lena_dilated.jpg", result);       
    cv::imshow("Dilated Image", result);            
    cv::waitKey(0);

    delete[] output_image;
}

int main() {
    std::string image_path = "../imgs/lena.jpg";
    int block_size = 32;
    int filter_size = 3;

    apply_dilation(image_path, filter_size, block_size);

    return 0;
}
