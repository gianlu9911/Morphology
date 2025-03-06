#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA error check macro.
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in " << __FILE__                \
                      << " at line " << __LINE__ << ": "              \
                      << cudaGetErrorString(err) << std::endl;        \
            exit(err);                                                \
        }                                                             \
    } while (0)

// Functor for erosion: returns the minimum value.
struct ErosionOp {
    __device__ __host__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return a < b ? a : b;
    }
    // Identity element for erosion: maximum possible value.
    __device__ __host__ unsigned char identity() const { return 255; }
};

// Functor for dilation: returns the maximum value.
struct DilationOp {
    __device__ __host__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return a > b ? a : b;
    }
    // Identity element for dilation: minimum possible value.
    __device__ __host__ unsigned char identity() const { return 0; }
};

//
// Templated vertical kernel: processes one pixel per thread.
// It computes an extremum (min for erosion or max for dilation) over a vertical window.
//
template <typename Operation>
__global__ void verticalKernel(const unsigned char* input,
                               unsigned char* output,
                               int width, int height,
                               int radius,
                               Operation op)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx >= totalPixels) return;

    int col = idx % width;
    int row = idx / width;

    // Determine vertical neighborhood boundaries.
    int rowStart = max(0, row - radius);
    int rowEnd   = min(height - 1, row + radius);

    unsigned char result = op.identity();
    for (int r = rowStart; r <= rowEnd; ++r) {
        unsigned char pixel = input[r * width + col];
        result = op(result, pixel);
    }
    output[idx] = result;
}

//
// Templated horizontal kernel using shared memory for improved coalesced access.
// Each block processes a segment of a single row, loading a tile (plus left/right halo)
// from global memory into shared memory before processing.
//
template <typename Operation>
__global__ void horizontalKernelShared(const unsigned char* input,
                                       unsigned char* output,
                                       int width, int height,
                                       int radius,
                                       Operation op)
{
    // Each block processes one row.
    int row = blockIdx.y;
    // Global x coordinate.
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory: tile width plus halo on both sides.
    extern __shared__ unsigned char s_data[];
    int s_index = threadIdx.x + radius;

    // Load the center value.
    if (col < width)
        s_data[s_index] = input[row * width + col];
    else
        s_data[s_index] = op.identity();

    // Load left halo.
    if (threadIdx.x < radius) {
        int halo_col = blockIdx.x * blockDim.x + threadIdx.x - radius;
        s_data[threadIdx.x] = (halo_col >= 0) ? input[row * width + halo_col] : op.identity();
    }

    // Load right halo.
    int rightHaloIndex = threadIdx.x + blockDim.x + radius;
    int halo_col = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
    if (threadIdx.x < radius) {
        s_data[rightHaloIndex] = (halo_col < width) ? input[row * width + halo_col] : op.identity();
    }

    __syncthreads();

    // Perform the horizontal operation if within image bounds.
    if (col < width) {
        unsigned char result = op.identity();
        // Process the neighborhood: from s_index - radius to s_index + radius.
        for (int offset = -radius; offset <= radius; ++offset) {
            result = op(result, s_data[s_index + offset]);
        }
        output[row * width + col] = result;
    }
}

// Helper function to add a label to an image using OpenCV drawing functions.
void addLabel(cv::Mat& image, const std::string& label)
{
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    int baseline = 0;
    
    cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Place the text at the bottom-center of the image.
    cv::Point textOrg((image.cols - textSize.width) / 2, image.rows - 10);

    // Draw a filled rectangle as background.
    cv::rectangle(image, textOrg + cv::Point(0, baseline),
                  textOrg - cv::Point(-textSize.width, textSize.height),
                  cv::Scalar(0, 0, 0), cv::FILLED);

    // Draw the label text.
    cv::putText(image, label, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
}

int main()
{
    // Load the image in grayscale.
    cv::Mat image = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if(image.empty()){
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;
    size_t imgSize = totalPixels * sizeof(unsigned char);

    // Allocate device memory for the input image.
    unsigned char *d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMemcpy(d_input, image.ptr(), imgSize, cudaMemcpyHostToDevice));

    // Allocate buffers for the basic erosion and dilation pipelines.
    unsigned char *d_horizontal_erode = nullptr, *d_final_erode = nullptr;
    unsigned char *d_horizontal_dilate = nullptr, *d_final_dilate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_horizontal_erode, imgSize));
    CUDA_CHECK(cudaMalloc(&d_final_erode, imgSize));
    CUDA_CHECK(cudaMalloc(&d_horizontal_dilate, imgSize));
    CUDA_CHECK(cudaMalloc(&d_final_dilate, imgSize));

    // Set the structuring element radius.
    int radius = 3;

    // Configure the horizontal kernel launch: each block processes a segment of one row.
    int tileWidth = 256;
    dim3 blockDim(tileWidth, 1, 1);
    dim3 gridDim((width + tileWidth - 1) / tileWidth, height, 1);
    size_t sharedMemSize = (tileWidth + 2 * radius) * sizeof(unsigned char);

    // --- Erosion Pipeline (for later use in Opening) ---
    horizontalKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_horizontal_erode, width, height, radius, ErosionOp());
    CUDA_CHECK(cudaDeviceSynchronize());
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    verticalKernel<<<numBlocks, blockSize>>>(d_horizontal_erode, d_final_erode, width, height, radius, ErosionOp());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Dilation Pipeline (for later use in Closing) ---
    horizontalKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_horizontal_dilate, width, height, radius, DilationOp());
    CUDA_CHECK(cudaDeviceSynchronize());
    verticalKernel<<<numBlocks, blockSize>>>(d_horizontal_dilate, d_final_dilate, width, height, radius, DilationOp());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate additional buffers for Opening and Closing operations.
    unsigned char *d_horizontal_open_dilate = nullptr, *d_final_open = nullptr;
    unsigned char *d_horizontal_close_erode = nullptr, *d_final_close = nullptr;
    CUDA_CHECK(cudaMalloc(&d_horizontal_open_dilate, imgSize));
    CUDA_CHECK(cudaMalloc(&d_final_open, imgSize));
    CUDA_CHECK(cudaMalloc(&d_horizontal_close_erode, imgSize));
    CUDA_CHECK(cudaMalloc(&d_final_close, imgSize));

    // --- Opening: erosion followed by dilation ---
    horizontalKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_final_erode, d_horizontal_open_dilate, width, height, radius, DilationOp());
    CUDA_CHECK(cudaDeviceSynchronize());
    verticalKernel<<<numBlocks, blockSize>>>(d_horizontal_open_dilate, d_final_open, width, height, radius, DilationOp());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Closing: dilation followed by erosion ---
    horizontalKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_final_dilate, d_horizontal_close_erode, width, height, radius, ErosionOp());
    CUDA_CHECK(cudaDeviceSynchronize());
    verticalKernel<<<numBlocks, blockSize>>>(d_horizontal_close_erode, d_final_close, width, height, radius, ErosionOp());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy basic results back to host.
    cv::Mat outputErode(height, width, CV_8UC1);
    cv::Mat outputDilate(height, width, CV_8UC1);
    cv::Mat outputOpen(height, width, CV_8UC1);
    cv::Mat outputClose(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(outputErode.ptr(), d_final_erode, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputDilate.ptr(), d_final_dilate, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputOpen.ptr(), d_final_open, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputClose.ptr(), d_final_close, imgSize, cudaMemcpyDeviceToHost));

    // --- Compute Additional Morphological Operations on the Host ---
    // Gradient = Dilated - Eroded
    cv::Mat gradient;
    cv::subtract(outputDilate, outputErode, gradient);
    
    // Top-hat = Original - Opening
    cv::Mat topHat;
    cv::subtract(image, outputOpen, topHat);
    
    // Black-hat = Closing - Original
    cv::Mat blackHat;
    cv::subtract(outputClose, image, blackHat);

    // Add labels to the images.
    addLabel(gradient, "Gradient");
    addLabel(topHat, "Tophat");
    addLabel(blackHat, "Blackhat");

    // Display all images.
    cv::imshow("Original Image", image);
    cv::imshow("Eroded Image", outputErode);
    cv::imshow("Dilated Image", outputDilate);
    cv::imshow("Opening (Erosion then Dilation)", outputOpen);
    cv::imshow("Closing (Dilation then Erosion)", outputClose);
    cv::imshow("Gradient", gradient);
    cv::imshow("Tophat", topHat);
    cv::imshow("Blackhat", blackHat);
    cv::waitKey(0);

    // Cleanup device memory.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_horizontal_erode));
    CUDA_CHECK(cudaFree(d_final_erode));
    CUDA_CHECK(cudaFree(d_horizontal_dilate));
    CUDA_CHECK(cudaFree(d_final_dilate));
    CUDA_CHECK(cudaFree(d_horizontal_open_dilate));
    CUDA_CHECK(cudaFree(d_final_open));
    CUDA_CHECK(cudaFree(d_horizontal_close_erode));
    CUDA_CHECK(cudaFree(d_final_close));

    return 0;
}
