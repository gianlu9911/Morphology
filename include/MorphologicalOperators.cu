#include <cuda_runtime.h>
#include <cuda.h>

__global__ void dilationKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char max_value = 0;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                max_value = max(max_value, input[new_index]);
            }
        }
    }
    output[index] = max_value;
}

__global__ void erosionKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char min_value = 255;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                min_value = min(min_value, input[new_index]);
            }
        }
    }
    output[index] = min_value;
}

__global__ void openingKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char min_value = 255;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                min_value = min(min_value, input[new_index]);
            }
        }
    }
    output[index] = min_value;
}

__global__ void closingKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char max_value = 0;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                max_value = max(max_value, input[new_index]);
            }
        }
    }
    output[index] = max_value;
}

__global__ void gradientKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char max_value = 0;
    unsigned char min_value = 255;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                max_value = max(max_value, input[new_index]);
                min_value = min(min_value, input[new_index]);
            }
        }
    }
    output[index] = max_value - min_value;
}

__global__ void tophatKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char min_value = 255;
    unsigned char max_value = 0;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                min_value = min(min_value, input[new_index]);
                max_value = max(max_value, input[new_index]);
            }
        }
    }
    output[index] = max_value - min_value;
}

__global__ void blackhatKernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int kernel_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int index = y * width + x;
    unsigned char min_value = 255;
    unsigned char max_value = 0;
    for(int i = -kernel_radius; i <= kernel_radius; i++){
        for(int j = -kernel_radius; j <= kernel_radius; j++){
            int new_x = x + i;
            int new_y = y + j;
            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height){
                int new_index = new_y * width + new_x;
                min_value = min(min_value, input[new_index]);
                max_value = max(max_value, input[new_index]);
            }
        }
    }
    output[index] = min_value - max_value;
}