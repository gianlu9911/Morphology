#pragma once
#include <algorithm>    

void opening(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void closure(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void dilation(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void erosion(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void gradient(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void top_hat(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);
void black_hat(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size);


void erosion(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    int kernel_radius = kernel_size / 2;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char min_value = 255;
            for(int k = -kernel_radius; k <= kernel_radius; k++){
                for(int l = -kernel_radius; l <= kernel_radius; l++){
                    int x = i + k;
                    int y = j + l;
                    if(x >= 0 && x < height && y >= 0 && y < width){
                        min_value = std::min(min_value, input[x * width + y]);
                    }
                }
            }
            output[i * width + j] = min_value;
        }
    }
}

void dilation(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    int kernel_radius = kernel_size / 2;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char max_value = 0;
            for(int k = -kernel_radius; k <= kernel_radius; k++){
                for(int l = -kernel_radius; l <= kernel_radius; l++){
                    int x = i + k;
                    int y = j + l;
                    if(x >= 0 && x < height && y >= 0 && y < width){
                        max_value = std::max(max_value, input[x * width + y]);
                    }
                }
            }
            output[i * width + j] = max_value;
        }
    }
}

void opening(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    unsigned char* temp = new unsigned char[width * height];
    erosion(input, temp, width, height, kernel_size);
    dilation(temp, output, width, height, kernel_size);
    delete[] temp;
}

void closure(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    unsigned char* temp = new unsigned char[width * height];
    dilation(input, temp, width, height, kernel_size);
    erosion(temp, output, width, height, kernel_size);
    delete[] temp;
}

void top_hat(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    unsigned char* temp = new unsigned char[width * height];
    opening(input, temp, width, height, kernel_size);
    for(int i = 0; i < width * height; i++){
        output[i] = input[i] - temp[i];
    }
    delete[] temp;
}

void black_hat(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    unsigned char* temp = new unsigned char[width * height];
    closure(input, temp, width, height, kernel_size);
    for(int i = 0; i < width * height; i++){
        output[i] = temp[i] - input[i];
    }
    delete[] temp;
}

void gradient(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size){
    unsigned char* temp = new unsigned char[width * height];
    dilation(input, temp, width, height, kernel_size);
    erosion(input, output, width, height, kernel_size);
    for(int i = 0; i < width * height; i++){
        output[i] = temp[i] - output[i];
    }
    delete[] temp;
}