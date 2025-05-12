// benchmark_cuda.cu — minimal CUDA‑only inference timing for a single image
// Compile: nvcc -O2 -std=c++17 benchmark_cuda.cu -lcuda -lcudart -lm -lstdc++ -I.
// Usage  : ./benchmark_cuda <image.raw> [model.dat]
//   <image.raw>  — 784‑byte 28×28 grayscale file (MNIST format, no header)
//   [model.dat]  — binary dump produced by training code (default "model/model.dat")
// Prints predicted label and elapsed GPU time in milliseconds.

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

extern "C" {
#include "lenet.h"         // LeNet5, load/Initial declarations
#include "lenet_cuda.h"    // predict_cuda(), lenet_cuda_init(), lenet_cuda_free()
}

// Simple helper identical to the one used in test.cpp
static int load_model(LeNet5 *net, const char *file)
{
    FILE *fp = fopen(file, "rb");
    if (!fp) return 1;
    fread(net, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

// 28×28 image container matching typedef image in lenet.h
struct RawImage {
    uint8 px[28][28];
};

static bool load_image(const char *path, RawImage &img)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    f.read(reinterpret_cast<char*>(img.px), sizeof(img.px));
    return f.good();
}

int main(int argc, char **argv)
{
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <image.raw> [model.dat]\n";
        return 1;
    }

    const char *img_path   = argv[1];
    const char *model_path = (argc >= 3) ? argv[2] : "model/model.dat";

    // 1) Load / init network on host
    LeNet5 net;
    if ( load_model(&net, model_path) ) {
        std::cerr << "Model not found (" << model_path << "), using random weights" << std::endl;
        Initial(&net);
    }
    lenet_cuda_init(&net);   // copy weights to device

    // 2) Load image
    RawImage img;
    if(!load_image(img_path, img)){
        std::cerr << "Failed to load image " << img_path << std::endl;
        lenet_cuda_free();
        return 2;
    }

    // 3) Time a single GPU inference
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    uint8 pred = predict_cuda(&net, img.px, 10);   // forward pass
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    float us = ms * 1000.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    std::cout << "Prediction: " << int(pred) << " | Latency: " << ms << " ms" << std::endl;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    lenet_cuda_free();
    return 0;
}
