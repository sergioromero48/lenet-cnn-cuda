// benchmark_cuda.cu — minimal CUDA‑only inference timing for a single image
// Compile: nvcc -O2 -std=c++17 benchmark_cuda.cu -lcuda -lcudart -lm -lstdc++ -I.
// Usage  : ./benchmark_cuda [model.dat]
//   [model.dat]  — binary dump produced by training code (default "model/model.dat")
// Prints predicted label and elapsed GPU time in milliseconds.

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
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

// 28×28 image container
struct RawImage { uint8 px[28][28]; };

// Load first MNIST row from CSV (skip label)
static bool load_image_csv(const char *path, RawImage &img)
{
    std::ifstream f(path);
    if(!f) return false;
    std::string line;
    std::getline(f, line);
    std::stringstream ss(line);
    std::string tok;
    std::getline(ss, tok, ','); // skip label
    for(int y=0;y<28;++y)
        for(int x=0;x<28;++x) {
            if(!std::getline(ss, tok, ',')) return false;
            int v = std::stoi(tok);
            img.px[y][x] = static_cast<uint8>(v);
        }
    return true;
}

int main(int argc, char **argv)
{
    const char *csv_path   = "data/mnist_test-1.csv";
    const char *model_path = (argc>=2)? argv[1] : "model/model.dat";

    // 1) Load/init network on host
    LeNet5 net;
    if ( load_model(&net, model_path) ) {
        std::cerr << "Model not found (" << model_path << "), using random weights" << std::endl;
        Initial(&net);
    }
    lenet_cuda_init(&net);   // copy weights to device

    // 2) Load first MNIST sample from CSV
    RawImage img;
    if(!load_image_csv(csv_path, img)){
        std::cerr << "Failed to load CSV image " << csv_path << std::endl;
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
