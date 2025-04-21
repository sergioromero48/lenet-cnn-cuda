#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <unistd.h> // for sleep()

#ifdef __cplusplus
extern "C" {
#endif
#include "lenet.h"
#include "lenet_cuda.h"

int load(LeNet5 *lenet, const char *filename);
#ifdef __cplusplus
}
#endif

// define load() here so we don't need main.c
extern "C" int load(LeNet5 *lenet, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

const int MNIST_SIZE = 28;
//const int PADDED_SIZE = 32; (unused)
const char*  MODEL_FILE = "model/model.dat";
const char*  CSV_FILE   = "data/mnist_test-1.csv";

int main()
{
    // 1) Read first CSV row
    std::ifstream ifs(CSV_FILE);
    if (!ifs) {
        std::cerr << "Error: cannot open " << CSV_FILE << "\n";
        return 1;
    }
    std::string line;
    std::getline(ifs, line);
    ifs.close();

    std::stringstream ss(line);
    std::string tok;
    std::vector<unsigned char> mnist;
    mnist.reserve(MNIST_SIZE * MNIST_SIZE);

    /* discard leading label */
    std::getline(ss, tok, ',');

    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;          // skip blank tokens
        int val = std::stoi(tok);           // throws if not an int
        if (val < 0 || val > 255) {
            std::cerr << "Bad pixel value: " << val << '\n';
            return 1;
        }
        mnist.push_back(static_cast<unsigned char>(val));
    }

    if (mnist.size() != MNIST_SIZE*MNIST_SIZE) {
        std::cerr << "Error: expected "
                  << MNIST_SIZE*MNIST_SIZE
                  << " values, got " << mnist.size() << "\n";
        return 1;
    }

    // 2) Convert to image type
    image img;
    for(int y = 0; y < MNIST_SIZE; ++y)
      for(int x = 0; x < MNIST_SIZE; ++x)
        img[y][x] = mnist[y*MNIST_SIZE + x];

    // 3) Load or initialize LeNet5
    LeNet5 net;
    if ( load(&net, MODEL_FILE) ) {
        Initial(&net);
    }
    // initialize GPU buffers
    Init_CUDA(&net);

    // 4) Benchmark CPU Predict
    auto t0 = std::chrono::high_resolution_clock::now();
    uint8 cpu_pred = Predict(&net, img, 10);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // 5) Benchmark GPU Predict_CUDA using CUDA events for timing
    auto t0_gpu = std::chrono::high_resolution_clock::now();
    uint8 gpu_pred = Predict_CUDA(&net, img, 10);
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(t1_gpu - t0_gpu).count();

    // 6) Report
    std::cout
      << "Predict (CPU)       : label=" << int(cpu_pred)
      << "  time=" << cpu_us << " us\n"
      << "Predict_CUDA (GPU)  : label=" << int(gpu_pred)
      << "  time=" << gpu_us << " us\n";

    // cleanup GPU buffers
    Cleanup_CUDA();

    return 0;
}