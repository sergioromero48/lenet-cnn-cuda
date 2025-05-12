// benchmark_cuda.cu — measure inference latency of predict_cuda()
// Compile with: nvcc -O2 -std=c++17 benchmark_cuda.cu -lcuda -lcudart -lm -lstdc++ -I. 
// Usage   : ./benchmark_cuda [iterations]
// Default iterations = 1.  Prints single‑shot latency and the average over N runs.

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "lenet.h"
#include "lenet_cuda.h"
}

// Simple binary loader so we do not depend on main.c
static int load_lenet(LeNet5 *net, const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(net, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

static image load_first_mnist_row(const char *csv_path)
{
    image img = {0};
    std::ifstream ifs(csv_path);
    if (!ifs) {
        std::cerr << "Error: cannot open " << csv_path << "\n";
        std::exit(1);
    }
    std::string line; std::getline(ifs,line); ifs.close();
    std::stringstream ss(line); std::string tok;
    // discard leading label
    std::getline(ss, tok, ',');
    for(int y=0;y<28;++y)
        for(int x=0;x<28;++x){
            if(!std::getline(ss,tok,',')) {
                std::cerr << "CSV parse error\n"; std::exit(1);
            }
            int v = std::stoi(tok);
            img[y][x] = static_cast<uint8>(v);
        }
    return img;
}

int main(int argc, char **argv)
{
    const char *MODEL_FILE = "model/model.dat";
    const char *CSV_FILE   = "data/mnist_test-1.csv";
    int runs = (argc>1)? std::atoi(argv[1]) : 1;
    if(runs<=0) runs=1;

    // Load first sample
    image sample = load_first_mnist_row(CSV_FILE);

    // Load or initialize network
    LeNet5 net; if(load_lenet(&net, MODEL_FILE)) Initial(&net);

    // Init CUDA buffers
    lenet_cuda_init(&net);

    // Warm‑up once to avoid first‑launch overhead
    predict_cuda(&net, sample, 10);
    cudaDeviceSynchronize();

    // Benchmark loop using CUDA events
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_ms = 0.0f;
    uint8 last_pred = 0;

    for(int i=0;i<runs;++i){
        cudaEventRecord(start);
        last_pred = predict_cuda(&net, sample, 10);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f; cudaEventElapsedTime(&ms,start,stop);
        total_ms += ms;
        if(runs==1) std::cout << "Run latency: " << ms*1000.0f << " µs\n"; // detailed for single run
    }

    float avg_ms = total_ms / runs;
    std::cout << "\nCUDA Predict label : " << int(last_pred) << '\n';
    std::cout << "Iterations         : " << runs << '\n';
    std::cout << "Average latency    : " << avg_ms*1000.0f << " µs (" << avg_ms << " ms)\n";

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    lenet_cuda_free();
    return 0;
}
