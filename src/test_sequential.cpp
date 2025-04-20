#include <iostream>
#include <cstring>  // for memset

const int MNIST_SIZE = 28;
const int PADDED_SIZE = 32;

void pad_mnist_to_32x32(const unsigned char* mnist_784, unsigned char* padded_1024) {
    // Step 1: Initialize 32x32 output to zeros
    std::memset(padded_1024, 0, PADDED_SIZE * PADDED_SIZE * sizeof(unsigned char));

    // Step 2: Calculate offset
    int offset = (PADDED_SIZE - MNIST_SIZE) / 2;

    // Step 3: Copy MNIST into center of 32x32
    for (int y = 0; y < MNIST_SIZE; y++) {
        for (int x = 0; x < MNIST_SIZE; x++) {
            int padded_index = (y + offset) * PADDED_SIZE + (x + offset);
            padded_1024[padded_index] = mnist_784[y * MNIST_SIZE + x];
        }
    }
}
