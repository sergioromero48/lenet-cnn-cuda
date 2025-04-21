#pragma once
#include "lenet.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Convenience wrapper: full forward pass that
   calls conv1_cuda_forward, then finishes the
   remaining layers on the CPU. */
uint8 Predict_CUDA(const LeNet5* lenet, image img, uint8 count);

// initialize GPU device buffers and upload weights
void Init_CUDA(const LeNet5* lenet);
// free GPU device buffers
void Cleanup_CUDA(void);

void load_input(Feature*, image);   /* defined in lenet.c */

#ifdef __cplusplus
}
#endif
