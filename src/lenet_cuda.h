#pragma once
#include "lenet.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Convenience wrapper: full forward pass that
   calls conv1_cuda_forward, then finishes the
   remaining layers on the CPU. */
uint8 Predict_CUDA(const LeNet5* lenet, image img, uint8 count);

#ifdef __cplusplus
}
#endif
