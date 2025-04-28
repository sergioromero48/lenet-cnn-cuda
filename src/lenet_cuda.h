#pragma once
#include "lenet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* implemented in a .cu file */
uint8 predict_cuda(LeNet5 *lenet, image input, uint8 count);
void lenet_cuda_init(const LeNet5 *hostNet);
void lenet_cuda_free();

#ifdef __cplusplus
}
#endif
