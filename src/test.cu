#include <cuda_runtime.h>
#include "lenet_cuda.h"

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array)/sizeof(double))
#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define TILE_SIZE  256

__device__ double relu(double x) {
    return x*(x > 0);
}

__global__ void conv_forward_kernel(
    const double* __restrict__ input,   // [InC][InH][InW], flattened
          double* __restrict__ output,  // [OutC][OutH][OutW], flattened
    const double* __restrict__ weight,  // [InC][OutC][KH][KW], flattened
    const double* __restrict__ bias,    // [OutC]
    int InputChannels, int OutputChannels,
    int InputHeight, int InputWidth,
    int KernelHeight,  int KernelWidth
) {
    // pick one output‐channel + pixel per thread
    int oc = blockIdx.z;                                 // 0..OutC-1
    int o1 = blockIdx.x*blockDim.x + threadIdx.x;        // col 0..OutW-1
    int o0 = blockIdx.y*blockDim.y + threadIdx.y;        // row 0..OutH-1

    int OutH = InputHeight - KernelHeight + 1;
    int OutW = InputWidth - KernelWidth + 1;
    if (oc >= OutputChannels || o0 >= OutH || o1 >= OutW) return;

    // === sum over input channels and kernel window ===
    double acc = 0.0;
    for (int ic = 0; ic < InputChannels; ++ic) {
        // pointer to input plane ic
        const double* in_plane = input  + ic * (InputHeight * InputWidth);
        // pointer to filter for (ic→oc)
        const double* filt     = weight + ((ic*OutputChannels + oc) * (KernelHeight*KernelWidth));

        for (int w0 = 0; w0 < KernelHeight; ++w0) {
            for (int w1 = 0; w1 < KernelWidth; ++w1) {
                acc += in_plane[(o0 + w0)*InputWidth + (o1 + w1)]
                     * filt[w0*KernelWidth      +  w1];
            }
        }
    }

    // === add bias, apply ReLU, store ===
    double v = relu(acc + bias[oc]);
    output[(oc*OutH + o0)*OutW + o1] = v;
}

__global__
void subsample_max_kernel(
    const double* __restrict__ input_data,
          double* __restrict__ output_data,
    int channel_count,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    // 1) Compute how big each pooling window is:
    //    len0 = number of input rows pooled into one output row
    //    len1 = number of input cols pooled into one output col
    int window_height = input_height  / output_height;
    int window_width  = input_width   / output_width;

    // 2) Figure out which (channel, output_row, output_col) this thread handles:
    int channel_index   = blockIdx.z;                            // 0 ≤ channel_index < channel_count
    int output_column   = blockIdx.x*blockDim.x + threadIdx.x;   // 0 ≤ output_column < output_width
    int output_row      = blockIdx.y*blockDim.y + threadIdx.y;   // 0 ≤ output_row < output_height

    // 3) Bounds check: exit if we landed outside the real output dimensions
    if (channel_index >= channel_count ||
        output_row    >= output_height ||
        output_column >= output_width) {
        return;
    }

    // 4) Compute base offsets into the flat arrays
    //    a) Start of this channel in the input array
    int input_channel_offset  = channel_index * input_height * input_width;
    //    b) Start of this channel in the output array
    int output_channel_offset = channel_index * output_height * output_width;

    // 5) Compute the top‐left corner of our pooling window in the input:
    //    row_start = output_row * window_height
    //    col_start = output_column * window_width
    int input_row_start    = output_row    * window_height;
    int input_col_start    = output_column * window_width;

    // 6) Initialize max_value to the first element in the window
    int first_index = input_channel_offset
                    + input_row_start * input_width
                    + input_col_start;
    double max_value = input_data[first_index];

    // 7) Now slide over the full window (nested loops):
    for (int row_offset = 0; row_offset < window_height; ++row_offset) {
        for (int col_offset = 0; col_offset < window_width; ++col_offset) {
            int in_row = input_row_start + row_offset;
            int in_col = input_col_start + col_offset;
            int in_index = input_channel_offset
                         + in_row * input_width
                         + in_col;
            double candidate = input_data[in_index];
            if (candidate > max_value) {
                max_value = candidate;
            }
        }
    }

    // 8) Write the maximum into the corresponding output cell
    int out_index = output_channel_offset
                  + output_row * output_width
                  + output_column;
    output_data[out_index] = max_value;
}

__global__
void dot_product_forward_kernel(
    const double * __restrict__ input_data,
          double * __restrict__ output_data,
    const double * __restrict__ weight_matrix,
    const double * __restrict__ bias_vector,
    int InputLength,
    int OutputLength
) {
    // 1) Identify which output neuron this thread computes:
    //    threadIdx.x + blockIdx.x*blockDim.x → output index j
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // 2) Bounds check: if outside range, do nothing
    if (outputIndex >= OutputLength) {
        return;
    }

    // 3) Compute dot product:
    //    Σ over all InputLength inputs:
    //      input_data[i] * weight_matrix[i][outputIndex]
    double accumulator = 0.0;
    for (int inputIndex = 0; inputIndex < InputLength; ++inputIndex) {
        // Flattened index into weight_matrix:
        // weight_matrix[inputIndex][outputIndex]
        int weightIndex = inputIndex * OutputLength + outputIndex;
        accumulator += input_data[inputIndex] * weight_matrix[weightIndex];
    }

    // 4) Add bias and apply ReLU activation:
    double activated = relu(accumulator + bias_vector[outputIndex]);

    // 5) Write result into the output vector:
    output_data[outputIndex] = activated;
}

extern "C"
void forward_cuda(LeNet5 *lenet, Feature *features)
{
    // ---- Layer 1: Conv1 (1×32×32 → 6×28×28), 5×5 kernels ----
    {
        const int inChannels    = 1;            // 1
        const int outChannels   = 6;           // 6
        const int inHeight      = 32;  // 32
        const int inWidth       = 32;  // 32
        const int kernelHeight  = 5;    // 5
        const int kernelWidth   = 5;    // 5
        const int outHeight     = 28;  // 28
        const int outWidth      = 28;  // 28

        dim3 block(16,16);
        dim3 grid(
          (outWidth  + block.x - 1)/block.x,
          (outHeight + block.y - 1)/block.y,
          outChannels
        );

        conv_forward_kernel<<<grid,block>>>(
            (const double*)features->input,
            (      double*)features->layer1,
            (const double*)lenet->weight0_1,
            (const double*)lenet->bias0_1,
            inChannels, outChannels,
            inHeight,   inWidth,
            kernelHeight, kernelWidth
        );
    }

    // ---- Layer 2: Pool1 (6×28×28 → 6×14×14), 2×2 max-pool ----
    {
        const int channels   = 6;            // 6
        const int inHeight   = 28;   // 28
        const int inWidth    = 28;   // 28
        const int outHeight  = 14;   // 14
        const int outWidth   = 14;   // 14

        dim3 block(16,16);
        dim3 grid(
          (outWidth  + block.x - 1)/block.x,
          (outHeight + block.y - 1)/block.y,
          channels
        );

        subsample_max_kernel<<<grid,block>>>(
            (const double*)features->layer1,
            (      double*)features->layer2,
            channels,
            inHeight, inWidth,
            outHeight, outWidth
        );
    }

    // ---- Layer 3: Conv2 (6×14×14 → 16×10×10), 5×5 kernels ----
    {
        const int inChannels    = 6;           // 6
        const int outChannels   = 16;           // 16
        const int inHeight      = 14;  // 14
        const int inWidth       = 14;  // 14
        const int kernelHeight  = 5;    // 5
        const int kernelWidth   = 5;    // 5
        const int outHeight     = 10;  // 10
        const int outWidth      = 10;  // 10

        dim3 block(16,16);
        dim3 grid(
          (outWidth  + block.x - 1)/block.x,
          (outHeight + block.y - 1)/block.y,
          outChannels
        );

        conv_forward_kernel<<<grid,block>>>(
            (const double*)features->layer2,
            (      double*)features->layer3,
            (const double*)lenet->weight2_3,
            (const double*)lenet->bias2_3,
            inChannels, outChannels,
            inHeight,   inWidth,
            kernelHeight, kernelWidth
        );
    }

    // ---- Layer 4: Pool2 (16×10×10 → 16×5×5), 2×2 max-pool ----
    {
        const int channels   = 16;            // 16
        const int inHeight   = 10;   // 10
        const int inWidth    = 10;   // 10
        const int outHeight  = 5;   // 5
        const int outWidth   = 5;   // 5

        dim3 block(16,16);
        dim3 grid(
          (outWidth  + block.x - 1)/block.x,
          (outHeight + block.y - 1)/block.y,
          channels
        );

        subsample_max_kernel<<<grid,block>>>(
            (const double*)features->layer3,
            (      double*)features->layer4,
            channels,
            inHeight, inWidth,
            outHeight, outWidth
        );
    }

    // ---- Layer 5: Conv3 (16×5×5 → 120×1×1), 5×5 kernels ----
    {
        const int inChannels    = 16;           // 16
        const int outChannels   = 120;           // 120
        const int inHeight      = 5;  // 5
        const int inWidth       = 5;  // 5
        const int kernelHeight  = 5;    // 5
        const int kernelWidth   = 5;    // 5
        // valid conv → outHeight = outWidth = LENGTH_FEATURE5 = 1

        dim3 block(1,1);
        dim3 grid(outChannels);

        conv_forward_kernel<<<grid,block>>>(
            (const double*)features->layer4,
            (      double*)features->layer5,
            (const double*)lenet->weight4_5,
            (const double*)lenet->bias4_5,
            inChannels, outChannels,
            inHeight,   inWidth,
            kernelHeight, kernelWidth
        );
    }

    // ---- Layer 6: Fully‐Connected (120 → 10) ----
    {
        const int inputLength  = 120 * 1 * 1; // 120*1*1 = 120
        const int outputLength = 10;                                     // 10
        const int blockSize    = 256;                                  // 256
        const int gridSize     = (outputLength + blockSize - 1) / blockSize;

        dot_product_forward_kernel<<<gridSize,blockSize>>>(
            (const double*)features->layer5,   // flat vector of length 120
            (      double*)features->output,   // flat vector of length 10
            (const double*)lenet->weight5_6,   // [120][10]
            (const double*)lenet->bias5_6,     // [10]
            inputLength,
            outputLength
        );
    }

    // Wait for GPU to finish before returning
    cudaDeviceSynchronize();
}

