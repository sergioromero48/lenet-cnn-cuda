/**********************************************************************
 * lenet_cuda.cu  –  GPU implementation of the entire LeNet‑5 forward
 *                  path (inference only).
 *
 * Public symbol exposed in lenet_cuda.h:
 *     uint8 Predict_CUDA(const LeNet5*, image, uint8);
 *
 * Only dependency:  lenet.h  (structures + constants).
 *
 * REQUIRED EDIT IN lenet.c:
 *     remove the keyword 'static' from  load_input()
 *********************************************************************/
#include <cuda_runtime.h>
#include "lenet_cuda.h"

/* ───────────── helper: device ReLU ───────────── */
__device__ inline double d_relu(double x){ return x>0?x:0; }

/* ───────────── CONV‑5×5 valid kernel ───────────
 * generic over C_in and C_out, but kernel size fixed to 5
 */
__global__
void conv5_valid(const double* __restrict__ in,   // (C_in,H,W)
                 const double* __restrict__ w,    // (C_out,C_in,5,5)
                 const double* __restrict__ bias, // (C_out)
                 double*       __restrict__ out,  // (C_out,H-4,W-4)
                 int C_in, int H, int W, int C_out)
{
    int c_out = blockIdx.z;                             // which output map
    int oy    = blockIdx.y * blockDim.y + threadIdx.y;
    int ox    = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - 4,  W_out = W - 4;
    if (oy >= H_out || ox >= W_out) return;

    double sum = bias[c_out];
    const double* w_base = w + c_out * C_in * 25;

    for (int c = 0; c < C_in; ++c){
        const double* w_ck = w_base + c * 25;
        const double* in_c = in     + c * H * W;
#pragma unroll
        for (int ky = 0; ky < 5; ++ky)
#pragma unroll
            for (int kx = 0; kx < 5; ++kx)
                sum += in_c[(oy + ky) * W + (ox + kx)] *
                       w_ck[ky * 5 + kx];
    }
    out[c_out * H_out * W_out + oy * W_out + ox] = d_relu(sum);
}

/* ───────────── 2×2 max‑pool, stride 2 ─────────── */
__global__
void maxpool2x2(const double* __restrict__ in,    // (C,H,W)
                double*       __restrict__ out,   // (C,H/2,W/2)
                int C, int H, int W)
{
    int c  = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H >> 1, W_out = W >> 1;
    if (oy >= H_out || ox >= W_out) return;

    const double* in_c = in + c * H * W;
    double a = in_c[(2*oy  ) * W + 2*ox    ];
    double b = in_c[(2*oy  ) * W + 2*ox + 1];
    double d = in_c[(2*oy+1) * W + 2*ox    ];
    double e = in_c[(2*oy+1) * W + 2*ox + 1];
    double m = a; if (b>m) m=b; if (d>m) m=d; if (e>m) m=e;
    out[c * H_out * W_out + oy * W_out + ox] = m;
}

/* ───────────── fully‑connected 120 → 10 ───────── */
__global__
void fc120x10(const double* __restrict__ in120,
              const double* __restrict__ w120x10,
              const double* __restrict__ bias10,
              double*       __restrict__ out10)
{
    int o = threadIdx.x;  // we will launch 10 threads
    if (o >= 10) return;
    double sum = bias10[o];
#pragma unroll
    for (int i = 0; i < 120; ++i)
        sum += in120[i] * w120x10[i * 10 + o];
    out10[o] = d_relu(sum);
}

/* helper to upload a host tensor to device (returns d_ptr) */
template<typename T>
static T* h2d(const T* h, size_t bytes){
    T* d; cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    return d;
}

/* --------------------- forward_cuda --------------------- */
void forward_cuda(const LeNet5* net, Feature* feat)
{
    /* upload weights & biases */
    double* d_w01 = h2d(net->weight0_1, sizeof(net->weight0_1));
    double* d_b01 = h2d(net->bias0_1  , sizeof(net->bias0_1 ));
    double* d_w23 = h2d(net->weight2_3, sizeof(net->weight2_3));
    double* d_b23 = h2d(net->bias2_3  , sizeof(net->bias2_3 ));
    double* d_w45 = h2d(net->weight4_5, sizeof(net->weight4_5));
    double* d_b45 = h2d(net->bias4_5  , sizeof(net->bias4_5 ));
    double* d_w56 = h2d(net->weight5_6, sizeof(net->weight5_6));
    double* d_b56 = h2d(net->bias5_6  , sizeof(net->bias5_6 ));

    /* stage buffers */
    double *d_in, *d_c1, *d_p1, *d_c2, *d_p2, *d_c3, *d_out;
    d_in = h2d(feat->input, sizeof(feat->input));           // (1,32,32)
    cudaMalloc(&d_c1, 6  * 28 * 28 * sizeof(double));
    cudaMalloc(&d_p1, 6  * 14 * 14 * sizeof(double));
    cudaMalloc(&d_c2, 16 * 10 * 10 * sizeof(double));
    cudaMalloc(&d_p2, 16 *  5 *  5 * sizeof(double));
    cudaMalloc(&d_c3, 120*  1 *  1 * sizeof(double));
    cudaMalloc(&d_out,10  * sizeof(double));

    dim3 blk(16,16);

    /* conv1  (1→6, 32→28) */
    dim3 grd1((28+15)/16, (28+15)/16, 6);
    conv5_valid<<<grd1,blk>>>(d_in,d_w01,d_b01,d_c1, 1,32,32,6);

    /* pool1 (6,28→14) */
    grd1 = dim3((14+15)/16, (14+15)/16, 6);
    maxpool2x2<<<grd1,blk>>>(d_c1,d_p1,6,28,28);

    /* conv2 (6→16, 14→10) */
    dim3 grd2((10+15)/16,(10+15)/16,16);
    conv5_valid<<<grd2,blk>>>(d_p1,d_w23,d_b23,d_c2, 6,14,14,16);

    /* pool2 (16,10→5) */
    grd2 = dim3((5+15)/16,(5+15)/16,16);
    maxpool2x2<<<grd2,blk>>>(d_c2,d_p2,16,10,10);

    /* conv3 (16→120, 5→1) */
    dim3 grd3(1,1,120);
    conv5_valid<<<grd3,blk>>>(d_p2,d_w45,d_b45,d_c3, 16,5,5,120);

    /* FC 120→10  (launch exactly 1 block, 10 threads) */
    fc120x10<<<1,10>>>(d_c3,d_w56,d_b56,d_out);

    /* copy final output → feat.output */
    cudaMemcpy(feat->output,d_out,10*sizeof(double),cudaMemcpyDeviceToHost);

    /* optional: copy layer5 if you need it */
    cudaMemcpy(feat->layer5,d_c3,120*sizeof(double),cudaMemcpyDeviceToHost);

    /* free GPU buffers */
    cudaFree(d_in); cudaFree(d_c1); cudaFree(d_p1);
    cudaFree(d_c2); cudaFree(d_p2); cudaFree(d_c3); cudaFree(d_out);
    cudaFree(d_w01); cudaFree(d_b01);
    cudaFree(d_w23); cudaFree(d_b23);
    cudaFree(d_w45); cudaFree(d_b45);
    cudaFree(d_w56); cudaFree(d_b56);
}

/* --------------------- Predict_CUDA -------------------- */
extern void load_input(Feature*, image);   /* make non‑static in lenet.c */

uint8 Predict_CUDA(const LeNet5* net, image img, uint8 count)
{
    Feature feat={0};
    load_input(&feat,img);
    forward_cuda(net,&feat);

    /* arg‑max */
    double* o=feat.output;
    uint8 idx=0; double mx=o[0];
    for(uint8 i=1;i<count;++i) if(o[i]>mx){mx=o[i];idx=i;}
    return idx;
}
