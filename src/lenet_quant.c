#include "lenet_quant.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(int8_t))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define SUBSAMP_MAX_FORWARD_q(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																									\
}

// 8bit quant inference
#define CONVOLUTE_VALID_q(input,output,weight_array)                                        \
{                                                                                       \
    FOREACH(o0, GETLENGTH(output))                                                      \
        FOREACH(o1, GETLENGTH(*(output)))                                               \
            FOREACH(w0, GETLENGTH(weight_array))                                        \
                FOREACH(w1, GETLENGTH(*(weight_array)))                                \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] *                     \
                                     dequantize((weight_array)[w0][w1]);      \
}

// Quantized convolution: use quant variant
#define CONVOLUTION_FORWARD_q(IN, OUT, WQ, BQ, ACT)                             \
{                                                                               \
    const int IC = GETLENGTH(WQ);                                               \
    const int OC = GETLENGTH(*WQ);                                              \
    const int OH = GETLENGTH(*(OUT));                                           \
    const int OW = GETLENGTH(**(OUT));                                          \
                                                                                \
    for (int oc = 0; oc < OC; ++oc)                                             \
        for (int y = 0; y < OH; ++y)                                            \
            for (int x = 0; x < OW; ++x) {                                      \
                float acc = 0.f;                                                \
                for (int ic = 0; ic < IC; ++ic)                                 \
                    for (int r = 0; r < LENGTH_KERNEL; ++r)                     \
                        for (int c = 0; c < LENGTH_KERNEL; ++c)                 \
                            acc +=                                              \
                                dequantize((IN)[ic][y + r][x + c]) *            \
                                dequantize((WQ)[ic][oc][r][c]);                 \
                float v = ACT(acc + dequantize((BQ)[oc]));                      \
                (OUT)[oc][y][x] = quantize(v);                                  \
            }                                                                   \
}



#define DOT_PRODUCT_FORWARD_q(q_input, q_output, Wq, bq, act)               \
{                                                                            \
    /* 1) clear a tiny FP accumulator */                                     \
    double _acc[ GETLENGTH(*(Wq)) ] = {0.0};                                  \
                                                                             \
    /* 2) MAC in FP: dequantize both input and weight */                     \
    for (int _x = 0; _x < GETLENGTH(Wq); ++_x) {                              \
        double a = dequantize(((int8_t*)(q_input))[_x]);                     \
        for (int _y = 0; _y < GETLENGTH(*(Wq)); ++_y)                         \
            _acc[_y] += a * dequantize((Wq)[_x][_y]);                        \
    }                                                                        \
                                                                             \
    /* 3) add bias, apply activation, then quantize once per output */       \
    FOREACH(_j, GETLENGTH(bq)) {                                             \
        double v = act(_acc[_j] + dequantize((bq)[_j]));                     \
        ((int8_t*)(q_output))[_j] = quantize(v);                             \
    }                                                                        \
}                                                                     \

/// Saturating quantize into signed‐8 range, then cast
int8_t quantize(float x) {
	// round to nearest
	int q = (int)lrintf(x * QUANT_SCALE);
	// clamp to [-127, 127]
	if (q >  127) q =  127;
	if (q < -127) q = -127;
	return (int8_t)q;
}

/// Simple dequantize: recover float exactly, no extra clamping
float dequantize(int8_t x) {
	return ((float)x) / QUANT_SCALE;
}

float relu(float x)
{
	return x*(x > 0);
}

void forward_q(LeNet5_q *lenet, Feature_q *features, float(*action)(float))
{                                                                                     
   CONVOLUTION_FORWARD_q(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
   SUBSAMP_MAX_FORWARD_q(features->layer1, features->layer2);
   CONVOLUTION_FORWARD_q(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
   SUBSAMP_MAX_FORWARD_q(features->layer3, features->layer4);
   CONVOLUTION_FORWARD_q(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
   DOT_PRODUCT_FORWARD_q(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

/* lenet_quant.c ------------------------------------------ */
static void load_input_q(Feature_q *f, image img)
{
    const int SZ = 28*28;
    double mean = 0, var = 0;

    for (int j = 0; j < 28; ++j)
        for (int k = 0; k < 28; ++k) {
            mean += img[j][k];
            var  += img[j][k] * img[j][k];
        }
    mean /= SZ;
    var   = sqrt(var / SZ - mean*mean);

    memset(f->input, 0, sizeof(f->input));

    for (int j = 0; j < 28; ++j)
        for (int k = 0; k < 28; ++k) {
            float n = (float)((img[j][k] - mean) / var);
            f->input[0][j + PADDING][k + PADDING] = quantize(n);
        }
}


uint8 get_result_q(const Feature_q *f, uint8 classes)
{
    const int8_t *out = (const int8_t *)f->output;   // logits in int8
    uint8  best  = 0;
    int8_t vmax  = out[0];

    for (uint8 i = 1; i < classes; ++i)
    {
        if (out[i] > vmax) {
            vmax = out[i];
            best = i;
        }
    }
    return best;   // arg‑max index
}

// only quantized predict in this file
uint8 Predict_q(LeNet5_q *net, image img, uint8 count)
{
    Feature_q f = {0};
    load_input_q(&f, img);
    forward_q(net, &f, relu);
    return get_result_q(&f, count);
}


