#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PADDING
#define PADDING (LENGTH_KERNEL/2)
#endif

// forward_qa prototype (in case header declaration was missed)
void forward_qa(const LeNet5 *lenet, Feature *features, double (*action)(double));

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}
// Added printing, quantization, and dequantization here
#define CONVOLUTION_FORWARD_QA(input, output, weight, bias, action, scale_w, scale_b) \
{ \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            FOREACH(i0, GETLENGTH(*(input))) \
                FOREACH(i1, GETLENGTH(**(input))) \
                    FOREACH(w0, GETLENGTH(weight[x][y])) \
                        FOREACH(w1, GETLENGTH(*(weight[x][y]))) { \
                            int8_t q_weight = quantize(weight[x][y][w0][w1]); \
                            double dq_weight = dequantize(q_weight); \
                            (output[y][i0][i1]) += (input[x][i0 + w0][i1 + w1]) * dq_weight; \
                        } \
    FOREACH(j, GETLENGTH(output)) \
        FOREACH(i, GETCOUNT(output[j])) { \
            int8_t q_bias = quantize(bias[j]); \
            double dq_bias = dequantize(q_bias); \
            ((double *)output[j])[i] = action(((double *)output[j])[i] + dq_bias); \
        } \
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
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
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}
// Added printing, quantization, and dequantization here
#define DOT_PRODUCT_FORWARD_QA(input,output,weight,bias,action, input_scale, weight_scale, bias_scale) \
{                                                                                                             \
	    int  input_len  = GETLENGTH(weight);                                                                      \
	    int  output_len = GETLENGTH(*weight);                                                                     \
	    double *act     = (double *)input;    /* your 120 floats */                                               \
	    int8_t *w       = (int8_t *)weight;   /* quantized weights */                                             \
	    int8_t *b       = (int8_t *)bias;     /* quantized biases */                                              \
	                                                                                                             \
	    /* zero out outputs */                                                                                   \
	    for (int y = 0; y < output_len; ++y)                                                                      \
	        ((double *)output)[y] = 0.0;                                                                          \
	                                                                                                             \
	    /* accumulate in_fp * dequantized_weight */                                                              \
	    for (int x = 0; x < input_len; ++x) {                                                                     \
	        double in_val = act[x];                                                                              \
	        for (int y = 0; y < output_len; ++y) {                                                                \
	            double w_val = dequantize(w[x * output_len + y]);                                   \
	            ((double *)output)[y] += in_val * w_val;                                                          \
	        }                                                                                                    \
	    }                                                                                                        \
	                                                                                                             \
	    /* add dequantized biases + activation */                                                                 \
	    for (int j = 0; j < output_len; ++j) {                                                                    \
	        double b_val = dequantize(b[j]);                                                          \
	        ((double *)output)[j] = action(((double *)output)[j] + b_val);                                        \
	    }                                                                                                        \
	}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

float input_scale = 127;

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

// Quantize floating to int8
int8_t quantize(double value) {
	
	return (int8_t)(value * QUANT_SCALE);
}

// Dequantize int8 back to float
double dequantize(int8_t qval) {
	double norm = (qval / QUANT_SCALE);
	if (norm > 1.0) {
		norm = 1.0;
	} else if (norm < -1.0) {
		norm = -1.0;
	}
    return norm;
}

void print_original(float original_input){
	printf("%11f\t", original_input);
}

void print_quantized(int8_t quant_input){
	printf("%4d\t", quant_input);
}

void print_dequantized(float dequant_input){
	printf("%10f\n", dequant_input);
}

static int abs_cmp(const void *a, const void *b)
{
    return (*(int*)a) - (*(int*)b);
}

/* prune `rate` fraction (0–1) of the smallest‑magnitude weights */
void prune_weights(LeNet5 *net, double rate)
{
    /* weights occupy the contiguous block [weight0_1 .. bias0_1)   :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1} */
    int8_t *w_begin = (int8_t*)net->weight0_1;
    int8_t *w_end   = (int8_t*)net->bias0_1;
    size_t  N       = (size_t)(w_end - w_begin);

    /* 1) copy |w| into temp buffer, find percentile threshold */
    int *mag = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; ++i) mag[i] = abs((int)w_begin[i]);

    qsort(mag, N, sizeof(int), abs_cmp);
    int kth = mag[(size_t)(rate * N)];   /* 25‑th percentile */
    free(mag);

    /* 2) zero out any weight whose magnitude ≤ threshold */
    for (size_t i = 0; i < N; ++i)
        if (abs((int)w_begin[i]) <= kth) w_begin[i] = 0;
}

static void forward_fp(LeNet5FP *lenet, Feature *features, double(*action)(double))
{
	// scale for quantization
    double scale_w = 1;
    double scale_b = 1; 
	// calling functions with scale added in
    CONVOLUTION_FORWARD_QA(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action, scale_w, scale_b);
    SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
    CONVOLUTION_FORWARD_QA(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action, scale_w, scale_b);
    SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
    CONVOLUTION_FORWARD_QA(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action, scale_w, scale_b);
    DOT_PRODUCT_FORWARD_QA(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action, input_scale, scale_w, scale_b);
}


static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}

/* ------------------------------------------------------------------ */
/*  Quant‑aware helpers – one global symmetric scale is enough to     */
/*  prove the point.  Tune per‑layer later if you like.               */

static inline int8_t q_s8(double x, double s)
{
    int v = (int)lrint(x / s);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}
static inline double dq_s8(int8_t q, double s) { return q * s; }

/* Copy int8 → float ------------------------------------------------- */
void int8_to_fp(const LeNet5 *src, LeNet5FP *dst, double s)
{
    const int8_t *p  = (const int8_t *)src;
    double       *pd = (double       *)dst;
    size_t N = sizeof(LeNet5);
    for (size_t i = 0; i < N; ++i)               /* 470 kB → 3.8 MB   */
        pd[i] = dq_s8(p[i], s);
}

/* Copy float → int8 ------------------------------------------------- */
void fp_to_int8(const LeNet5FP *src, LeNet5 *dst, double s)
{
    const double *ps = (const double *)src;
    int8_t       *pd = (int8_t       *)dst;
    size_t N = sizeof(LeNet5);
    for (size_t i = 0; i < N; ++i)
        pd[i] = q_s8(ps[i], s);
}

/* ------------------------------------------------------------------ */
/*  Quant‑aware TrainBatch wrapper                                    */
/* ------------------------------------------------------------------ */
void TrainBatch_QAT(LeNet5 *qnet,
                    image  *inputs,
                    uint8  *labels,
                    int     B,
                    double  scale)
{
    static LeNet5FP fp;                 /* re‑used every call          */

    /* 1) de‑quantise once */
    int8_to_fp(qnet, &fp, scale);

    /* 2) run the original float trainer (rename yours to TrainBatch_FP) */
    /*    ↓↓↓ Paste the body of your previous TrainBatch here ↓↓↓        */

    /* ----------------------------------------------------------------
       Begin:  EXACT copy of your old TrainBatch, but operating on &fp
       ----------------------------------------------------------------*/
    double buffer[GETCOUNT(LeNet5)] = {0};
#pragma omp parallel for
    for (int i = 0; i < B; ++i)
    {
        Feature features = {0};
        Feature errors   = {0};
        LeNet5   deltas  = {0};

        load_input(&features, inputs[i]);
        forward_qa((LeNet5 *)&fp, &features, relu);
        load_target(&features, &errors, labels[i]);
        backward((LeNet5FP *)&fp, (LeNet5 *)&deltas, &errors, &features, relugrad);

#pragma omp critical
        {
            FOREACH(j, GETCOUNT(LeNet5))
                buffer[j] += ((double *)&deltas)[j];
        }
    }
    double k = ALPHA / B;
    FOREACH(i, GETCOUNT(LeNet5))
        ((double *)&fp)[i] += k * buffer[i];
    /* ----------------------------------------------------------------
       End   (unchanged optimiser, still double precision)             */

    /* 3) re‑quantise back to int8 */
    fp_to_int8(&fp, qnet, scale);
}

// Quant-aware forward (uses int8 weights/biases + activation)
void forward_qa(const LeNet5 *lenet, Feature *features, double (*action)(double))
{
    double scale_w = QUANT_SCALE, scale_b = QUANT_SCALE;
    double input_scale = QUANT_SCALE;
    CONVOLUTION_FORWARD_QA(features->input, features->layer1,
                            lenet->weight0_1, lenet->bias0_1,
                            action, scale_w, scale_b);
    SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
    CONVOLUTION_FORWARD_QA(features->layer2, features->layer3,
                            lenet->weight2_3, lenet->bias2_3,
                            action, scale_w, scale_b);
    SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
    CONVOLUTION_FORWARD_QA(features->layer4, features->layer5,
                            lenet->weight4_5, lenet->bias4_5,
                            action, scale_w, scale_b);
    DOT_PRODUCT_FORWARD_QA(features->layer5, features->output,
                           lenet->weight5_6, lenet->bias5_6,
                           action, input_scale, scale_w, scale_b);
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward_qa(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward_qa(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}
/* lenet.c ----------------------------------------------------------- */
static inline double  dq(int8_t q, double s)         { return q * s;                  }
static inline int8_t  q (double x, double s)
{
    int v = (int)lrint(x / s);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward_qa(lenet, &features, relu);
	return get_result(&features, count);
}

#define QSCALE (1.0/64)

void Initial(LeNet5 *qnet)
{
    LeNet5FP fp;        /* float‑precision workspace */
    double *pos;

    /* 1) exactly your original Xavier/He loops, but writing into fp */
    for (pos = (double*)fp.weight0_1;
         pos < (double*)fp.bias0_1;
         *pos++ = f64rand());

    for (pos = (double*)fp.weight0_1;
         pos < (double*)fp.weight2_3;
         *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));

    for (pos = (double*)fp.weight2_3;
         pos < (double*)fp.weight4_5;
         *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));

    for (pos = (double*)fp.weight4_5;
         pos < (double*)fp.weight5_6;
         *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));

    for (pos = (double*)fp.weight5_6;
         pos < (double*)fp.bias0_1;
         *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));

    /* zero out all biases (fp.bias0_1 → end‑of‑struct) */
    for (pos = (double*)fp.bias0_1;
         pos < (double*)(&fp + 1);
         *pos++ = 0.0);

    /* 2) quantise entire fp workspace back into your int8 model */
    fp_to_int8(&fp, qnet, QSCALE);
}

/*  lenet.c  — QAT utilities + fixed TrainBatch_QAT  */
