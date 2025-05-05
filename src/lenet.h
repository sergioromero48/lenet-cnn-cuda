#include <stdint.h>
#include <stdio.h> 

#pragma once

#define LENGTH_KERNEL 5
#define LENGTH_FEATURE0 32
#define LENGTH_FEATURE1 (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2 (LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3 (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE4 (LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5 (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT  1
#define LAYER1 6
#define LAYER2 6
#define LAYER3 16
#define LAYER4 16
#define LAYER5 120
#define OUTPUT 10

#define ALPHA  0.5
#define QSCALE (1.0/64)
#define PADDING (LENGTH_KERNEL/2)

typedef unsigned char uint8;
typedef uint8 image[28][28];

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* FLOAT WORKSPACE for QAT – exact same shapes but float32 (double)   */
typedef struct LeNet5 {
    int8_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight5_6[LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5][OUTPUT];

    int8_t bias0_1[LAYER1];
    int8_t bias2_3[LAYER3];
    int8_t bias4_5[LAYER5];
    int8_t bias5_6[OUTPUT];
} LeNet5;

/* ---------------- float workspace (one per batch) ---------------- */
typedef struct LeNet5FP {
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];
} LeNet5FP;

/* feature maps and activations */
typedef struct Feature {
    double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    double output[OUTPUT];
} Feature;

/* ---------------- helper prototypes ------------------------------ */
void int8_to_fp (const LeNet5  *src, LeNet5FP *dst, double s);
void fp_to_int8 (const LeNet5FP *src, LeNet5  *dst, double s);
void TrainBatch_QAT(LeNet5 *qnet, image *inputs, uint8 *labels,
                    int batch_size, double scale);

void Initial(LeNet5 *net);
void prune_weights(LeNet5 *net, double rate);

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);

void Train(LeNet5 *lenet, image input, uint8 label);

uint8 Predict(LeNet5 *lenet, image input, uint8 count);

void Initial(LeNet5 *lenet);

void forward_qa(const LeNet5 *lenet, Feature *features, double (*action)(double));

#ifdef __cplusplus
}
#endif