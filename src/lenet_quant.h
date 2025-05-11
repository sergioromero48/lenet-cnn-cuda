/*
@author : 范文捷
@data    : 2016-04-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api	:

批量训练
void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);

训练
void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT],uint8 label);

预测
uint8 Predict(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 count);

初始化
void Initial(LeNet5 *lenet);
*/
#include <stdint.h>

#pragma once

#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

#define QUANT_SCALE 127.0

typedef unsigned char uint8;
typedef uint8 image[28][28];

/*
typedef struct LeNet5
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

}LeNet5;
*/
typedef struct LeNet5_q
{
	int8_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	int8_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	int8_t weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	int8_t bias0_1[LAYER1];
	int8_t bias2_3[LAYER3];
	int8_t bias4_5[LAYER5];
	int8_t bias5_6[OUTPUT];
}LeNet5_q;

/*
typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];
}Feature;
*/

typedef struct Feature_q {
    // normalization still produces doubles:
    int8_t input[ INPUT ][ LENGTH_FEATURE0 ][ LENGTH_FEATURE0 ];
    // every downstream layer & output is *really* int8:
    int8_t layer1[ LAYER1 ][ LENGTH_FEATURE1 ][ LENGTH_FEATURE1 ];
    int8_t layer2[ LAYER2 ][ LENGTH_FEATURE2 ][ LENGTH_FEATURE2 ];
    int8_t layer3[ LAYER3 ][ LENGTH_FEATURE3 ][ LENGTH_FEATURE3 ];
    int8_t layer4[ LAYER4 ][ LENGTH_FEATURE4 ][ LENGTH_FEATURE4 ];
    int8_t layer5[ LAYER5 ][ LENGTH_FEATURE5 ][ LENGTH_FEATURE5 ];
    int8_t output[ OUTPUT ];
} Feature_q;


#ifdef __cplusplus
extern "C" {
#endif

uint8 Predict_q (LeNet5_q *lenet, image input, uint8 count);   /* NEW */
void  Initial_q (LeNet5_q *lenet); 

uint8 get_result_q(const Feature_q *f, uint8 classes);
void forward_q(LeNet5_q *lenet, Feature_q *features, float(*action)(float));

#ifdef __cplusplus
}
#endif