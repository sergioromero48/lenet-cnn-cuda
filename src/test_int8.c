#include "lenet_quant.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define FILE_TEST_IMAGE			"data/t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL			"data/t10k-labels-idx1-ubyte"
#undef LENET_FILE
#define LENET_FILE 				"model/model_int8.dat"
#define COUNT_TEST				10000


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

int load_q(LeNet5_q *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5_q), 1, fp);
    fclose(fp);
    return 0;
}

int testing_q(LeNet5_q *lenet, image *test_data, uint8 *test_label, int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i) {
        uint8 l = test_label[i];
        int p = Predict_q(lenet, test_data[i], 10);
        right += (l == p);
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

void foo()
{
    /* allocate only test dataset */
    image *test_data = calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = calloc(COUNT_TEST, sizeof(uint8));
    if (!test_data || !test_label) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    /* read only test data */
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        fprintf(stderr, "Error: Could not read test dataset files.\n");
        exit(1);
    }
    /* load quantized INT8 model */
    LeNet5_q *lenet = malloc(sizeof(LeNet5_q));
    if (!lenet || load_q(lenet, LENET_FILE)) {
        fprintf(stderr, "Error: Could not load model from %s\n", LENET_FILE);
        exit(1);
    }
    /* run inference and measure time */
    clock_t start = clock();
    int correct = testing_q(lenet, test_data, test_label, COUNT_TEST);
    clock_t end = clock();
    /* report results */
    printf("Accuracy: %d/%d (%.2f%%)\n", correct, COUNT_TEST, correct * 100.0 / COUNT_TEST);
    printf("Time: %.3f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    /* clean up */
    free(lenet);
    free(test_data);
    free(test_label);
}

int main()
{
	foo();
	return 0;
}