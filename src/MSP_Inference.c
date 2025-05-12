/* main.c – minimal LeNet-5 inference on MSP430FR5994 using DMA, FRAM2 */

#include <msp430.h>
#include <stdint.h>
#include <stdio.h>

#include "lenet_quant.h"
#include "model.h"
#include "image_input.h"

/* DMA helper for byte-wise copy using channel 0, 20-bit addresses */
static inline void dma_memcpy20(const void *src20, void *dst20, uint16_t len) {
    DMACTL4 = DMARMWDIS; /* allow FRAM DMA */
    DMA0CTL &= ~DMAEN;
    __data20_write_long((uintptr_t)&DMA0SA, (uintptr_t)src20);
    __data20_write_long((uintptr_t)&DMA0DA, (uintptr_t)dst20);
    DMA0SZ = len;
    DMA0CTL = DMADT_0 | DMASRCINCR_3 | DMADSTINCR_3 | DMASRCBYTE | DMADSTBYTE | DMAEN;
    while (DMA0CTL & DMAEN);
}

/* Place the network struct into FRAM2 (.fr2) */
#pragma DATA_SECTION(net, ".fr2")
static LeNet5_q net;

/* Weight loader using DMA */
static void load_weights(void) {
    const int8_t *cur;
    uint32_t n;

    /* conv1 */
    cur = weight0_1;
    n = INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL;
    dma_memcpy20(cur, net.weight0_1, n);
    cur += n;

    /* conv2 */
    cur = weight2_3;
    n = LAYER2 * LAYER3 * LENGTH_KERNEL * LENGTH_KERNEL;
    dma_memcpy20(cur, net.weight2_3, n);
    cur += n;

    /* conv3 */
    cur = weight4_5;
    n = LAYER4 * LAYER5 * LENGTH_KERNEL * LENGTH_KERNEL;
    dma_memcpy20(cur, net.weight4_5, n);
    cur += n;

    /* fc */
    cur = weight5_6;
    n = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5 * OUTPUT;
    dma_memcpy20(cur, net.weight5_6, n);
    cur += n;

    /* biases */
    dma_memcpy20(bias0_1, net.bias0_1, LAYER1);
    dma_memcpy20(bias2_3, net.bias2_3, LAYER3);
    dma_memcpy20(bias4_5, net.bias4_5, LAYER5);
    dma_memcpy20(bias5_6, net.bias5_6, OUTPUT);
}

/* ReLU activation */
static float relu(float x) { return x > 0 ? x : 0; }

int main(void) {
    WDTCTL = WDTPW | WDTHOLD; /* stop watchdog */
    load_weights();           /* DMA copies (~5ms) */

    Feature_q feat = {0};
    forward_q(&net, &feat, relu);
    printf("Predicted class = %d\r\n", (int)get_result_q(&feat, OUTPUT));

    while (1) __no_operation();
}

/* HAL callbacks stubs */
void ButtonCallback_SW1(uint8_t s) { (void)s; }
void ButtonCallback_SW2(uint8_t s) { (void)s; }
void TimerCallback(void) { }
