/* main_lenet_lea.c  –  minimal MSP430/LEA inference harness
 *
 * ★ Build  (FR5994 example):
 *   msp430-gcc -mmcu=msp430fr5994 -I. -Os \
 *       main_lenet_lea.c lenet_quant.c \
 *       -ldsplib_fr5xx_3.40.00.00 \
 *       -o lenet_demo.elf
 *   msp430-objcopy -O ihex lenet_demo.elf lenet_demo.hex
 *   mspdebug rf2500 "prog lenet_demo.hex"
 */
#include <msp430.h>
#include "DSPLib.h"

#include "lenet_quant.h"
#include "model.h"          /* weight0_1 … weight5_6 arrays */

/*--------------- LEA / DSPLib housekeeping ----------------*/
static void initLEA(void)
{
    /* 1.  Point LEA stack to top of 4‑KB leaRAM (0x3C00) */
    extern uint16_t __LEA_MSP430_BASE;
    LEACNF2  = (uint16_t)((uintptr_t)&__LEA_MSP430_BASE >> 2);
    /* 2.  Enable the engine */
    LEAPMCTL |= LEACMDEN;
}

/*--------------- Network instance & helpers ---------------*/
#pragma PERSISTENT(net)          /* keep in FRAM */
LeNet5_q net;

/* Copy the flat INT‑8 weight blobs from model.h into the
   struct layout that lenet_quant.c expects */
static void populateNetwork(void)
{
    /* layer 0‑1 */
    memcpy(net.weight0_1, weight0_1, sizeof(net.weight0_1));
    memcpy(net.bias0_1,   bias0_1,   sizeof(net.bias0_1));

    memcpy(net.weight2_3, weight2_3, sizeof(net.weight2_3));
    memcpy(net.bias2_3,   bias2_3,   sizeof(net.bias2_3));

    memcpy(net.weight4_5, weight4_5, sizeof(net.weight4_5));
    memcpy(net.bias4_5,   bias4_5,   sizeof(net.bias4_5));

    memcpy(net.weight5_6, weight5_6, sizeof(net.weight5_6));
    memcpy(net.bias5_6,   bias5_6,   sizeof(net.bias5_6));
}

/*--------------- Placeholder input image ------------------*/
#pragma PERSISTENT(dummy)
image dummy = { 0 };             /* all‑zero 28×28 */

/*--------------- main -------------------------------------*/
volatile uint8_t resultClass;     /* debugger watch point */

int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;    /* stop watchdog          */
#ifdef __MSP430FR5XX_6XX_FAMILY__
    PM5CTL0 &= ~LOCKLPM5;        /* unlock GPIO if FRAM    */
#endif

    initLEA();                   /* turn on accelerator    */
    populateNetwork();           /* load weights/biases    */

    /* --- single inference --- */
    resultClass = Predict_q(&net, dummy, 10);

    /* halt here – inspect resultClass or add your own output */
    __no_operation();
    while (1) ;                  /* LPM later if you like  */
}
