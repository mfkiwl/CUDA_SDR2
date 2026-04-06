/*
 * CDMA Code Generation and Processing Implementation
 * 
 * This module provides functions for generating GPS L1 C/A Gold codes
 * using G1/G2 linear feedback shift registers (LFSR), sampling codes
 * for FFT-based processing, and frequency-domain code preparation
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include <fftw3.h>
#include <complex.h>

#include "core/params.h"
#include "core/types.h"
#include "dsp/code.h"


// G2 delay values for PRN 1-37 (chips)
static const uint16_t L1CA_G2_delay[] = {
       5,   6,   7,   8,  17,  18, 139, 140, 141, 251, 252, 254, 255, 256, 257,
     258, 469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
     861, 862, 863, 950, 947, 948, 950
};


/*
 * LFSR Register Layout (10 bits):
 * ==========================================================
 * new -> || 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 || -> out
 * ==========================================================
 * Bit Indexing:
 * ==========================================================
 * new -> || 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |  0 || -> out
 * ==========================================================
 */

 /*
 * Performs one step of G1 linear feedback shift register
 */
static inline uint16_t lfsr_step_g1(uint16_t state) {
    // G1 polynomial: x^10 + x^3 + 1
    // Taps at bits 0 and 7
    uint16_t bit = (state ^ (state >> 7)) & 1;
    return (state >> 1) | (bit << 9);
}

/*
 * Performs one step of G2 linear feedback shift register
 */
static inline uint16_t lfsr_step_g2(uint16_t state) {
    // G2 polynomial: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
    // Taps at bits: 0, 1, 2, 4, 7, 8
    uint16_t bit = (state ^ (state >> 1) ^ (state >> 2) ^ (state >> 4) ^ (state >> 7) ^ (state >> 8)) & 1;
    return (state >> 1) | (bit << 9);
}



void gen_code_L1CA(int prn, int8_t *buffer, int n_samples, int start_chip) {

    if (prn < 1 || prn > 37 || buffer == NULL) return;

    // Initialize registers to all-ones state
    uint16_t g1 = 0x3FF;
    uint16_t g2 = 0x3FF;

    // Apply G2 delay for specific PRN
    uint16_t shift = GPS_CODE_LEN - L1CA_G2_delay[prn - 1];
    for (int i = 0; i < shift; i++) {
        g2 = lfsr_step_g2(g2);
    }

    // Skip to starting chip position
    int total_skip = (start_chip % GPS_CODE_LEN);
    for (int i = 0; i < total_skip; i++) {
        g1 = lfsr_step_g1(g1);
        g2 = lfsr_step_g2(g2);
    }

    // Generate PRN sequence
    for (int i = 0; i < n_samples; i++) {

        // Extract output bits (LSB) from each register
        int bit1 = g1 & 1;
        int bit2 = g2 & 1;

        // Convert to polar NRZ: XOR becomes multiplication
        buffer[i] = (bit1 == bit2) ? 1 : -1;

        // Advance both registers
        g1 = lfsr_step_g1(g1);
        g2 = lfsr_step_g2(g2);
    }
}



void sample_code(
    const int8_t *code,
    float complex *code_sampled,
    int fft_size,
    const receiver_t *recv,
    double coff
) {

    double chip_step = FS_GPS / recv->f_adc;

    int N = (int)round(recv->f_adc * (CODE_PERIOD_MS / 1000.0));

    double curr_chip = coff;

    // Fill buffer with sampled code values
    int i = 0;
    for (; i < N; i++) {
        // Write to I-channel, Q = 0
        code_sampled[i] = (float) code[(int) curr_chip] + I*0.0f;

        curr_chip += chip_step;
        if (curr_chip >= GPS_CODE_LEN) curr_chip -= GPS_CODE_LEN;
    }
    for (; i < fft_size; i++) {
        // Zero padding to FFT size
        code_sampled[i] = 0.0f + I*0.0f;
    }
}

void gen_code_fft(
    fftwf_complex *code_sampled,
    fftwf_complex *code_fft,
    int fft_size
) {

    fftwf_plan plan = fftwf_plan_dft_1d(fft_size, code_sampled, code_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    if (!plan) {
        fprintf(stderr, "FFT Plan creation failed in gen_code_fft\n");
        return;
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Complex conjugate of spectrum C*(f) for correlation
    for (int i = 0; i < fft_size; i++) {
        code_fft[i][1] = -code_fft[i][1];
    }
}
