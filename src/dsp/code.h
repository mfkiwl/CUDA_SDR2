/*
 * CDMA Code Generation and Processing
 * 
 * This header provides functions for generating GPS L1 C/A Gold codes
 * using G1/G2 linear feedback shift registers (LFSR), sampling codes
 * for FFT-based processing, and frequency-domain code preparation
 */

#ifndef DSP_CODE_H
#define DSP_CODE_H

#include <fftw3.h>
#include <complex.h>
#include "core/params.h"
#include "core/types.h"


/*
 * Generate GPS L1 C/A code sequence for a specific PRN
 * Generates n_samples chips starting from the specified chip position
 *
 * Parameters:
 *   prn        - PRN number (1-37)
 *   buffer     - Output buffer for code chips (values: +1/-1)
 *   n_samples  - Number of samples to generate
 *   start_chip - Starting chip position (0-1022)
 */
void gen_code_L1CA(int prn, int8_t *buffer, int n_samples, int start_chip);

/*
 * Sample the generated code for FFT processing
 * Maps code chips to ADC sample rate with proper interpolation
 *
 * Parameters:
 *   code         - Generated code sequence (+1/-1 values)
 *   code_sampled - Output buffer for sampled code (complex)
 *   fft_size     - FFT size (may be larger than code length for zero-padding)
 *   recv         - Receiver configuration (for sampling rate)
 *   coff         - Code phase offset in chips
 */
void sample_code(
    const int8_t *code,
    float complex *code_sampled,
    int fft_size,
    const receiver_t *recv,
    double coff
);

/*
 * Generate frequency-domain representation of the code
 * Computes FFT of the sampled code and applies complex conjugation
 * for correlation processing
 *
 * Parameters:
 *   code_sampled - Time-domain sampled code (complex)
 *   code_fft     - Output frequency-domain representation
 *   fft_size     - FFT size
 */
void gen_code_fft(
    fftwf_complex *code_sampled,
    fftwf_complex *code_fft,
    int fft_size
);

#endif /* DSP_CODE_H */
