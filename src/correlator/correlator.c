/*
 * Correlation Functions for GPS Signal Processing Implementation
 * 
 * This module provides three correlation methods:
 * - Sequential time-domain correlation
 * - Parallel frequency-domain (FFT-based) correlation
 * - Parallel code-domain (FFT-based) correlation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <cuComplex.h>
#include <fftw3.h>
#include <complex.h>

#include "core/params.h"
#include "core/types.h"
#include "core/utils.h"
#include "dsp/code.h"
#include "dsp/mixer.h"
#include "dsp/mixer_gpu.cuh"
#include "correlator/correlator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void corr_sequential(
    const float complex *signal,
    const int8_t *local_code,
    const receiver_t *recv,
    const correlator_config_t *cfg,
    double *corr_map
) {

    // N of samples in one code period
    int N = (int) round(recv->f_adc * (CODE_PERIOD_MS / 1000.0));

    float complex *code_buf = (float complex *) malloc(sizeof(float complex) * N);
    float complex *sig_buf = (float complex *) malloc(sizeof(float complex) * N);

    // Search code start chip
    for (int c = 0; c < GPS_CODE_LEN; c++) {

        // Make ADC code representation (starting with chip c)
        sample_code(local_code, code_buf, N, recv, c);

        // Search Doppler offset
        for (int d = 0; d < cfg->n_dop; d++) {

            // Current frequency guess: IF + Doppler
            double f_curr = recv->f_if + cfg->dop_min + d * cfg->dop_step;

            mix_freq(signal, sig_buf, N, f_curr, recv);

            float complex sum = cpx_dot_product(N, sig_buf, code_buf);

            float re = crealf(sum);
            float im = cimagf(sum);
            
            //int c_offset = c;                                     // If so, res is code start phase
            int c_offset = (GPS_CODE_LEN - c) % GPS_CODE_LEN;       // If so, res is code offset

            // Compute power (with accumulation support)
            corr_map[d * GPS_CODE_LEN + c_offset] += (double) (re * re + im * im);
        }
    }
    free(code_buf);
    free(sig_buf);
}



void corr_parallel_freq(
    const float complex *signal,
    const int8_t *local_code,
    const receiver_t *recv,
    const correlator_config_t *cfg,
    double *corr_map
) {

    // N of samples in one code period
    int N = (int) round(recv->f_adc * (CODE_PERIOD_MS / 1000.0));

    int fft_size = N;  // TODO: zero padding

    // Buffers
    fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    float complex *code_buf = (float complex *) malloc(sizeof(float complex) * fft_size);

    if (!in || !out || !code_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    fftwf_plan plan = fftwf_plan_dft_1d(fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    if (!plan) {
        fprintf(stderr, "FFT Plan creation failed\n");
        return;
    }

    int f_sign = recv->iq ? -1 : 1;
    double df = recv->f_adc / fft_size;

    // Search code start chip
    for (int c = 0; c < GPS_CODE_LEN; c++) {

        // Make ADC code representation (starting with chip c)
        sample_code(local_code, code_buf, N, recv, c);

        cpx_vec_mul(N, signal, code_buf, (float complex *) in);

        for (int i = N; i < fft_size; i++) {
            in[i][0] = 0.0f;
            in[i][1] = 0.0f;
        }

        // Forward FFT of sig*code
        fftwf_execute(plan);

        // Search Doppler offset
        for (int d = 0; d < cfg->n_dop; d++) {

            // Current frequency guess: IF + Doppler
            double f_curr = f_sign*(recv->f_if + cfg->dop_min + d * cfg->dop_step);

            int freq_bin = (int) round(f_curr / df);
            while (freq_bin < 0) freq_bin += fft_size;
            while (freq_bin >= fft_size) freq_bin -= fft_size;

            float re = out[freq_bin][0];
            float im = out[freq_bin][1];

            //int c_offset = c;                                     // If so, res is code start phase
            int c_offset = (GPS_CODE_LEN - c) % GPS_CODE_LEN;       // If so, res is code offset

            // Compute power (with accumulation support)
            corr_map[d * GPS_CODE_LEN + c_offset] += (double) (re * re + im * im);
        }
    }

    fftwf_destroy_plan(plan);

    fftwf_free(in);
    fftwf_free(out);
    free(code_buf);
}


void corr_parallel_code(
    const float complex *signal,
    const int8_t *local_code,
    const receiver_t *recv,
    const correlator_config_t *cfg,
    double *corr_map
) {
    // N of samples in one code period
    int N = (int) round(recv->f_adc * (CODE_PERIOD_MS / 1000.0));

    int fft_size = N;  // TODO: zero padding

    // Buffers
    fftwf_complex *sig_buf = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *code_buf = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *code_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *prod_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *corr_ifft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);

    if (!sig_buf || !code_buf || !code_fft || !prod_fft || !corr_ifft) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    fftwf_plan plan_fwd = fftwf_plan_dft_1d(fft_size, sig_buf, prod_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan plan_inv = fftwf_plan_dft_1d(fft_size, prod_fft, corr_ifft, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!plan_fwd || !plan_inv) {
        fprintf(stderr, "FFT Plan creation failed\n");
        return;
    }

    // Get FFT of properly sampled code
    sample_code(local_code, (float complex *) code_buf, fft_size, recv, 0.0);
    gen_code_fft(code_buf, code_fft, fft_size);
    fftwf_free(code_buf);

    // Search Doppler offset
    for (int d = 0; d < cfg->n_dop; d++) {

        // Current frequency guess: IF + Doppler
        double f_curr = recv->f_if + cfg->dop_min + d * cfg->dop_step;

        mix_freq(signal, (float complex *) sig_buf, N, f_curr, recv);

        // Forward FFT of signal
        fftwf_execute(plan_fwd);

        // Frequency-domain multiplication (CPU)
        cpx_vec_mul(fft_size, (float complex *) prod_fft, (float complex *) code_fft, (float complex *) prod_fft);

        // Frequency-domain multiplication (CUDA)
        /*
        int ret = cpx_vec_mul_cuda(
            fft_size,
            (cuFloatComplex *) prod_fft,
            (cuFloatComplex *) code_fft,
            (cuFloatComplex *) prod_fft
        );

        if (ret != 0) {
            fprintf(stderr, "CUDA multiplication failed!\n");

            fftwf_destroy_plan(plan_fwd);
            fftwf_destroy_plan(plan_inv);

            fftwf_free(sig_buf);
            fftwf_free(code_fft);
            fftwf_free(prod_fft);
            fftwf_free(corr_ifft);

            return;
        }
        */

        // Inverse FFT
        fftwf_execute(plan_inv);


        // Search code offset (samples)
        for (int s = 0; s < N; s++) {
            float re = corr_ifft[s][0] / fft_size;
            float im = corr_ifft[s][1] / fft_size;

            int c_offset = s;               // If so, res is code offset
            //int c_offset = (N - s) % Т;       // If so, res is code start phase

            // Compute power (with accumulation support)
            corr_map[d * N + c_offset] += (double) (re * re + im * im);
        }
    }


    fftwf_destroy_plan(plan_fwd);
    fftwf_destroy_plan(plan_inv);

    fftwf_free(sig_buf);
    fftwf_free(code_fft);
    fftwf_free(prod_fft);
    fftwf_free(corr_ifft);
}
