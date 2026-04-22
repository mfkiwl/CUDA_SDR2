/*
 * CUDA Batch Correlator Implementation (GPU)
 *
 * This module provides a CUDA-based batch correlator for GPS signal
 * processing on the GPU. It performs frequency-domain correlation across
 * multiple satellites, Doppler bins, and code periods with non-coherent
 * power accumulation
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include <stdint.h>
#include <stdio.h>

#include "core/params.h"
#include "core/types.h"
#include "correlator_gpu.cuh"
#include "dsp/mixer_gpu.cuh"
#include "dsp/code_gpu.cuh"


__global__ void power_accum_batched_kernel(
    const cuFloatComplex *d_prod,
    float *d_accum,
    int batch_p,
    int num_sats,
    int n_dop,
    int N,
    int fft_size
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_per_period = (long)num_sats * n_dop * N;
    long total_all = (long)batch_p * total_per_period;
    
    if (idx >= total_all) return;
    
    // Decompose index
    long local_idx = idx % total_per_period;
    
    cuFloatComplex val = d_prod[idx];
    float re = val.x / fft_size;
    float im = val.y / fft_size;
    float power = re * re + im * im;
    
    // Atomic add to shared accumulator
    atomicAdd(&d_accum[local_idx], power);
}


int correlate_block_gpu(
    satellite_channel_block_t *blk,
    int num_periods
) {
    if (!blk) return -1;

    int num_sats = blk->num_channels;
    int N = blk->config.n_samples;
    int fft_size = N;
    int n_dop = blk->config.n_dop;

    // Zero the accumulator
    size_t accum_bytes = (size_t)num_sats * n_dop * N * sizeof(float);
    cudaMemset(blk->d_corr_maps, 0, accum_bytes);

    // Calculate grid sizes for batch processing
    int mix_grid = ((long)num_periods * n_dop * fft_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int mul_grid = ((long)num_periods * num_sats * n_dop * fft_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int pow_batch_grid = ((long)num_periods * num_sats * n_dop * N + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Mix signal for all Doppler bins and periods
    mix_batch_kernel<<<mix_grid, CUDA_BLOCK_SIZE>>>(
        blk->rf_ch->buffer, blk->d_mixed, N, fft_size, n_dop, num_periods, blk->d_phase_steps
    );

    // FFT all mixed signals
    cufftResult cufft_err = cufftExecC2C(blk->plan_fwd,
                        blk->d_mixed,
                        blk->d_mixed,
                        CUFFT_FORWARD);
    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT mixed forward failed (code %d)\n", cufft_err);
        return -1;
    }

    // Multiply mixed FFT with precomputed code FFTs 
    multiply_batch_kernel<<<mul_grid, CUDA_BLOCK_SIZE>>>(
        blk->d_mixed, blk->d_codes_fft, blk->d_prod,
        num_sats, n_dop, fft_size, num_periods
    );

    // IFFT all products
    cufft_err = cufftExecC2C(blk->plan_inv,
                        blk->d_prod,
                        blk->d_prod,
                        CUFFT_INVERSE);
    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT product inverse failed (code %d)\n", cufft_err);
        return -1;
    }

    // Power accumulation
    power_accum_batched_kernel<<<pow_batch_grid, CUDA_BLOCK_SIZE>>>(
        blk->d_prod, blk->d_corr_maps, num_periods, num_sats, n_dop, N, fft_size
    );

    // Check for kernel errors
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in process_block: %s\n", cudaGetErrorString(kerr));
        return -1;
    }

    // Synchronize to ensure results are ready
    cudaDeviceSynchronize();
    return 0;
}
