/*
 * GPS Signal Acquisition Application
 * 
 * This application performs satellite signal acquisition from recorded
 * GPS data. It searches for satellites across Doppler and code phase
 * dimensions, detects peaks, and estimates C/N0 ratios.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <complex.h>

#include <time.h>
#include <getopt.h>

#include "core/params.h"
#include "core/types.h"
#include "core/utils.h"
#include "dsp/code.h"
#include "correlator/corr_interface.h"
#include "io/file_io.h"
#include "io/config_io.h"
#include "acquisition/acq_processor.h"


#define DEFAULT_THRESHOLD_DB 35.0  // Default detection threshold (C/N0 in dB-Hz)

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static void print_usage(const char *prog_name) {
    printf("GPS Signal Acquisition\n\n");
    printf("Usage: %s -f <input_file> [options]\n\n", prog_name);
    printf("Required arguments:\n");
    printf("  -f, --file <path>           Input binary file (int8_t samples).\n");
    printf("\nOptional arguments:\n");
    printf("  --dop_min <freq>            Minimum Doppler search frequency in Hz. (Default: %.0f)\n", DEFAULT_DOP_MIN);
    printf("  --dop_max <freq>            Maximum Doppler search frequency in Hz. (Default: %.0f)\n", DEFAULT_DOP_MAX);
    printf("  --dop_step <freq>           Doppler search step in Hz. (Default: %.1f)\n", DEFAULT_DOP_STEP);
    printf("  --tintegration <ms>         Integration time in ms. (Default: %.1f)\n", DEFAULT_T_INT_MS);
    printf("  --iq                        Specify if input file contains IQ data (default is I-only).\n");
    printf("  --method <1|2|3>            Correlation method:\n");
    printf("                              1: Time Domain (Brute Force)\n");
    printf("                              2: Parallel Frequency Search\n");
    printf("                              3: Parallel Code Search (FFT, Default)\n");
    printf("  --prn <num>                 Search only specified PRN (1-37). If not specified, scan all.\n");
    printf("  --threshold <cn0>           Detection threshold (C/N0 in dB-Hz). (Default: %.1f)\n", DEFAULT_THRESHOLD_DB);
    printf("  -h, --help                  Show this help message.\n");
    printf("\nOutput:\n");
    printf("  Table of detected satellites with parameters:\n");
    printf("  PRN | Status | C/N0 (dB-Hz) | Doppler (Hz) | Code Phase (chips)\n");
    printf("Example:\n");
    printf("  %s -f test_iq.bin --tint=10 --iq --method 3 --threshold 35 \n", prog_name);
}

/*
 * Print result table header
 */
static void print_result_header(void) {
    printf("\n");
    printf("==================================================================\n");
    printf("                  GPS SIGNAL ACQUISITION RESULTS                  \n");
    printf("==================================================================\n");
    printf(" PRN | Status | C/N0 (dB-Hz) | Doppler (Hz) | Code Offset (ms)   |\n");
    printf("-----+--------+--------------+--------------+--------------------+\n");
}

/*
 * Print result row for a single satellite
 */
static void print_result_row(const satellite_channel_t *result) {

    printf(" %3d | %6s | %10.2f   | %10.2f   | %17.5f\n",
            result->prn,
            result->active ? "FOUND" : "----",
            result->cn0_db_hz,
            result->fdoppler,
            result->code_phase_start_chips/GPS_CODE_LEN);

}

/*
 * Print summary statistics
 */
static void print_summary(int total_prn, int detected_count, double max_cn0, int best_prn) {
    printf("-----+--------+--------------+--------------+-------------------+\n");
    printf("\nSummary:\n");
    printf("  Total PRN scanned: %d\n", total_prn);
    printf("  Satellites detected: %d\n", detected_count);
    if (detected_count > 0) {
        printf("  Best C/N0: %.2f dB-Hz (PRN %d)\n", max_cn0, best_prn);
    }
    printf("\n");
}

/* ============================================================================
 * BATCH ACQUISITION FUNCTION
 * ============================================================================ */

/*
 * Perform acquisition for a range of PRNs using batch correlation.
 */
static int acquire_batch_prn(
    const float complex *signal,
    int prn_start,
    int prn_end,
    const receiver_t *recv,
    double dop_min,
    double dop_max,
    double dop_step,
    double t_integration_ms,
    correlator_method_t method,
    double threshold_db,
    satellite_channel_t *results
) {
    int num_periods = (int)(t_integration_ms / CODE_PERIOD_MS);
    if (num_periods < 1) num_periods = 1;

    int n_samples_per_period = (int)round(recv->f_adc * (CODE_PERIOD_MS / 1000.0));
    int num_prns = prn_end - prn_start + 1;

    // Allocate memory for code array (all codes concatenated)
    int8_t *local_codes = (int8_t*)malloc(sizeof(int8_t) * GPS_CODE_LEN * num_prns);
    if (!local_codes) {
        fprintf(stderr, "Error: Failed to allocate memory for local codes.\n");
        return -1;
    }

    // Generate codes for all satellites
    for (int i = 0; i < num_prns; i++) {
        int prn = prn_start + i;
        gen_code_L1CA(prn, local_codes + (i * GPS_CODE_LEN), GPS_CODE_LEN, 0);
    }

    // Allocate memory for results
    int n_rows = (int)round((dop_max - dop_min) / dop_step);
    int n_cols = (method == METHOD_PARALLEL_CODE) ? n_samples_per_period : GPS_CODE_LEN;

    double **corr_maps = (double**)malloc(sizeof(double*) * num_prns);

    if (!corr_maps) {
        fprintf(stderr, "Error: Failed to allocate memory for correlation maps pointer array.\n");
        free(local_codes);
        return -1;
    }

    for (int i = 0; i < num_prns; i++) {
        corr_maps[i] = (double*)malloc(sizeof(double) * n_rows * n_cols);
        if (!corr_maps[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for correlation map %d.\n", i);
            for (int j = 0; j < i; j++) {
                free(corr_maps[j]);
            }
            free(corr_maps);
            free(local_codes);
            return -1;
        }
        memset(corr_maps[i], 0, sizeof(double) * n_rows * n_cols);
    }

    // Configure correlator for batch processing
    correlator_config_t config;
    config.dop_min = dop_min;
    config.dop_step = dop_step;
    config.n_dop = n_rows;
    config.num_periods = num_periods;
    config.method = method;
    config.verbose = 0;
    config.num_prns = num_prns;
    
    for (int i = 0; i < MAX_SATS; i++) {
        config.prns[i] = 0;
    }
    for (int i = 0; i < num_prns; i++) {
        config.prns[i] = prn_start + i;
    }

    // Execute batch correlation
    int ret = batch_corr_execute(
        signal,
        local_codes, &config, recv,
        corr_maps
    );

    if (ret != 0) {
        fprintf(stderr, "Error: batch_corr_execute failed.\n");
        for (int i = 0; i < num_prns; i++) {
            free(corr_maps[i]);
        }
        free(corr_maps);
        free(local_codes);
        return -1;
    }

    // Analyze results for each satellite
    double t_integration_sec = t_integration_ms / 1000.0;

    for (int i = 0; i < num_prns; i++) {
        int prn = prn_start + i;
        
        perform_acquisition(
            corr_maps[i],
            n_rows,
            n_cols,
            dop_min,
            dop_step,
            prn,
            t_integration_sec,
            threshold_db,
            &results[prn - 1]
        );
    }

    // Cleanup
    for (int i = 0; i < num_prns; i++) {
        free(corr_maps[i]);
    }
    free(corr_maps);
    free(local_codes);

    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    // Default parameters
    const char *input_file = NULL;
    int is_iq = 0;  // Default: I-only data
    double doppler_min = DEFAULT_DOP_MIN;
    double doppler_max = DEFAULT_DOP_MAX;
    double doppler_step = DEFAULT_DOP_STEP;
    double t_integration_ms = DEFAULT_T_INT_MS;
    correlator_method_t method = METHOD_PARALLEL_CODE;
    int specific_prn = -1;  // -1 = scan all
    double threshold_db = DEFAULT_THRESHOLD_DB;

    // Parse command line arguments
    static struct option long_options[] = {
        {"file",         required_argument, 0, 'f'},
        {"dop_min",      required_argument, 0, 'd'},
        {"dop_max",      required_argument, 0, 'm'},
        {"dop_step",     required_argument, 0, 's'},
        {"tintegration", required_argument, 0, 't'},
        {"iq",           no_argument,       0, 'i'},
        {"method",       required_argument, 0, 'M'},
        {"prn",          required_argument, 0, 'p'},
        {"threshold",    required_argument, 0, 'T'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "f:d:m:s:t:iM:p:T:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                input_file = optarg;
                break;
            case 'd':
                doppler_min = atof(optarg);
                break;
            case 'm':
                doppler_max = atof(optarg);
                break;
            case 's':
                doppler_step = atof(optarg);
                break;
            case 't':
                t_integration_ms = atof(optarg);
                break;
            case 'i':
                is_iq = 1;
                break;
            case 'M':
                {
                    int m = atoi(optarg);
                    if (m == 1) method = METHOD_TIME_DOMAIN;
                    else if (m == 2) method = METHOD_PARALLEL_FREQ;
                    else if (m == 3) method = METHOD_PARALLEL_CODE;
                    else {
                        fprintf(stderr, "Error: Invalid method %d. Use 1, 2, or 3.\n", m);
                        return 1;
                    }
                }
                break;
            case 'p':
                specific_prn = atoi(optarg);
                if (specific_prn < 1 || specific_prn > 37) {
                    fprintf(stderr, "Error: PRN must be between 1 and 37.\n");
                    return 1;
                }
                break;
            case 'T':
                threshold_db = atof(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Check required arguments
    if (!input_file) {
        fprintf(stderr, "Error: Input file is required (-f option).\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Read receiver configuration
    receiver_t recv = read_receiver_config(RECV_CONFIG_FILE);
    recv.iq = is_iq;

    // Open input file to determine size
    FILE *fp = fopen(input_file, "rb");
    if (!fp) {
        perror("Error opening input file");
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    fclose(fp);

    // Round to integer number of periods
    int n_samples_per_period = (int)round(recv.f_adc * CODE_PERIOD_MS / 1000.0);
    int num_periods = (int)round(t_integration_ms / CODE_PERIOD_MS);

    if (num_periods < 1) {
        fprintf(stderr, "Error: Integration time must be at least %.1f ms.\n", CODE_PERIOD_MS);
        return EXIT_FAILURE;
    }

    int n_samples_total = n_samples_per_period * num_periods;


    // Read signal
    float complex *signal = read_signal(input_file, is_iq, n_samples_total);
    if (!signal) {
        fprintf(stderr, "Error: Failed to read input signal.\n");
        return 1;
    }

    // Results array
    satellite_channel_t results[MAX_SATS];
    memset(results, 0, sizeof(results));

    // Determine PRN range to scan
    int prn_start = (specific_prn > 0) ? specific_prn : 1;
    int prn_end = (specific_prn > 0) ? specific_prn : MAX_SATS;

    double total_start = get_time_sec();

    // Perform batch acquisition for all PRNs
    int ret = acquire_batch_prn(
        signal,
        prn_start, prn_end,
        &recv,
        doppler_min, doppler_max, doppler_step,
        t_integration_ms,
        method,
        threshold_db,
        results
    );

    if (ret != 0) {
        fprintf(stderr, "Error: acquire_batch_prn failed.\n");
        free(signal);
        return EXIT_FAILURE;
    }

    double total_time = get_time_sec() - total_start;

    // Count statistics
    int detected_count = 0;
    double max_cn0 = 0.0;
    int best_prn = -1;

    for (int prn = prn_start; prn <= prn_end; prn++) {
        if (results[prn - 1].active) {
            detected_count++;
            if (results[prn - 1].cn0_db_hz > max_cn0) {
                max_cn0 = results[prn - 1].cn0_db_hz;
                best_prn = prn;
            }
        }
    }

    // Print results
    print_result_header();

    for (int prn = prn_start; prn <= prn_end; prn++) {
        print_result_row(&results[prn - 1]);
    }

    print_summary(prn_end - prn_start + 1, detected_count, max_cn0, best_prn);
    
    printf("Total acquisition time: %.3f seconds\n", total_time);
    if (detected_count > 0) {
        printf("Average time per satellite: %.3f seconds\n", total_time / (prn_end - prn_start + 1));
    }

    // Cleanup
    free(signal);

    return 0;
}
