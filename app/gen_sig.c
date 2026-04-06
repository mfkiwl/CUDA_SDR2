/*
 * GPS L1 C/A Synthetic Signal Generator (Multi-Satellite)
 * 
 * This application generates synthetic GPS L1 C/A signals with multiple
 * satellites. It supports configurable Doppler shifts, code phases,
 * carrier phases, and C/N0 ratios for each satellite.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <getopt.h>
#include <math.h>

#include "core/params.h"
#include "core/types.h"
#include "dsp/code.h"
#include "io/config_io.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/*
 * Generate Gaussian noise using Box-Muller transform.
 * Returns a normally distributed random number with mean 0 and std dev sigma.
 */
double gaussian_noise(double sigma) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return sigma * spare;
    }
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    has_spare = 1;
    return sigma * u * s;
}

/*
 * Quantize a sample to 2-bit representation.
 * Uses threshold-based quantization with levels {-3, -1, 1, 3}.
 */
int8_t quantize_2bit(double val, double sigma_noise) {
    double threshold = 1 * sigma_noise;

    if (val > threshold) return 3;
    if (val < -threshold) return -3;
    return (val >= 0) ? 1 : -1;
}


/*
 * Generate multi-satellite GPS L1 C/A signal.
 * Writes int8_t samples to binary file.
 */
void generate_L1CA_signal(
    satellite_channel_t *sats,
    int num_sats,
    int duration_ms,
    const char *filename,
    const receiver_t *recv
) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    static int8_t prn_codes[MAX_SATS][GPS_CODE_LEN];

    int active_count = 0;
    for (int i = 0; i < num_sats; i++) {
        if (sats[i].active && sats[i].prn >= 1 && sats[i].prn <= 37) {
            gen_code_L1CA(sats[i].prn, prn_codes[sats[i].prn - 1], GPS_CODE_LEN, 0);
            active_count++;
        }
    }

    if (active_count == 0) {
        printf("Warning: No active satellites to generate.\n");
        fclose(fp);
        return;
    }

    int total_samples = (int)(recv->f_adc * duration_ms / 1000.0);
    double dt = 1.0 / recv->f_adc;
    double sigma_noise = 1.0;

    double current_carrier_phase[MAX_SATS];
    double current_chip_index[MAX_SATS];
    double phase_step_carrier[MAX_SATS];
    double signal_amplitude[MAX_SATS];
    double chip_step = FS_GPS / recv->f_adc;

    for (int i = 0; i < num_sats; i++) {
        if (!sats[i].active) continue;

        int idx = sats[i].prn - 1;

        double cn0_lin = pow(10.0, sats[i].cn0_db_hz / 10.0);
        signal_amplitude[idx] = sqrt(cn0_lin / recv->f_bw) * sigma_noise;

        double carrier_freq = recv->f_if + sats[i].fdoppler;
        phase_step_carrier[idx] = 2.0 * M_PI * carrier_freq * dt;

        current_carrier_phase[idx] = sats[i].carrier_phase_start_rad;

        current_chip_index[idx] = sats[i].code_phase_start_chips;
        while (current_chip_index[idx] >= GPS_CODE_LEN) current_chip_index[idx] -= GPS_CODE_LEN;
        while (current_chip_index[idx] < 0) current_chip_index[idx] += GPS_CODE_LEN;
    }

    printf("Generating MULTI-SAT signal (%d active sats, %d ms)...\n", active_count, duration_ms);

    for (int i = 0; i < total_samples; i++) {
        double i_sum = 0.0;
        double q_sum = 0.0;


        for (int s = 0; s < num_sats; s++) {
            if (!sats[s].active) continue;

            int prn_idx = sats[s].prn - 1;

            int idx_chip = (int) current_chip_index[prn_idx];

            int code_val = prn_codes[prn_idx][idx_chip];

            double cos_val = cos(current_carrier_phase[prn_idx]);
            double sin_val = sin(current_carrier_phase[prn_idx]);

            double amp = signal_amplitude[prn_idx] * code_val;

            i_sum += amp * cos_val;
            q_sum += amp * (-sin_val);

            current_carrier_phase[prn_idx] += phase_step_carrier[prn_idx];
            if (current_carrier_phase[prn_idx] >= 2.0 * M_PI) current_carrier_phase[prn_idx] -= 2.0 * M_PI;
            if (current_carrier_phase[prn_idx] < 0) current_carrier_phase[prn_idx] += 2.0 * M_PI;

            current_chip_index[prn_idx] += chip_step;
            if (current_chip_index[prn_idx] >= GPS_CODE_LEN) current_chip_index[prn_idx] -= GPS_CODE_LEN;
        }

        double i_raw = i_sum + gaussian_noise(sigma_noise);
        double q_raw = q_sum + gaussian_noise(sigma_noise);

        if (recv->iq) {
            int8_t i_digit = quantize_2bit(i_raw, sigma_noise);
            int8_t q_digit = quantize_2bit(q_raw, sigma_noise);
            fwrite(&i_digit, sizeof(int8_t), 1, fp);
            fwrite(&q_digit, sizeof(int8_t), 1, fp);
        } else {
            int8_t i_digit = quantize_2bit(i_raw, sigma_noise);
            fwrite(&i_digit, sizeof(int8_t), 1, fp);
        }
    }

    fclose(fp);
    printf("Done. File: %s\n", filename);
}


/*
 * Parse satellite configuration file.
 * Each line: PRN DOPPLER_HZ CODE_PHASE_CHIPS CARRIER_PHASE_RAD CN0_DB_HZ
 */
int parse_config_file(const char *filename, satellite_channel_t *sats, int max_sats) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening config file");
        return -1;
    }

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        char *trimmed = line;
        while (isspace((unsigned char)*trimmed)) trimmed++;
        if (*trimmed == '\0' || *trimmed == '#') continue;

        if (count >= max_sats) {
            fprintf(stderr, "Warning: Maximum number of satellites (%d) reached.\n", max_sats);
            break;
        }

        satellite_channel_t sat;
        sat.active = 0;

        int items = sscanf(trimmed, "%d %lf %lf %lf %lf",
                           &sat.prn,
                           &sat.fdoppler,
                           &sat.code_phase_start_chips,
                           &sat.carrier_phase_start_rad,
                           &sat.cn0_db_hz);

        if (items == 5) {
            if (sat.prn < 1 || sat.prn > 37) {
                fprintf(stderr, "Warning: Invalid PRN %d. Skipping.\n", sat.prn);
                continue;
            }
            sat.active = 1;
            sats[count++] = sat;
        } else {
            fprintf(stderr, "Warning: Malformed line in config file. Skipping.\n");
        }
    }
    fclose(fp);
    return count;
}

/*
 * ============================================================================
 * MAIN WITH ARGUMENT PARSING
 * ============================================================================
 */

void print_usage(const char *prog_name) {
    printf("GPS L1 C/A Synthetic Signal Generator (Multi-Satellite)\n\n");
    printf("Usage: %s -f <config> -d <duration_ms> -o <output> [-m <mode>]\n\n", prog_name);
    printf("Required arguments:\n");
    printf("  -f, --config <file>      Path to the satellite configuration file.\n");
    printf("  -d, --duration <ms>      Signal duration in milliseconds (integer).\n");
    printf("  -o, --output <file>      Path to the output binary file.\n\n");
    printf("Optional arguments:\n");
    printf("  -m, --mode <mode>        Output mode: 'i' (I-channel only, default) or 'iq'.\n");
    printf("  -h, --help               Show this help message.\n\n");
    printf("Config file format (space-separated, one satellite per line):\n");
    printf("  PRN  DOPPLER_HZ  CODE_PHASE_CHIPS  CARRIER_PHASE_RAD  CN0_DB_HZ\n");
    printf("  Lines starting with '#' are treated as comments.\n\n");
    printf("Example:\n");
    printf("  %s -f testsig.cfg -d 100 -o test_iq.bin -m iq\n", prog_name);
}

int main(int argc, char *argv[]) {
    char *config_file = NULL;
    char *output_file = NULL;
    int duration_ms = 0;
    int mode_iq = 0;

    static struct option long_options[] = {
        {"config",   required_argument, 0, 'f'},
        {"duration", required_argument, 0, 'd'},
        {"output",   required_argument, 0, 'o'},
        {"mode",     required_argument, 0, 'm'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "f:d:o:m:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                config_file = optarg;
                break;
            case 'd':
                duration_ms = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'm':
                if (strcmp(optarg, "i") == 0 || strcmp(optarg, "I") == 0) {
                    mode_iq = 0;
                } else if (strcmp(optarg, "iq") == 0 || strcmp(optarg, "IQ") == 0) {
                    mode_iq = 1;
                } else {
                    fprintf(stderr, "Error: Invalid mode '%s'. Use 'iq' or 'i'.\n", optarg);
                    return EXIT_FAILURE;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (config_file == NULL || output_file == NULL || duration_ms <= 0) {
        fprintf(stderr, "Error: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    satellite_channel_t sats[MAX_SATS];
    for(int i=0; i<MAX_SATS; i++) sats[i].active = 0;

    printf("Reading configuration from: %s\n", config_file);
    int num_sats = parse_config_file(config_file, sats, MAX_SATS);

    if (num_sats <= 0) {
        fprintf(stderr, "Error: No valid satellites found in config file.\n");
        return EXIT_FAILURE;
    }

    printf("Found %d valid satellite(s).\n", num_sats);

    receiver_t recv = read_receiver_config(RECV_CONFIG_FILE);
    recv.iq = mode_iq;

    printf("Receiver config: F_ADC=%.2f MHz, F_BW=%.2f MHz, F_LO=%.3f MHz, F_IF=%.2f MHz\n",
           recv.f_adc / 1e6, recv.f_bw / 1e6, recv.f_lo / 1e6, recv.f_if / 1e6);

    generate_L1CA_signal(sats, num_sats, duration_ms, output_file, &recv);

    return EXIT_SUCCESS;
}
