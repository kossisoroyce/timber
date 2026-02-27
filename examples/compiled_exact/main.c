/* main.c — Run Timber compiled model inference on test samples */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

#define MAX_SAMPLES 100
#define MAX_LINE 8192

int main(int argc, char** argv) {
    const char* csv_path = "test_samples.csv";
    if (argc > 1) csv_path = argv[1];

    /* Init model context */
    TimberCtx* ctx;
    int rc = timber_init(&ctx);
    if (rc != 0) {
        fprintf(stderr, "Failed to initialize model context\n");
        return 1;
    }

    /* Read CSV */
    FILE* f = fopen(csv_path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", csv_path);
        return 1;
    }

    float samples[MAX_SAMPLES][TIMBER_N_FEATURES];
    int n_samples = 0;
    char line[MAX_LINE];

    /* Skip header */
    if (fgets(line, sizeof(line), f) == NULL) {
        fprintf(stderr, "Empty file\n");
        fclose(f);
        return 1;
    }

    while (fgets(line, sizeof(line), f) && n_samples < MAX_SAMPLES) {
        char* tok = strtok(line, ",");
        int col = 0;
        while (tok && col < TIMBER_N_FEATURES) {
            samples[n_samples][col] = (float)atof(tok);
            tok = strtok(NULL, ",\n");
            col++;
        }
        if (col == TIMBER_N_FEATURES) {
            n_samples++;
        }
    }
    fclose(f);

    printf("Timber Inference — %d samples, %d features, %d trees\n",
           n_samples, TIMBER_N_FEATURES, TIMBER_N_TREES);
    printf("==================================================\n");

    /* Run inference */
    for (int i = 0; i < n_samples; i++) {
        float output[TIMBER_N_OUTPUTS];
        rc = timber_infer_single(samples[i], output, ctx);
        if (rc != 0) {
            fprintf(stderr, "Inference failed on sample %d\n", i);
            return 1;
        }
        printf("  sample %d: %.6f\n", i, output[0]);
    }

    printf("==================================================\n");
    printf("Done.\n");

    timber_free(ctx);
    return 0;
}
