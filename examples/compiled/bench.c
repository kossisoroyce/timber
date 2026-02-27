/* bench.c — Benchmark Timber compiled inference vs wall-clock timing */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model.h"

#define MAX_SAMPLES 1000
#define MAX_LINE    8192
#define WARMUP      1000
#define ITERS       10000

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    const char* csv_path = "test_samples.csv";
    if (argc > 1) csv_path = argv[1];

    TimberCtx* ctx;
    timber_init(&ctx);

    /* Load samples */
    FILE* f = fopen(csv_path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }

    float samples[MAX_SAMPLES][TIMBER_N_FEATURES];
    int n_samples = 0;
    char line[MAX_LINE];
    fgets(line, sizeof(line), f); /* skip header */

    while (fgets(line, sizeof(line), f) && n_samples < MAX_SAMPLES) {
        char* tok = strtok(line, ",");
        int col = 0;
        while (tok && col < TIMBER_N_FEATURES) {
            samples[n_samples][col] = (float)atof(tok);
            tok = strtok(NULL, ",\n");
            col++;
        }
        if (col == TIMBER_N_FEATURES) n_samples++;
    }
    fclose(f);

    printf("Timber C99 Inference Benchmark\n");
    printf("==================================================\n");
    printf("Trees:    %d\n", TIMBER_N_TREES);
    printf("Features: %d\n", TIMBER_N_FEATURES);
    printf("Samples:  %d\n", n_samples);
    printf("Warmup:   %d iters\n", WARMUP);
    printf("Timed:    %d iters\n", ITERS);
    printf("==================================================\n\n");

    float output[TIMBER_N_OUTPUTS];

    /* --- Single-sample latency --- */
    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        timber_infer_single(samples[0], output, ctx);
    }

    /* Timed single-sample runs */
    double* latencies = (double*)malloc(ITERS * sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        int idx = i % n_samples;
        double t0 = now_us();
        timber_infer_single(samples[idx], output, ctx);
        double t1 = now_us();
        latencies[i] = t1 - t0;
    }

    qsort(latencies, ITERS, sizeof(double), cmp_double);
    double p50 = latencies[ITERS / 2];
    double p95 = latencies[(int)(ITERS * 0.95)];
    double p99 = latencies[(int)(ITERS * 0.99)];
    double mean = 0;
    for (int i = 0; i < ITERS; i++) mean += latencies[i];
    mean /= ITERS;

    printf("Single-sample (batch=1):\n");
    printf("  Mean:  %.2f us\n", mean);
    printf("  P50:   %.2f us\n", p50);
    printf("  P95:   %.2f us\n", p95);
    printf("  P99:   %.2f us\n", p99);
    printf("  Throughput: %.0f samples/sec\n\n", 1e6 / mean);

    /* --- Batch latency --- */
    int batch_sizes[] = {1, 4, 10};
    int n_batch_sizes = 3;
    if (n_samples < 10) {
        batch_sizes[2] = n_samples;
    }

    for (int b = 0; b < n_batch_sizes; b++) {
        int bs = batch_sizes[b];
        if (bs > n_samples) bs = n_samples;

        float* batch_out = (float*)malloc(bs * TIMBER_N_OUTPUTS * sizeof(float));

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            timber_infer(samples[0], bs, batch_out, ctx);
        }

        /* Timed */
        int batch_iters = ITERS;
        for (int i = 0; i < batch_iters; i++) {
            double t0 = now_us();
            timber_infer(samples[0], bs, batch_out, ctx);
            double t1 = now_us();
            latencies[i] = t1 - t0;
        }

        qsort(latencies, batch_iters, sizeof(double), cmp_double);
        double bp50 = latencies[batch_iters / 2];
        double bp95 = latencies[(int)(batch_iters * 0.95)];
        double bp99 = latencies[(int)(batch_iters * 0.99)];
        double bmean = 0;
        for (int i = 0; i < batch_iters; i++) bmean += latencies[i];
        bmean /= batch_iters;

        printf("Batch=%d:\n", bs);
        printf("  Mean:  %.2f us\n", bmean);
        printf("  P50:   %.2f us\n", bp50);
        printf("  P95:   %.2f us\n", bp95);
        printf("  P99:   %.2f us\n", bp99);
        printf("  Throughput: %.0f samples/sec\n\n", bs * 1e6 / bmean);

        free(batch_out);
    }

    free(latencies);
    timber_free(ctx);
    return 0;
}
