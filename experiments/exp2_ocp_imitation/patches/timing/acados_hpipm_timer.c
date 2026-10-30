#include "acados_hpipm_timer.h"
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#define FRS_TLS __declspec(thread)
static double frs_tick(void) { LARGE_INTEGER t, f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f); return (double)t.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
#define FRS_TLS _Thread_local
static double frs_tick(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
#endif
static FRS_TLS int enabled, depth;
static FRS_TLS double seconds, started;
static FRS_TLS unsigned long long counts[FRS_NUM_KINDS];
static FRS_TLS unsigned long long qp_calls;
void frs_hpipm_timer_reset_v3(void) {
    int k; enabled = 1; depth = 0; seconds = 0.; qp_calls = 0;
    for (k = 0; k < FRS_NUM_KINDS; ++k) counts[k] = 0;
}
double frs_hpipm_timer_read_v3(void) {
    int k; unsigned long long calls = 0;
    enabled = 0;
    for (k = 0; k < FRS_NUM_KINDS; ++k) calls += counts[k];
    return calls && depth == 0 ? seconds : NAN;
}
unsigned long long frs_hpipm_timer_count_v3(int kind) {
    return kind >= 0 && kind < FRS_NUM_KINDS ? counts[kind] : 0;
}
unsigned long long frs_hpipm_timer_qp_calls_v4(void) { return qp_calls; }
void frs_hpipm_record_qp_call_v4(void) { if (enabled) ++qp_calls; }
void frs_hpipm_kernel_begin_v3(int kind) {
    if (!enabled) return;
    if (kind >= 0 && kind < FRS_NUM_KINDS) ++counts[kind];
    if (depth++ == 0) started = frs_tick();
}
void frs_hpipm_kernel_end_v3(void) {
    if (enabled && --depth == 0) seconds += frs_tick() - started;
}
double frs_hpipm_kernel_value_v3(double value) { frs_hpipm_kernel_end_v3(); return value; }
