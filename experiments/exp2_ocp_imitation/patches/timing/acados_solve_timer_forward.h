#ifndef FRS_ACADOS_SOLVE_TIMER_FORWARD_H
#define FRS_ACADOS_SOLVE_TIMER_FORWARD_H
#ifdef _MSC_VER
#define FRS_TIMING_EXPORT __declspec(dllexport)
#define FRS_HPIPM_IMPORT __declspec(dllimport)
#else
#define FRS_TIMING_EXPORT __attribute__((visibility("default")))
#define FRS_HPIPM_IMPORT
#endif
FRS_HPIPM_IMPORT void frs_hpipm_timer_reset_v3(void);
FRS_HPIPM_IMPORT double frs_hpipm_timer_read_v3(void);
FRS_HPIPM_IMPORT unsigned long long frs_hpipm_timer_count_v3(int kind);
FRS_HPIPM_IMPORT unsigned long long frs_hpipm_timer_qp_calls_v4(void);
FRS_TIMING_EXPORT void frs_acados_solve_timer_reset_v3(void) { frs_hpipm_timer_reset_v3(); }
FRS_TIMING_EXPORT double frs_acados_solve_timer_read_v3(void) { return frs_hpipm_timer_read_v3(); }
FRS_TIMING_EXPORT unsigned long long frs_acados_solve_timer_count_v3(int kind) { return frs_hpipm_timer_count_v3(kind); }
FRS_TIMING_EXPORT int frs_acados_forward_sens_timer_enabled_v3(void) { return 1; }
FRS_TIMING_EXPORT int frs_acados_hpipm_kernel_timer_enabled_v3(void) { return 1; }
FRS_TIMING_EXPORT void frs_acados_solve_timer_reset_v4(void) { frs_hpipm_timer_reset_v3(); }
FRS_TIMING_EXPORT double frs_acados_solve_timer_read_v4(void) { return frs_hpipm_timer_read_v3(); }
FRS_TIMING_EXPORT unsigned long long frs_acados_solve_timer_count_v4(int kind) { return frs_hpipm_timer_count_v3(kind); }
FRS_TIMING_EXPORT unsigned long long frs_acados_solve_timer_qp_calls_v4(void) { return frs_hpipm_timer_qp_calls_v4(); }
FRS_TIMING_EXPORT int frs_acados_hpipm_kernel_timer_enabled_v4(void) { return 1; }
#endif
