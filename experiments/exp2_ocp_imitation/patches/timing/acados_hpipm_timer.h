#ifndef FRS_HPIPM_TIMER_H
#define FRS_HPIPM_TIMER_H
/* Project-private, double-precision HPIPM instrumentation. No solver API changes. */
#ifdef _MSC_VER
#define FRS_HPIPM_EXPORT __declspec(dllexport)
#else
#define FRS_HPIPM_EXPORT __attribute__((visibility("default")))
#endif
enum { FRS_FACTOR, FRS_SOLVE, FRS_RESIDUAL, FRS_REDUCTION, FRS_CONDENSING,
       FRS_VECTOR, FRS_FORWARD, FRS_ADJOINT, FRS_IPM_CONTROL, FRS_NUM_KINDS };
FRS_HPIPM_EXPORT void frs_hpipm_timer_reset_v3(void);
FRS_HPIPM_EXPORT double frs_hpipm_timer_read_v3(void);
FRS_HPIPM_EXPORT unsigned long long frs_hpipm_timer_count_v3(int kind);
FRS_HPIPM_EXPORT unsigned long long frs_hpipm_timer_qp_calls_v4(void);
void frs_hpipm_record_qp_call_v4(void);
void frs_hpipm_kernel_begin_v3(int kind);
void frs_hpipm_kernel_end_v3(void);
double frs_hpipm_kernel_value_v3(double value);
#define FRS_HPIPM_VOID(kind, call) do { frs_hpipm_kernel_begin_v3(kind); call; frs_hpipm_kernel_end_v3(); } while (0)
#define FRS_HPIPM_IF(test, kind, call) do { if (test) { FRS_HPIPM_VOID(kind, call); } else { call; } } while (0)
#define FRS_HPIPM_VALUE(kind, call) (frs_hpipm_kernel_begin_v3(kind), frs_hpipm_kernel_value_v3(call))
#endif
