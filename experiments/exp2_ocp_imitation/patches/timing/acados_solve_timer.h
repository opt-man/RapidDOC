/* Included once in ocp_nlp_common.c. Time real QP/adjoint kernels only. */
#include <math.h>
#include "acados/utils/timing.h"
#ifdef _MSC_VER
#define FRS_TLS __declspec(thread)
#define FRS_TIMING_EXPORT __declspec(dllexport)
#else
#define FRS_TLS _Thread_local
#define FRS_TIMING_EXPORT __attribute__((visibility("default")))
#endif
static FRS_TLS int frs_acados_enabled=0;
static FRS_TLS unsigned frs_acados_calls=0;
static FRS_TLS double frs_acados_seconds=0.;
FRS_TIMING_EXPORT void frs_acados_solve_timer_reset_v3(void){
    frs_acados_enabled=1;frs_acados_calls=0;frs_acados_seconds=0.;
}
FRS_TIMING_EXPORT double frs_acados_solve_timer_read_v3(void){
    frs_acados_enabled=0;
    return frs_acados_calls?frs_acados_seconds:NAN;
}
static void frs_acados_record(acados_timer* timer){
    if(frs_acados_enabled){frs_acados_seconds+=acados_toc(timer);++frs_acados_calls;}
}
#undef FRS_TLS
#undef FRS_TIMING_EXPORT
