#ifndef F8_EXPERIMENT_H
#define F8_EXPERIMENT_H
#include <stddef.h>
#ifdef _WIN32
#ifdef F8_BUILD
#define F8_API __declspec(dllexport)
#else
#define F8_API __declspec(dllimport)
#endif
#else
#define F8_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

enum { F8_NX = 5, F8_NU = 2, F8_PARAMETERS = 4 };
typedef struct F8Solver F8Solver;
typedef struct {
    double forward_solve_s;
    double backward_solve_s;       /* NaN when a strict isolated timer is unavailable. */
    double backward_native_s;      /* Full native backward, excluding object creation. */
    double target_build_s;
    double initial_relative_residual;
    double final_relative_residual;
    int forward_iterations;
    int backward_iterations;
    int active_constraints;
    int status;
} F8Stats;

F8_API const char *f8_backend(void);
F8_API const char *f8_backend_details(void);
F8_API const char *f8_model_id(void);
F8_API const char *f8_last_error(void);
F8_API int f8_horizon(void);
F8_API int f8_trajectory_size(void);
F8_API double f8_dt(void);
F8_API F8Solver *f8_create(const char *assets_directory);
F8_API void f8_destroy(F8Solver *solver);
/* External trajectory ordering: x0,u0,x1,u1,...,xN (5 states, 2 controls). */
F8_API int f8_forward(F8Solver *, const double theta[4], const double initial[5],
                     const double *guess, double *trajectory, F8Stats *);
/* seed is d(outer_loss)/d(trajectory); theta occurs in the inner cost only. */
F8_API int f8_backward(F8Solver *, const double *seed, double gradient[4], F8Stats *);
/* Diagnostic names/pointers survive until the next forward/backward call. */
F8_API int f8_trace_count(F8Solver *);
F8_API const char *f8_trace_name(F8Solver *, int index);
F8_API const double *f8_trace_values(F8Solver *, int index, size_t *count);
F8_API void f8_reference(double phase, double state[5], double control[2]);
F8_API void f8_dynamics(const double state[5], const double control[2], double next[5]);
F8_API void f8_initial_guess(const double initial[5], double *guess);
F8_API void f8_shift_guess(const double *trajectory, const double initial[5], double *guess);
F8_API void f8_initial_theta(double theta[4]);
F8_API void f8_teacher_theta(double theta[4]);
F8_API double f8_imitation_loss(const double *trajectory, const double *teacher, double *seed);

#ifdef __cplusplus
}
#endif
#endif
