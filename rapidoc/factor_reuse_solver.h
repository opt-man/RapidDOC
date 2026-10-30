#ifndef FACTOR_REUSE_SOLVER_H
#define FACTOR_REUSE_SOLVER_H

#ifdef _WIN32
#ifdef FRS_BUILD
#define FRS_API __declspec(dllexport)
#else
#define FRS_API __declspec(dllimport)
#endif
#else
#define FRS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { FRS_MAX_RESIDUAL_HISTORY = 256 };

typedef struct FrsStats
{
    double elapsed_seconds;
    double mean_iterations;
    double relative_residual;
    int used_fallback;
    double rhs_pack_seconds;
    double backward_seconds;
    double trsm_seconds;
    double forward_seconds;
    double output_unpack_seconds;
    double arnoldi_seconds;
    double residual_seconds;
    double residual_pack_seconds;
    double residual_hx_seconds;
    double residual_at_lambda_seconds;
    double residual_ax_seconds;
    double residual_unpack_seconds;
    double residual_subtract_norm_seconds;
    double backward_gemm_seconds;
    double backward_copy_seconds;
    double backward_scaled_copy_seconds;
    double backward_trsm_kernel_seconds;
    int backward_gemm_calls;
    int backward_copy_calls;
    int backward_scaled_copy_calls;
    int backward_trsm_calls;
    double forward_gemm_seconds;
    double forward_copy_seconds;
    double forward_scaled_copy_seconds;
    double forward_trsm_kernel_seconds;
    int forward_gemm_calls;
    int forward_copy_calls;
    int forward_scaled_copy_calls;
    int forward_trsm_calls;
    double setup_seconds;
    double difference_matvec_seconds;
    double projection_seconds;
    double input_copy_seconds;
    double workspace_seconds;
    double control_seconds;
    double final_update_seconds;
    double mean_cached_backsolves;
    int residual_history_count;
    double residual_history[FRS_MAX_RESIDUAL_HISTORY];
    double target_build_seconds;
    double sensitivity_linearization_seconds;
} FrsStats;

typedef void *FrsHandle;

/* Native cached solve callback. Arrays are row-major matrices with shape
   (reduced KKT dimension, nrhs). Returning zero reports success. */
typedef int (*FrsCachedSolve)(void *context, int nrhs,
                              const double *rhs, double *solution);

/* The returned message is valid until the next API call on the same thread. */
FRS_API const char *frs_last_error(void);

/* Fresh factorization helper for reference baselines only. Integrated
   factor-reuse callers should use frs_create_from_cached_solve(); the exported
   factor constructor below is retained for standalone files and regression
   tests. */
FRS_API int frs_create(
    int horizon,
    int nx,
    int nu,
    const double *h_forward,
    const double *h_forward_terminal,
    const double *a_blocks,
    FrsHandle *handle);

/* Numerical target assembly/factorization performed by frs_create(), excluding
   one-time solver-object and workspace construction. This supports the same
   timing boundary as acados setup_qp_matrices_and_factorize(). */
FRS_API int frs_get_fresh_factorization_seconds(
    FrsHandle handle,
    double *seconds);

/* Create the reuse solver from the numerical factorization F_F exported by
   the matching FATROP forward solve. h_forward and a_blocks describe the
   forward reduced KKT system associated with F_F. The returned handle applies
   P_F^{-1} v := Solve(F_F, v); it does not form P_F, equate P_F with K_F, or
   refactor the forward matrix. */
FRS_API int frs_create_from_factors(
    int horizon,
    int nx,
    int nu,
    const double *h_forward,
    const double *h_forward_terminal,
    const double *a_blocks,
    const double *value_hessians,
    const double *control_cholesky,
    const double *cross_factors,
    FrsHandle *handle);

/* Integrated path: borrow the factorization owned by the forward solver and
   apply it through cached_solve. No factor export, import, packing, or
   refactorization is performed. context must outlive the returned handle. */
FRS_API int frs_create_from_cached_solve(
    int horizon,
    int nx,
    int nu,
    FrsCachedSolve cached_solve,
    void *context,
    FrsHandle *handle);

/* Share immutable F_F while creating a private solve workspace. */
FRS_API int frs_clone(FrsHandle source, FrsHandle *clone);

FRS_API void frs_destroy(FrsHandle handle);

/* Solve the target KKT system with right-preconditioned batched Global GMRES.
   The preconditioner is the FATROP forward factorization bound to the handle.
   The target operator is exact; optional explicit columnwise residual
   certification checks the returned solution against that target. */
FRS_API int frs_solve(
    FrsHandle handle,
    int nrhs,
    const double *h_target,
    const double *h_target_terminal,
    const double *a_target,
    const double *rhs,
    double tolerance,
    int max_iterations,
    int chunk_size,
    int recycle_rank,
    int certify_residual,
    double *solution,
    FrsStats *stats);

/* Ablation-only solver: solve the same target system from the zero initial
   guess with plain (unpreconditioned) GMRES. The handle supplies dimensions
   and private workspaces only; its cached forward factorization is never
   applied. This entry point is intentionally separate from frs_solve so the
   production factor-reuse path is unchanged. */
FRS_API int frs_solve_unpreconditioned(
    FrsHandle handle,
    int nrhs,
    const double *h_target,
    const double *h_target_terminal,
    const double *a_target,
    const double *rhs,
    double tolerance,
    int max_iterations,
    int certify_residual,
    double *solution,
    FrsStats *stats);

/* One-RHS ablation timing entry point. The existing algorithms and mandatory
   true-residual certification are unchanged. stats->elapsed_seconds excludes
   target-operator construction, workspace preparation and API boundary copies.
   call_seconds retains the inclusive duration of the same measured interval.
   Numerical packing, cached initialization, preconditioning, Krylov operations
   and residual evaluation remain timed. Other entry points keep their timing. */
FRS_API int frs_solve_ablation(
    FrsHandle handle,
    const double *h_target,
    const double *h_target_terminal,
    const double *a_target,
    const double *rhs,
    double tolerance,
    int max_iterations,
    int reuse_forward_factors,
    double *solution,
    FrsStats *stats,
    double *call_seconds);

/* Solve one adjoint RHS for each independent OCP in a single native call.
   Every batch entry owns a distinct forward-factor handle, while all entries
   share the same dimensions and use contiguous batch-major numerical arrays.
   Independent entries execute through the caller-configured OpenMP worker
   pool; each individual OCP and its cached factor workspace remain private. */
FRS_API int frs_solve_adjoint_batch(
    FrsHandle *handles,
    int batch_size,
    const double *h_target_batch,
    const double *h_target_terminal_batch,
    const double *a_target_batch,
    const double *rhs_batch,
    double tolerance,
    int max_iterations,
    int certify_residual,
    double *solution_batch,
    FrsStats *stats_batch);

/* Apply exactly the right-preconditioned operator used by GMRES:
   output = K_B P_F^{-1} input. This entry point is diagnostic only and is
   excluded from all formal timing measurements. */
FRS_API int frs_apply_preconditioned_operator(
    FrsHandle handle,
    int nrhs,
    const double *h_target,
    const double *h_target_terminal,
    const double *a_target,
    const double *input,
    double *output);

/* Build the exact nonlinear-chain target system from the accepted optimum and
   immediately solve it with the cached FATROP factors. The constructed target
   arrays remain native and are consumed by frs_solve in the same call. */
FRS_API int frs_build_chain_target_and_solve(
    FrsHandle handle,
    int parameters,
    const double *optimal_primal,
    const double *dynamics_multipliers,
    const double *coupling,
    const double *nonlinearity,
    const double *state_weights,
    const double *control_weights,
    const double *state_parameters,
    const double *control_parameters,
    const double *terminal_parameters,
    const double *dynamics_parameters,
    double terminal_multiplier,
    double tolerance,
    int max_iterations,
    int chunk_size,
    int recycle_rank,
    int certify_residual,
    double *solution,
    FrsStats *stats);

/* ABI-compatible multi-handle entry point. The single-thread benchmark
   delegates to frs_solve and does not enable hidden worker parallelism. */
FRS_API int frs_solve_parallel(
    FrsHandle *handles,
    int handle_count,
    int nrhs,
    const double *h_target,
    const double *h_target_terminal,
    const double *a_target,
    const double *rhs,
    double tolerance,
    int max_iterations,
    int tile_size,
    int recycle_rank,
    int certify_residual,
    double *solution,
    FrsStats *stats);

#ifdef __cplusplus
}
#endif

#endif
