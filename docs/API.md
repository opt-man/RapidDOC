# Native API and integration

The public C declarations are in [factor_reuse_solver.h](../rapidoc/factor_reuse_solver.h). The lightweight Python binding is [solver.py](../rapidoc/solver.py). It loads the native library lazily. The Python package retains the import name `rapidoc`, while its distribution name is `rapiddoc`.

## Factor ownership and constructors

| C entry point | Role |
|---|---|
| `frs_create_from_cached_solve` | Live integration: borrow a forward-factor solve callback and its context |
| `frs_create_from_factors` | Standalone saved-case integration: import exported value, control and cross factors |
| `frs_create` | Fresh factorization helper for reference comparisons |
| `frs_clone` | Share immutable factors with a private solve workspace |
| `frs_destroy` | Release a handle |

For the live path, the forward solver and callback context must outlive the RapidDOC handle. Preserve the numerical factors during one adjoint solve. A clone has a private workspace; this does not certify arbitrary external callbacks as thread-safe.

The frozen-case tutorial uses `frs_create_from_factors` through the existing experiment helper. The live imitation and vehicle integrations can avoid exporting/importing saved arrays by borrowing a cached-solve action.

## Target solves

`frs_solve` receives a handle, target Hessian and dynamics blocks, right-hand sides, tolerance and iteration controls. It uses the bound factor solve as a fixed right preconditioner. Consult the header for the exact argument ordering, matrix layouts, batched solve modes and certification flags before calling it directly.

The tutorial calls the retained `NativeFactor.solve(target, rhs, tolerance, max_iterations)` helper. It verifies the returned solution with an independent Python target-operator action. This helper and its frozen-case data model are experimental utilities, rather than a general symbolic OCP modeling interface.

## Statistics

`FrsStats` contains, among others:

| Field | Meaning |
|---|---|
| `relative_residual` | Native relative residual diagnostic |
| `mean_iterations` | Mean correction iterations over the solved right-hand sides |
| `mean_cached_backsolves` | Mean cached factor applications |
| `arnoldi_seconds` | Arnoldi/orthogonalization timing |
| `residual_seconds` | Target residual evaluation timing |
| `setup_seconds`, `workspace_seconds` | Setup and workspace diagnostics |
| `backward_seconds`, `forward_seconds` | Internal Riccati sweep timings, not end-to-end differentiation/optimization times |
| `used_fallback` | Fresh-factor fallback indicator checked by the saved-case helper |

Timing fields belong to the native implementation's boundaries. For paper comparisons, use the experiment's recorded timing manifest; summing all nested diagnostic fields can double-count work.

## Errors and runtime location

Nonzero C statuses should be decoded with `frs_last_error`. The message is valid until the next API call on the same thread. The Python binding raises errors using that message.

`FRS_SOLVER_DLL` selects the core shared-library location; the name is historical and also accepts Linux/macOS library paths. The imitation backend has its separate `IMITATION_NATIVE_DLL` setting. See [Installation](INSTALL.md) for dependency discovery.
