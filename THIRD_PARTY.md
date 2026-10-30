# External dependencies and provenance

The repository does not vendor installed solvers, compiler runtimes or native binaries.

| Component | Reference / version | Local integration |
|---|---|---|
| FATROP | https://github.com/meco-group/fatrop , 1.1.7; patch base `5951e14989175108e0a64c0f726af8fbe19563f3` | `rapidoc/patches/fatrop_cache_export.patch` |
| Safe-PDP | https://github.com/wanxinjin/Safe-PDP , `de13f09c1fafe19e8d3aaf35615993efc8838814` | `experiments/exp2_ocp_imitation/patches/`; upstream license retained there |
| IPOPT | 3.14.19, https://github.com/coin-or/Ipopt | Kernel timing patch and upstream license in the imitation experiment |
| acados | 0.5.1, https://github.com/acados/acados | Instrumentation sources under the experiments |
| CasADi | 3.8.0, https://github.com/casadi/casadi | Python and optional native baseline |
| BLASFEO / HPIPM | Required through native solver builds | Use versions matched to those external builds |

Generated model code and recorded result manifests are retained from the original artifact. Historical local paths inside frozen metadata describe the original run, not required installation paths. No permissive license has been assigned to third-party content by this preparation. Existing upstream notices remain alongside patches.
