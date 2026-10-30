# OCP imitation learning

`imitation_learning.py` is the implementation and command-line entry point.
`data/` contains the six frozen demonstration files; `results/open_loop/` and
`results/closed_loop/` contain the final seed-2 histories, checkpoints and policy
outputs. `figures/` contains the exact final Fig. 3 and Fig. 4 exports.

From this directory, redraw the archived results with:

```text
python imitation_learning.py plot
python imitation_learning.py plot-closed-loop
```

Both commands write to `regenerated/`; `--figure-dir` selects another destination.
For a different result tree, supply `--output-dir`. Plotting never rewrites the
archived summaries or histories.

After configuring the external runtimes below, the complete paper commands are:

```text
python imitation_learning.py run
python imitation_learning.py closed-loop
```

Defaults select RapidDOC (`Ours` in stored records), SafePDP, CasADi and acados;
seed 2; horizon 40; step 0.05; 32/16 training/validation demonstrations; 800
open-loop epochs at learning rate 0.002; and 500 closed-loop epochs, 50 rollout
steps, at learning rate 0.005. New runs write to `rerun_open_loop/` and
`rerun_closed_loop/`. The frozen directories reject overwrite destinations.

The final native timings require the custom `solve-kernels-v4` instrumentation.
Plain upstream wheels or libraries do not supply those hooks. No native
libraries, install trees, compiler packages or generated solver caches are
bundled here. Configure:

| Setting | External dependency |
| --- | --- |
| `IMITATION_NATIVE_DLL` | Built `imitation_native` library; its FATROP and factor-reuse dependencies must be beside it. Default: archive `build/bin/imitation_native.dll`. |
| `IMITATION_IPOPT_V4_ROOT` | Instrumented CasADi/IPOPT install containing `build_manifest.json` and `bin/`. |
| `IMITATION_IPOPT_SOURCE_ROOT` | Source/build-input root whose relative files match that manifest's `source_sha256` entries. All source, binary and numerical-dependency hashes remain mandatory. |
| `IMITATION_SAFEPDP_ROOT` | External Safe-PDP checkout with `patches/safepdp_imitation.patch` applied. |
| `IMITATION_ACADOS_ROOT` | Instrumented acados install; `--acados-root` also works. |
| `IMITATION_ACADOS_TEMPLATE_DIR` | Matching directory containing the `acados_template` Python package. |

`--acados-cache-dir` selects an external solver-build cache; use a path without
spaces. IPOPT's supplied native timing integration currently targets Windows
x64 with CasADi 3.8.0 and IPOPT 3.14.19. The paper acados build used 0.5.1,
GENERIC BLASFEO/HPIPM and OpenMP disabled. Library loading verifies the actual
timing hooks; it does not substitute wall time if instrumentation is absent.

The SafePDP patch applies to upstream commit
`de13f09c1fafe19e8d3aaf35615993efc8838814` from
<https://github.com/wanxinjin/Safe-PDP>. It preserves the paper's supplied primal
guess, persistent parametric solver, rank/constrained-policy algebra fixes and
segmented forward/backward timers. `ipopt_kernel_timing.patch` applies to IPOPT
3.14.19 and retains the local numerical-kernel timing changes. Corresponding
upstream licenses are beside the patches.

`patches/timing/` contains the small custom C/C++ timer sources and CMake source
instrumentation, not an installed dependency. For native reconstruction, stage
those files as `cmake/` in an external working source root. The retained CMake
inputs expect `third_party/ipopt-highres/`, `.fatrop-v4/source/` and
`.acados/source/` under that root; use disposable upstream checkouts and apply
the patches there. The FATROP cache-export patch and core build are in the
archive's `rapidoc/`. A new external build must supply a manifest matching its
actual build inputs and libraries; the archived result manifests remain the
unchanged record of the original runs.
