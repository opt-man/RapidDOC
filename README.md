# RapidDOC

RapidDOC provides Riccati-preconditioned implicit differentiation for optimal-control problems solved with FATROP. It reuses the forward factorization to initialize the adjoint and to precondition a GMRES correction whenever the target residual exceeds the requested tolerance.

The repository contains the native solver, Python binding, FATROP cache integration, learning experiments, and vehicle deployment code. Install the Python distribution as `rapiddoc` and import it as `rapidoc`.

**Riccati-Preconditioned Implicit Differentiation for Differentiable Optimal Control under Operator Mismatch**<br>
Xiangda Zhu, Yuankun Chen, Zifei Nie*, and Yi Chang*  
*Corresponding authors. Contact: [zifei_nie@jlu.edu.cn](mailto:zifei_nie@jlu.edu.cn).

[Installation](#installation) · [Usage](#usage-examples) · [Experiments](#experiment-reproduction) · [Parameters](#solver-parameters) · [Video](https://youtu.be/b53gcrjdS7c)

<p align="center">
  <img src="assets/figures/fig1_framework.png" width="540" alt="RapidDOC forward-factor reuse and target-adjoint correction framework">
</p>

## Project Summary

RapidDOC differentiates an outer loss through an optimal-control solution. The forward solver supplies the solution and a retained Riccati factorization. The backward computation uses the prescribed target adjoint equation

$$
K_B^\top a^\star=g.
$$

The retained cache first supplies an adjoint candidate. When that candidate is insufficient, a fixed right-preconditioned GMRES solve corrects it against the target equation. The resulting adjoint is used to compute the parameter gradient. The target operator can differ from the operator associated with the forward cache.

Available components:

- C++ factor-reuse solver with a C interface for live cached-solve callbacks and saved factors.
- Python access through a lightweight `ctypes` binding.
- FATROP patch for retaining and accessing forward-factor information.
- Frozen-case adjoint examples and backward-solve comparisons.
- Open- and closed-loop imitation learning on cartpole, quadrotor and robot arm.
- Native vehicle training, a ROS1 controller, recorded runs and a demonstration video.

<details>
<summary>Backward equations and residual acceptance</summary>

The prescribed target adjoint is

$$
K_B^\top a^\star=g.
$$

The forward Riccati cache supplies the fixed solve action

$$
\mathcal S_F(v)
:=
\mathrm{CachedSolve}(\mathcal C_F,v).
$$

For a nonzero loss seed $g$, the cached candidate and its target-residual test are

$$
a_0:=\mathcal S_F(g).
$$

$$
r_0:=g-K_B^\top a_0,
\qquad
\rho_0:=\frac{\|r_0\|_2}{\|g\|_2}.
$$

If $\rho_0>\varepsilon$, GMRES solves the right-preconditioned correction equation

$$
K_B^\top\mathcal S_F(y)=r_0,
\qquad
a=a_0+\mathcal S_F(y).
$$

The parameter gradient is then obtained from the target adjoint using

$$
\frac{\mathrm d\mathcal L}{\mathrm d\theta}
=
\nabla_\theta\mathcal L-G_B^\top a^\star.
$$

The target operator $K_B$ and cached solve action $\mathcal S_F$ remain fixed during the backward solve. Every returned candidate is checked against the prescribed target equation.


</details>

## File Structure

```text
rapidoc/
  factor_reuse_solver.cpp        native factor-reuse and GMRES implementation
  factor_reuse_solver.h          C interface and statistics
  solver.py                      Python ctypes binding
  patches/                       FATROP cache-export patch
experiments/
  exp1_factor_reuse_ablation/     cached solve / target refactor / no-reuse GMRES
  exp2_ocp_imitation/             open- and closed-loop OCP parameter learning
  exp3_vehicle/                  vehicle training, ROS1 controller and recordings
tutorials/                       saved-case adjoint example and learning path
examples/                        data inspection without a native build
tests/                           offline integrity and operator checks
tools/                           data verifier and environment diagnostics
docs/                            installation, API and reproduction details
assets/                          original paper figures and demonstration video
```

## Installation

The native backend requires CMake >=3.21, a C++17 compiler, OpenMP, BLASFEO and a compatible FATROP source/install pair. The Python package and the native library are installed separately.

Check that CMake is available:

```bash
cmake --version
```

On Linux, also check the compiler:

```bash
c++ --version
```

On Windows, use Visual Studio Build Tools with the C++ workload and an x64 build configuration. The local native checks used MSVC 2022.

### Python Interface

Requirements:

- Python 3.10 or newer.
- NumPy and SciPy, installed with the package.
- For experiments: CasADi 3.8.0, Matplotlib and threadpoolctl.

Download the repository and install:

```bash
git clone https://github.com/opt-man/RapidDOC.git
cd RapidDOC
python -m pip install -e .
```

Verify the package:

```bash
python -c "import rapidoc; print(rapidoc.__file__); print(rapidoc.__version__)"
python -m pip show rapiddoc
```

For plotting and experiment entry points:

```bash
python -m pip install -e ".[experiments]"
```

Importing the package and inspecting saved data do not load the native library. Actual adjoint solves require the build below.

### Native Backend

Use FATROP 1.1.7 at patch-base commit `5951e14989175108e0a64c0f726af8fbe19563f3`. Apply the supplied cache-export patch, then build and install FATROP with compatible BLASFEO dependencies. The exact dependency commands are in [INSTALL.md](docs/INSTALL.md).

Configure RapidDOC with the patched source and install prefixes:

```bash
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Release -DFRS_FATROP_SOURCE_DIR=/path/to/patched/fatrop -DCMAKE_PREFIX_PATH="/path/to/fatrop/install;/path/to/blasfeo/install"
cmake --build build/core --config Release --parallel
```

The shared library is placed in `build/bin/`. Set `FRS_SOLVER_DLL` if it is located elsewhere. Its dependent native libraries must also be discoverable. The imitation-learning backend is a separate target; its build and runtime settings are documented in [INSTALL.md](docs/INSTALL.md) and the [OCP experiment guide](experiments/exp2_ocp_imitation/EXPERIMENT.md).

## Usage Examples

Start with the supplied examples:

```text
examples/inspect_frozen_case.py   inspect dimensions and retained factor arrays
tutorials/frozen_adjoint.py       solve one saved target adjoint with cached factors
tutorials/README.md               progression from saved cases to learning and deployment
```

Inspect a case without native dependencies:

```bash
python examples/inspect_frozen_case.py
```

After building the native backend, run a cached adjoint solve:

```bash
python tutorials/frozen_adjoint.py --problem cartpole --seed 0
```

The script loads one saved forward cache and target system, solves the adjoint, and independently checks its target residual. It reports correction iterations and cached factor applications. For a tighter residual requirement:

```bash
python tutorials/frozen_adjoint.py --problem cartpole --seed 0 --tolerance 1e-10
```

These examples operate on frozen post-forward data. The learning experiments below include forward optimization and parameter updates. Live integrations can borrow the forward solver's cached-solve callback directly; see [API.md](docs/API.md) for ownership and lifetime requirements.

## Experiment Reproduction

Experiments and recorded outputs are organized under `experiments/`. Run the following commands from the repository root. Keep new runs in separate output directories. `Ours` is the historical method label for RapidDOC.

### Paper-to-code Map

| Paper item | Entry point | Recorded output |
|---|---|---|
| Fig. 1: method overview | Method summary above | [Original framework figure](assets/figures/fig1_framework.png) |
| Fig. 2: factor-reuse ablation | [experiment.py](experiments/exp1_factor_reuse_ablation/experiment.py), [plot.py](experiments/exp1_factor_reuse_ablation/plot.py) | [Summary CSV](experiments/exp1_factor_reuse_ablation/results/gmres_real_ocp_ablation_summary.csv) |
| Fig. 3: open-loop learning | [imitation_learning.py](experiments/exp2_ocp_imitation/imitation_learning.py), `run` / `plot` | [Learning curves](experiments/exp2_ocp_imitation/figures/fig3_open_loop_imitation_learning.pdf) |
| Fig. 4 and Table I: closed-loop learning | Same script, `closed-loop` / `plot-closed-loop` | [Closed-loop curves](experiments/exp2_ocp_imitation/figures/fig4_closed_loop_imitation_learning.pdf), [records](experiments/exp2_ocp_imitation/results/closed_loop) |
| Fig. 5: vehicle learning | [Training sources](experiments/exp3_vehicle/training), [plot_learning.py](experiments/exp3_vehicle/plot_learning.py) | [Selection and RMSE metadata](experiments/exp3_vehicle/results/figure8_learning_metadata.json) |
| Fig. 6 and Table II: physical vehicle runs | [ROS1 controller](experiments/exp3_vehicle/ros), [experiment guide](experiments/exp3_vehicle/README.md) | [Recorded runs](experiments/exp3_vehicle/data), [video](https://youtu.be/b53gcrjdS7c) |

### Exp. 1: Factor-Reuse Ablation

There are 32 frozen cases per task for cartpole, quadrotor and robot arm. The three methods solve the same target system at a common residual tolerance: RapidDOC, a fresh target factorization, and unpreconditioned GMRES.

Run one case with all three methods:

```bash
python experiments/exp1_factor_reuse_ablation/experiment.py --problems cartpole --instances 1 --repeats 1 --output-dir runs/ablation-smoke
```

Use `--instances 32` and omit `--problems` to include all 96 cases. Native solves require the core library and CasADi/MUMPS for the independent reference.

Redraw the recorded results:

```bash
python experiments/exp1_factor_reuse_ablation/plot.py
```

![Factor-reuse ablation from the paper](assets/figures/fig2_factor_reuse_ablation.png)

### Exp. 2: OCP Imitation Learning

This experiment learns OCP parameters from demonstrations on three tasks. Prepare the imitation native library and check backend availability:

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py check-backends --methods Ours
```

Open-loop learning:

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py run --methods Ours --quick --output-dir runs/open-loop-smoke
```

Closed-loop learning:

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py closed-loop --methods Ours --quick --output-dir runs/closed-loop-smoke
```

The `--quick` settings shorten the run for setup checks. For the full paper protocol, dependency instrumentation and external baselines, follow [EXPERIMENT.md](experiments/exp2_ocp_imitation/EXPERIMENT.md).

Redraw the archived open- and closed-loop results:

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py plot
python experiments/exp2_ocp_imitation/imitation_learning.py plot-closed-loop
```

![Open-loop imitation learning from the paper](assets/figures/fig3_open_loop_imitation_learning.png)

![Closed-loop imitation learning from the paper](assets/figures/fig4_closed_loop_imitation_learning.png)

### Exp. 3: Vehicle Learning and Deployment

The vehicle experiment includes native closed-loop training and a matching ROS1 controller. The recorded data include learning histories and physical vehicle runs. Training and deployment target Linux; use the [vehicle README](experiments/exp3_vehicle/README.md) for build commands, model identity, controller topics and parameter files.

Reproduce the learning plot without vehicle hardware:

```bash
python experiments/exp3_vehicle/plot_learning.py
```

![Vehicle learning from the paper](assets/figures/fig5_figure8_learning.png)

Click the physical-experiment figure to watch the demonstration:

[![Physical vehicle runs from the paper](assets/figures/fig6_vehicle_before_after.png)](https://youtu.be/b53gcrjdS7c)

[YouTube video](https://youtu.be/b53gcrjdS7c) · [Download MP4](assets/RapidDOC_demo.mp4)

All displayed figures use the paper's original figure assets. [Figure provenance](assets/figures/provenance.json) records their source hashes. The full manuscript and LaTeX sources are excluded from this repository.

## Solver Parameters

### Problem Data

The saved-case helper uses the following data. The target blocks define the equation to solve; the forward blocks and retained factors define the reusable cache.

| Field | Meaning |
|---|---|
| `horizon`, `nx`, `nu` | OCP horizon, state dimension and control dimension |
| `target_hessian`, `target_terminal_hessian` | Target stage and terminal Hessian blocks |
| `target_dynamics` | Target dynamics-Jacobian blocks |
| `forward_hessian`, `forward_terminal_hessian`, `forward_dynamics` | Blocks associated with the saved forward solve |
| `value_hessians`, `control_cholesky`, `cross_factors` | Retained Riccati factor arrays |
| `rhs` | Target-system right-hand side |

The [frozen-case loader](experiments/exp1_factor_reuse_ablation/frozen_cases.py) defines array layouts. The [C header](rapidoc/factor_reuse_solver.h) specifies native memory layouts and batched interfaces.

### Cache Construction

| Entry point | Use |
|---|---|
| `frs_create_from_cached_solve` | Borrow the live forward solver's factor-solve callback |
| `frs_create_from_factors` | Import saved Riccati factors, as in the standalone tutorial |
| `frs_create` | Build fresh factors for reference comparisons |
| `frs_clone` | Share immutable factors with a private solve workspace |
| `frs_destroy` | Release a handle |

The callback context must remain valid for the handle's lifetime. The factor action stays fixed during one adjoint solve.

### Backward Settings

The standalone tutorial exposes:

| Option | Meaning | Default |
|---|---|---|
| `--problem` | `cartpole`, `quadrotor` or `robotarm` | `cartpole` |
| `--seed` | Saved case index, 0 through 31 | `0` |
| `--tolerance` | Required relative target-KKT residual | `1e-6` |
| `--max-iterations` | Maximum correction iterations | KKT dimension |

The ablation additionally accepts `--methods`, `--instances`, `--repeats`, `--cases-dir` and `--output-dir`. Its `--max-iterations 0` selects the KKT dimension; the standalone tutorial instead requires a positive explicit iteration limit.

### Learning Settings

These are the imitation-learning script's full-run defaults before applying `--quick`:

| Option | Meaning | Default |
|---|---|---|
| `--forward-tolerance` | Forward solver tolerance | `1e-8` |
| `--forward-max-iterations` | Forward iteration limit | `500` |
| `--tolerance` | Backward residual tolerance | `1e-6` |
| `--max-iterations` | Backward correction iteration limit | `200` |
| `--output-dir` | Destination for a new run | Command-dependent |

Use `python experiments/exp2_ocp_imitation/imitation_learning.py --help` for model, dataset and training options.

### Solve Calls and Results

The standalone example uses the retained Experiment 1 helper:

```python
with NativeFactor(case, forward=True) as factors:
    solution, stats, call_seconds = factors.solve(
        target, case.rhs, tolerance=1e-6,
        max_iterations=case.total_dimension,
    )
```

Here `case` is a loaded frozen case and `target` is the tuple `(case.target_hessian, case.target_terminal_hessian, case.target_dynamics)`. Imports and the independent residual check are shown in the complete [tutorial](tutorials/frozen_adjoint.py).

| Result | Meaning |
|---|---|
| `solution` | Target adjoint vector |
| `stats.relative_residual` | Native relative residual diagnostic |
| `stats.mean_iterations` | Mean GMRES correction iterations |
| `stats.mean_cached_backsolves` | Mean cached factor applications |
| `stats.used_fallback` | Fresh-factor fallback indicator; the saved-case helper rejects unexpected fallback |
| `call_seconds` | Python-observed helper solve-call duration |

Additional native fields report residual evaluation, factor sweeps and orthogonalization. For paper timing comparisons, follow the recorded manifests and the [API timing notes](docs/API.md).

### Verification

```bash
python tools/verify_artifact.py
python -m unittest discover -s tests -v
python tools/check_environment.py
```

[VALIDATION.md](VALIDATION.md) records checks performed on the supplied artifact. [THIRD_PARTY.md](THIRD_PARTY.md) lists external dependencies, and [CONTRIBUTING.md](CONTRIBUTING.md) describes contribution checks.
