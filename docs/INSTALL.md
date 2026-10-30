# Installation

## Python and offline checks

Use Python 3.10+ (local preparation checks used Python 3.13.12; CI is configured for Python 3.11). From the repository root:

```bash
python -m pip install -e .
python tools/verify_artifact.py
python tools/check_environment.py
```

`pip install -e ".[experiments]"` installs plotting and Python experiment dependencies. `requirements.txt` retains the archive's broader dependency list. Native solvers and ROS are separate installations.

## Core native solver

Install CMake >=3.21, a C++17 compiler, OpenMP and BLASFEO. Obtain the FATROP source at the exact patch-base commit, rather than applying the patch to an arbitrary latest revision:

```bash
git clone https://github.com/meco-group/fatrop.git external/fatrop
git -C external/fatrop checkout 5951e14989175108e0a64c0f726af8fbe19563f3
git -C external/fatrop apply --check ../../rapidoc/patches/fatrop_cache_export.patch
git -C external/fatrop apply ../../rapidoc/patches/fatrop_cache_export.patch
```

Build and install that patched FATROP checkout using its own build instructions and a compatible BLASFEO build. Then configure RapidDOC, replacing the paths below with your actual source and install prefixes:

```bash
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Release -DFRS_FATROP_SOURCE_DIR=/path/to/patched/fatrop -DCMAKE_PREFIX_PATH="/path/to/fatrop/install;/path/to/blasfeo/install"
cmake --build build/core --config Release --parallel
```

The shared library is written to `build/bin/`. On Windows, run CMake from an x64 MSVC environment; the retained Release configuration uses AVX2. On Linux/macOS, make the dependency libraries discoverable by the dynamic loader. Set `FRS_SOLVER_DLL` to an explicit library path if needed; despite its historical name, the environment variable accepts `.dll`, `.so` or `.dylib`.

This CMake target compiles selected patched FATROP implementation files and links `fatrop::fatrop`. The source headers, installed FATROP and BLASFEO must match. An unrelated local FATROP build is not a certified substitute.

## Imitation-learning native interface

```bash
cmake -S experiments/exp2_ocp_imitation/native -B build/imitation -DCMAKE_BUILD_TYPE=Release -DFRS_FATROP_SOURCE_DIR=/path/to/patched/fatrop -DCMAKE_PREFIX_PATH="/path/to/fatrop/install;/path/to/blasfeo/install"
cmake --build build/imitation --config Release --parallel
```

Set `IMITATION_NATIVE_DLL` to the resulting `imitation_native` library. The retained experiment defaults target a Windows DLL, so non-Windows users should set the path explicitly. Instrumented CasADi/IPOPT, acados and Safe-PDP are configured separately as described in `experiments/exp2_ocp_imitation/EXPERIMENT.md`. Full paper timing reproduction requires the original numerical-kernel hooks and a manifest matching the actual build; ordinary package wheels do not provide these hooks.

## Vehicle

Vehicle training and ROS1 target Linux; see `experiments/exp3_vehicle/README.md` for native paths, topics, parameter files and the model ID. Offline plotting does not require vehicle hardware. This preparation does not execute a physical controller.
