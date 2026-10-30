# Tutorials

Run these commands from the repository root.

## 1. Inspect an optimal-control case

```bash
python -m pip install -e .
python examples/inspect_frozen_case.py
```

This reads one saved problem and prints its dimensions and factor shapes. It needs NumPy, without loading native solvers. The saved target and forward-factor data come from the paper artifact.

## 2. Solve a target adjoint with cached factors

Build the native core using [Installation](../docs/INSTALL.md). Then run:

```bash
python tutorials/frozen_adjoint.py --problem cartpole --seed 0
python tutorials/frozen_adjoint.py --problem quadrotor --seed 0
python tutorials/frozen_adjoint.py --problem robotarm --seed 0
```

The example imports saved factors, obtains a cached candidate and applies GMRES correction when required. It independently checks the final target residual and exits with an error if the requested tolerance is not reached. These are backward solves on frozen targets; the example does not rerun forward optimization or a learning loop.

`NativeFactor` is the existing Experiment 1 helper. The low-level Python binding is `rapidoc.solver`. The [API guide](../docs/API.md) explains the separate live-solver integration path, where factors are borrowed directly from the forward solver.

## 3. Compare backward methods

Install the experiment dependencies and follow [Experiment 1](../docs/REPRODUCIBILITY.md). Use the same inputs and residual tolerance for the three branches. Send new outputs to `runs/`.

## 4. Run a learning example

Follow the [imitation experiment setup](../experiments/exp2_ocp_imitation/EXPERIMENT.md), then use its `check-backends` command before a `--quick` run. That path includes the forward optimization and parameter-learning loop. It requires the separately built imitation backend.

## 5. Vehicle deployment

The [vehicle guide](../experiments/exp3_vehicle/README.md) describes Linux training and ROS1 deployment. Offline plot reproduction is available without robot hardware.
