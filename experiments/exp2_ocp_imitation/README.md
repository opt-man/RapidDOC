# Experiment 2: OCP imitation learning

This experiment learns OCP parameters from open- and closed-loop demonstrations on cartpole, quadrotor and robot-arm problems.

The complete installation requirements, run commands, paper defaults, instrumented baselines and timing policy are in [EXPERIMENT.md](EXPERIMENT.md).

- Entry point: [imitation_learning.py](imitation_learning.py)
- Frozen demonstrations: [data/](data)
- Recorded learning results: [results/](results)
- Paper figures: [figures/](figures)
- Native integration: [native/](native)
- Dependency instrumentation: [patches/](patches)

From the repository root, start with:

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py --help
python experiments/exp2_ocp_imitation/imitation_learning.py check-backends --methods Ours
```

Use a separate output directory for new experiments. `Ours` is the stored method label for RapidDOC.
