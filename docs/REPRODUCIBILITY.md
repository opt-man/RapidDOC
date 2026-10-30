# Reproduction guide

Run commands from the repository root unless a different directory is specified. Keep new output separate from frozen records. Recorded hardware, source hashes and timing boundaries remain in the experiment metadata; they are not regenerated or relabeled by this packaging.

## Experiment 1 — factor-reuse ablation

There are 32 frozen cases for each of cartpole, quadrotor and robot arm (96 total). The three branches use a common target equation and tolerance. After building the core and installing the experiment dependencies, a small native check is:

```bash
python experiments/exp1_factor_reuse_ablation/experiment.py --problems cartpole --instances 1 --repeats 1 --output-dir runs/ablation-smoke
```

For all frozen cases, use `--instances 32` and omit `--problems`. This executes real native solves; it requires the built core and CasADi's MUMPS reference. `python experiments/exp1_factor_reuse_ablation/plot.py` redraws Fig.2 from recorded CSV data without native optimization.

## Experiment 2 — open- and closed-loop learning

Read `experiments/exp2_ocp_imitation/EXPERIMENT.md` before running. It lists the exact external runtime manifests, patches, timing hooks and defaults. Select only installed backends; `Ours` is the historical CLI identifier for RapidDOC.

```bash
python experiments/exp2_ocp_imitation/imitation_learning.py check-backends --methods Ours
python experiments/exp2_ocp_imitation/imitation_learning.py run --methods Ours --quick --output-dir runs/open-loop-smoke
python experiments/exp2_ocp_imitation/imitation_learning.py closed-loop --methods Ours --quick --output-dir runs/closed-loop-smoke
```

`--quick` changes dataset/epoch sizes and is a smoke check, NOT the paper protocol. Full paper defaults are seed2, N40, dt0.05, open-loop800 epochs/lr0.002 and closed-loop500 epochs/50 MPC steps/lr0.005. The frozen recorded histories remain under `results/open_loop` and `results/closed_loop`.

## Experiment 3 — figure-eight vehicle

```bash
python experiments/exp3_vehicle/plot_learning.py
```

This redraws Fig.5 from recorded histories into `figures/generated/`. The Linux native training command, baseline requirements and ROS1 controller setup are in the experiment README. The final video is available at https://youtu.be/b53gcrjdS7c and in `assets/RapidDOC_demo.mp4`. Raw camera footage is not included; therefore the full frame-selection/compositing pipeline requires supplying that recording, while its final figures and selection records are available.

## What the release checks establish

`tools/verify_artifact.py` verifies frozen case counts, shapes, finiteness and the release file manifest. Unit tests exercise a real frozen KKT residual application, binding imports without native libraries, and publication assets. Plot and CLI checks establish entry-point usability. These checks do not replace rerunning the full training campaigns or hardware experiments. See [VALIDATION.md](../VALIDATION.md) for checks actually executed during this preparation.
