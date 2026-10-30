# Experiment 3: figure-eight vehicle

`training/` is the native C++ full closed-loop training code from
`figure8_full/training/application`, with its generated model assets.
`ros/` is the matching `figure8_mpc_ros1` ROS1 controller. Their model ID is
`e3c18be8b2e099a224994a1249b13fcdd7631ec34d3cb7e46ac8d03d58bb8a5f`,
matching the training manifests and all five physical-run parameter files.
The experiment and controller source files are unchanged; CMake paths were
adjusted for this layout and external dependencies.

`data/` contains the recorded RapidDOC/acados learning histories, four
simulated before/after rollouts, and five complete measured vehicle runs.
`results/` contains run manifests and the Fig. 5 selection/RMSE metadata.
`figures/` contains the exact final Fig. 5 and Fig. 6 PDF/PNG/SVG exports.
Run `python plot_learning.py` to reproduce Fig. 5 in `figures/generated/`
with NumPy and Matplotlib.

Native training requires Linux, CMake, a C++17 compiler, FATROP 1.1.7 with
the factor-cache exports retained in `../../rapidoc/patches/`, BLASFEO,
and acados/HPIPM. The optional third backend uses CasADi 3.8.0 and IPOPT.
Dependencies and binaries are external. CMake accepts `fatrop_DIR`,
`casadi_DIR`, and `ACADOS_SOURCE_DIR`; disable the optional backend with
`-DF8_BUILD_CASADI=OFF`. The dependency timer hooks are required for the
recorded solve-kernel timings; without them the code reports NaN. Apply
`training/patches/instrument_acados.cmake` to the already instrumented
acados source to include the five initial-state sensitivity solves.

Build directly with `cmake -S training -B training/build` and
`cmake --build training/build`. The recorded initialization is explicit:
`training/build/bin/experiment_ours train NEW_OUT 0.1 0.01 1 0.1 500 0.008`
(use `experiment_acados` for acados). Each rollout has 707 steps;
the optimizer uses full closed-loop reverse accumulation.
For ROS1, place or link `ros/` into a catkin workspace and point
`F8_NATIVE_APPLICATION_BUILD` at the native build. Its launch file takes
`theta_file` and `log_file`; recorded parameters are beside each physical
trajectory. The input topics and unit scale are in `ros/config/controller.yaml`.

`video_composite/` retains the actual Fig. 6 composition scripts and timestamp
selections. It requires OpenCV, NumPy, Pillow, ReportLab, and the original
Times font files. From this experiment directory, run
`python video_composite/gen_vehicle_overlay.py --video RAW_VIDEO --start-marked --selection-file video_composite/connected_selection.json --draw-trajectory`,
then `python video_composite/add_end_labels.py` and
`python video_composite/export_vehicle_formats.py`. New files go to
`video_composite/generated/`. The 459 MB raw camera recording is not copied;
the submission video is retained with the paper. Video composites are not
synchronized with a particular CSV run. Historical source paths inside their
saved metadata document the original recording, not required local paths.
