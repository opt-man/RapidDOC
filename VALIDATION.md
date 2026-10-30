# Release preparation checks

Executed locally on 2026-10-30, Windows x64, Python 3.13.12, MSVC 2022. CI is configured separately for Ubuntu / Python 3.11; a hosted CI run has not yet occurred.

| Check | Result |
|---|---|
| Frozen input integrity | 96 cases checked (32 per task): dimensions, finite values, symmetric target Hessians |
| Immutable file integrity | 310 release-file SHA256 hashes verified (303 retained archive files and 7 figure assets/provenance files) |
| Python source parsing | Passed |
| Offline unit tests | 4 passed |
| Python wheel build | Passed with `pip wheel . --no-deps --no-build-isolation` |
| Experiment CLIs | Ablation and imitation `--help` passed |
| Offline plots | Factor-reuse ablation and vehicle learning plots generated |
| Native core | Release build passed from the pinned FATROP source plus the supplied cache-export patch |
| Native smoke solve | Three tasks, one frozen instance each, three methods: 9/9 reached relative target residual <= 1e-6 |
| Publication assets | Six manuscript PNG figures and local MP4 signatures checked; YouTube link present; manuscript directory excluded |

## Native build provenance

The project-homepage follow-up added `tutorials/frozen_adjoint.py`. It passed for cartpole, quadrotor and robotarm at the default 1e-6 relative-residual tolerance. An additional cartpole run at 1e-10 exercised the correction branch: one GMRES iteration, two cached factor applications, independently measured relative target residual 1.246903e-14. The tutorial and README changes leave the archived numerical implementation and recorded results unchanged. The README video cover is an unchanged copy of the paper's physical vehicle figure.

FATROP source commit: `5951e14989175108e0a64c0f726af8fbe19563f3`, exported from an existing local Git checkout, then patched with `rapidoc/patches/fatrop_cache_export.patch`. This check linked an existing local FATROP installation and compatible BLASFEO; it was not a clean-room dependency installation. Follow `docs/INSTALL.md` to build matched dependencies.

An initial attempt using a later, locally modified FATROP source tree failed to link because that tree required unrelated experimental solver symbols. The release check was repeated successfully using the pinned source. No numerical algorithm was changed to accommodate those local branches.

## Independent target-residual check

The native smoke command was:

```bash
python experiments/exp1_factor_reuse_ablation/experiment.py --problems cartpole quadrotor robotarm --instances 1 --repeats 1 --output-dir runs/ablation-smoke
```

| Task | Method | Relative target-KKT residual |
|---|---|---|
| cartpole | Ours | 3.130e-08 |
| cartpole | Target refactor | 9.537e-16 |
| cartpole | No-reuse GMRES | 4.064e-07 |
| quadrotor | Ours | 1.976e-08 |
| quadrotor | Target refactor | 2.409e-15 |
| quadrotor | No-reuse GMRES | 9.858e-07 |
| robotarm | Ours | 1.422e-08 |
| robotarm | Target refactor | 7.138e-16 |
| robotarm | No-reuse GMRES | 9.002e-07 |

The complete smoke summary is in [native_smoke_summary.csv](docs/validation/native_smoke_summary.csv). This is an entry-point and numerical-correctness check, not a statistically meaningful timing benchmark. Forward optimization was not rerun: these solves use the supplied frozen post-forward targets and caches. A CasADi/MUMPS solve supplies the independent adjoint reference. This does not constitute a fresh end-to-end finite-difference gradient test.

Full imitation-learning training, external instrumented baselines, physical ROS experiments and a fresh LaTeX rebuild were not run during this repository preparation. Recorded paper results remain unchanged. The demonstration MP4 is the existing supplied video, not a newly generated experiment.

## Publication-scope update

The full manuscript PDF, LaTeX, bibliography and class file were moved outside the code repository. README equations are copied from the manuscript equation environments, preserving its notation. The custom Mermaid diagram was replaced with original Fig. 1. All six figures were rendered directly from the manuscript figure PDFs with Poppler at 180 dpi. Figure provenance records source-PDF and output-PNG hashes. Numerical sources, frozen cases and recorded experimental results were not modified. This update does not rewrite historical commits.
