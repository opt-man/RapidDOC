# Contributing

Keep mathematical targets and experiment protocols explicit. Preserve the original KKT residual checks, factor-cache provenance and recorded failure cases. New results belong in a new output directory; do not overwrite frozen data or paper tables. Include the platform, dependency versions, tolerances, timing boundary and reproduction command when reporting an issue or a performance change.

Run `python tools/verify_artifact.py` and `python -m unittest discover -s tests -v` before proposing changes. Native changes additionally need target-residual and gradient checks with the compatible external solver builds.
