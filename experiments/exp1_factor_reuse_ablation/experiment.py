"""Backward-only factor-reuse ablation on frozen real-FATROP OCP cases.

CasADi--MUMPS is used only as an untimed accuracy reference. The timed branches
are Ours, exact-target refactorization followed by a direct solve, and
zero-initialized unpreconditioned GMRES. All are checked against the same target
relative-residual tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import casadi as ca
import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from linear_system import NativeFactor, _error, _kkt_apply, _residual, _solve, _trace
from frozen_cases import load


ROOT = Path(__file__).resolve().parent
PROTOCOL = "gmres-real-ocp-factor-reuse-ablation-v5-target-refactor"
TIMING_BOUNDARY = "frozen-target-post-forward-linear-solve-v2"
PROBLEMS = ("cartpole", "quadrotor", "robotarm")
METHODS = ("Ours", "Target refactor", "No-reuse GMRES")
DEFAULT_CASES = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "results" / "rerun"

FIELDS = (
    "protocol", "problem", "seed", "method", "N", "nx", "nu",
    "kkt_dimension", "natural_mismatch", "rho0", "initial_guess",
    "preconditioner", "converged", "iterations", "cached_backsolves",
    "time_median_ms", "time_p25_ms", "time_p75_ms",
    "relative_kkt_residual", "relative_adjoint_error_vs_casadi",
    "casadi_relative_kkt_residual", "reference_solver", "residual_trace",
    "repeats", "tolerance", "requested_max_iterations",
    "effective_max_iterations", "status", "message",
    "timing_boundary", "call_median_ms", "excluded_preparation_median_ms",
    "case_path",
)


def _target(case):
    return (
        case.target_hessian,
        case.target_terminal_hessian,
        case.target_dynamics,
    )


def _forward(case):
    return (
        case.forward_hessian,
        case.forward_terminal_hessian,
        case.forward_dynamics,
    )


def _mismatch(case) -> float:
    numerator = np.sqrt(
        np.sum((case.target_hessian - case.forward_hessian) ** 2)
        + np.sum((case.target_terminal_hessian
                  - case.forward_terminal_hessian) ** 2)
        + np.sum((case.target_dynamics - case.forward_dynamics) ** 2)
    )
    denominator = np.sqrt(
        np.sum(case.forward_hessian ** 2)
        + np.sum(case.forward_terminal_hessian ** 2)
        + np.sum(case.forward_dynamics ** 2)
    )
    return float(numerator / max(denominator, np.finfo(float).tiny))


def _target_kkt(case) -> sp.csc_matrix:
    """Assemble the sparse target KKT matrix in the native solver ordering."""
    T, nx, nu = case.horizon, case.nx, case.nu
    stage_dimension = nx + nu
    primal_dimension = case.primal_dimension
    hessian = sp.block_diag(
        [*(sp.csc_matrix(block) for block in case.target_hessian),
         sp.csc_matrix(case.target_terminal_hessian)],
        format="csc",
    )
    dynamics = sp.lil_matrix(((T + 1) * nx, primal_dimension), dtype=float)
    dynamics[:nx, :nx] = sp.eye(nx, format="csc")
    for stage in range(T):
        row = (stage + 1) * nx
        column = stage * stage_dimension
        dynamics[row:row + nx, column:column + stage_dimension] = \
            case.target_dynamics[stage]
        next_state = (stage + 1) * stage_dimension \
            if stage < T - 1 else T * stage_dimension
        dynamics[row:row + nx, next_state:next_state + nx] = \
            sp.eye(nx, format="csc")
    dynamics = dynamics.tocsc()
    return sp.bmat(
        [[hessian, dynamics.T], [dynamics, None]], format="csc"
    )


def _casadi_mumps_reference(case) -> tuple[np.ndarray, float]:
    """Compute an untimed sparse direct-solve reference with CasADi--MUMPS."""
    matrix = _target_kkt(case)
    # This independent check prevents an ordering mistake in the assembled
    # reference matrix from being mistaken for an adjoint error.
    probe = np.random.default_rng(0).standard_normal(
        (case.total_dimension, 1)
    )
    native_action = _kkt_apply(case, probe)
    assembly_error = np.linalg.norm(matrix @ probe - native_action) / max(
        np.linalg.norm(native_action), np.finfo(float).tiny
    )
    if assembly_error > 1e-13:
        raise RuntimeError(
            f"CasADi reference KKT ordering mismatch: {assembly_error:.3e}"
        )
    reference = np.asarray(
        ca.solve(ca.DM(matrix), ca.DM(case.rhs), "mumps"), dtype=float
    )
    residual = _residual(case, reference)
    if not np.all(np.isfinite(reference)) or residual > 1e-10:
        raise RuntimeError(
            f"CasADi--MUMPS reference residual is {residual:.3e}"
        )
    return reference, residual


def _target_refactor_solve(case, target, tolerance):
    """Time a fresh exact-target Riccati factorization and one direct solve."""
    started = time.perf_counter()
    with NativeFactor(case, forward=False) as factor:
        solution, stats, _ = factor.solve(target, case.rhs, tolerance, 2)
    elapsed_ms = 1e3 * (time.perf_counter() - started)
    stats.call_seconds = elapsed_ms * 1e-3
    stats.excluded_preparation_seconds = 0.0
    if abs(float(stats.mean_iterations)) > 1e-12:
        raise RuntimeError("target refactorization unexpectedly entered GMRES")
    return solution, stats, elapsed_ms, 0, ""


def _write(path: Path, rows, fields=FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _run_case(case, case_path: Path, repeats: int, tolerance: float,
              requested_max_iterations: int, effective_max_iterations: int,
              methods):
    target = _target(case)
    reference, reference_residual = _casadi_mumps_reference(case)

    with NativeFactor(case, forward=True) as factor:
        direct, _, _ = factor.solve(_forward(case), case.rhs, 1e-12, 2)
    rho0 = _residual(case, direct)

    methods = tuple(methods)
    samples = {method: [] for method in methods}
    with NativeFactor(case, forward=True) as ours_factor, \
            NativeFactor(case, forward=True) as plain_factor:
        factors = {"Ours": ours_factor, "No-reuse GMRES": plain_factor}
        for repeat in range(repeats):
            shift = repeat % len(methods)
            order = methods[shift:] + methods[:shift]
            for method in order:
                if method == "Target refactor":
                    samples[method].append(
                        _target_refactor_solve(case, target, tolerance))
                else:
                    samples[method].append(_solve(
                        factors[method], target, case.rhs, tolerance,
                        effective_max_iterations, method,
                    ))

    rows = []
    for method in methods:
        values = samples[method]
        times = np.asarray([value[2] for value in values])
        median_time = float(np.median(times))
        representative = min(values, key=lambda value: abs(value[2] - median_time))
        solution, stats = representative[0], representative[1]
        statuses = [value[3] for value in values]
        messages = sorted({value[4] for value in values if value[4]})
        residual = _residual(case, solution)
        backsolves = float(np.median([
            value[1].mean_cached_backsolves for value in values
        ]))
        invariant_ok = (
            (method != "No-reuse GMRES" or backsolves == 0.0)
            and (method != "Target refactor" or (
                backsolves == 1.0
                and float(np.median([value[1].mean_iterations for value in values])) == 0.0
            ))
        )
        converged = all(status == 0 for status in statuses) \
            and residual <= tolerance and invariant_ok
        if not invariant_ok:
            messages.append("method-specific solve invariant failed")
        if method == "Ours":
            initial_guess = "cached one-shot"
            preconditioner = "cached K_F"
        elif method == "Target refactor":
            initial_guess = "exact-target direct solve"
            preconditioner = "fresh target factors"
        else:
            initial_guess = "zero"
            preconditioner = "none"
        rows.append({
            "protocol": PROTOCOL,
            "problem": case.problem,
            "seed": case.seed,
            "method": method,
            "N": case.horizon,
            "nx": case.nx,
            "nu": case.nu,
            "kkt_dimension": case.total_dimension,
            "natural_mismatch": _mismatch(case),
            "rho0": rho0,
            "initial_guess": initial_guess,
            "preconditioner": preconditioner,
            "converged": int(converged),
            "iterations": float(np.median([
                value[1].mean_iterations for value in values
            ])),
            "cached_backsolves": backsolves,
            "time_median_ms": median_time,
            "timing_boundary": TIMING_BOUNDARY,
            "call_median_ms": float(np.median([
                1e3 * value[1].call_seconds for value in values])),
            "excluded_preparation_median_ms": float(np.median([
                1e3 * value[1].excluded_preparation_seconds for value in values])),
            "time_p25_ms": float(np.percentile(times, 25)),
            "time_p75_ms": float(np.percentile(times, 75)),
            "relative_kkt_residual": residual,
            "relative_adjoint_error_vs_casadi": _error(solution, reference),
            "casadi_relative_kkt_residual": reference_residual,
            "reference_solver": f"CasADi {ca.__version__} MUMPS (untimed)",
            "residual_trace": _trace(stats),
            "repeats": repeats,
            "tolerance": tolerance,
            "requested_max_iterations": requested_max_iterations,
            "effective_max_iterations": effective_max_iterations,
            "status": "ok" if converged else "tolerance_not_reached",
            "message": "; ".join(messages),
            "case_path": str(case_path),
        })
    return rows


def _summarize(rows):
    failures = [row for row in rows if not int(row["converged"])]
    if failures:
        examples = ", ".join(
            f"{row['problem']}/seed_{int(row['seed']):03d}/{row['method']}"
            for row in failures[:5]
        )
        raise RuntimeError(
            "The ablation is a time-to-equal-tolerance comparison, but some "
            f"runs did not reach the requested tolerance: {examples}. "
            "Increase --max-iterations and rerun with --rerun."
        )
    summary = []
    for problem in PROBLEMS:
        for method in METHODS:
            group = [row for row in rows
                     if row["problem"] == problem and row["method"] == method]
            if not group:
                continue
            times = np.asarray([float(row["time_median_ms"]) for row in group])
            iterations = np.asarray([float(row["iterations"]) for row in group])
            summary.append({
                "protocol": PROTOCOL,
                "timing_boundary": TIMING_BOUNDARY,
                "problem": problem,
                "method": method,
                "instances": len(group),
                "converged": sum(int(row["converged"]) for row in group),
                "convergence_rate": float(np.mean([
                    int(row["converged"]) for row in group
                ])),
                "time_median_ms": float(np.median(times)),
                "time_p25_ms": float(np.percentile(times, 25)),
                "time_p75_ms": float(np.percentile(times, 75)),
                "iterations_median": float(np.median(iterations)),
                "iterations_p25": float(np.percentile(iterations, 25)),
                "iterations_p75": float(np.percentile(iterations, 75)),
                "max_relative_kkt_residual": max(
                    float(row["relative_kkt_residual"]) for row in group
                ),
                "max_relative_adjoint_error_vs_casadi": max(
                    float(row["relative_adjoint_error_vs_casadi"])
                    for row in group
                ),
                "max_casadi_relative_kkt_residual": max(
                    float(row["casadi_relative_kkt_residual"])
                    for row in group
                ),
            })
    return summary


def _write_formatted_tables(output: Path, summary) -> None:
    task_names = {
        "cartpole": "Cartpole",
        "quadrotor": "Quadrotor",
        "robotarm": "Robot Arm",
    }
    ordered = [
        row for problem in PROBLEMS for method in METHODS
        for row in summary
        if row["problem"] == problem and row["method"] == method
    ]

    markdown = [
        "| Task | Method | Backward ms, median [P25, P75] | "
        "GMRES iterations, median [P25, P75] | Converged | "
        "Max relative KKT residual | Max adjoint error vs CasADi--MUMPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        markdown.append(
            f"| {task_names[row['problem']]} | {row['method']} | "
            f"{float(row['time_median_ms']):.3f} "
            f"[{float(row['time_p25_ms']):.3f}, "
            f"{float(row['time_p75_ms']):.3f}] | "
            f"{float(row['iterations_median']):.1f} "
            f"[{float(row['iterations_p25']):.1f}, "
            f"{float(row['iterations_p75']):.1f}] | "
            f"{int(row['converged'])}/{int(row['instances'])} | "
            f"{float(row['max_relative_kkt_residual']):.2e} | "
            f"{float(row['max_relative_adjoint_error_vs_casadi']):.2e} |"
        )
    (output / "gmres_real_ocp_ablation_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation of forward-factor reuse on frozen nonlinear OCPs. "
        "Times and iterations are reported as median [P25, P75] across "
        "frozen instances; each instance time is the median of its measured calls.}",
        "\\label{tab:gmres-reuse-ablation}",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Task & Method & Backward (ms) & Iter. & Conv. & Rel. KKT & "
        "Rel. adjoint vs. MUMPS \\\\",
        "\\midrule",
    ]
    for problem in PROBLEMS:
        problem_rows = [row for row in ordered if row["problem"] == problem]
        for index, row in enumerate(problem_rows):
            task = task_names[problem] if index == 0 else ""
            method = {
                "Ours": "\\textbf{Ours}",
                "Target refactor": "Target refactor",
                "No-reuse GMRES": "No-reuse GMRES",
            }[row["method"]]
            latex.append(
                f"{task} & {method} & "
                f"{float(row['time_median_ms']):.3f} "
                f"[{float(row['time_p25_ms']):.3f}, "
                f"{float(row['time_p75_ms']):.3f}] & "
                f"{float(row['iterations_median']):.1f} "
                f"[{float(row['iterations_p25']):.1f}, "
                f"{float(row['iterations_p75']):.1f}] & "
                f"{int(row['converged'])}/{int(row['instances'])} & "
                f"{float(row['max_relative_kkt_residual']):.2e} & "
                f"{float(row['max_relative_adjoint_error_vs_casadi']):.2e} \\\\"
            )
        if problem != PROBLEMS[-1]:
            latex.append("\\midrule")
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    (output / "gmres_real_ocp_ablation_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS,
                        default=list(PROBLEMS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--instances", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--max-iterations", type=int, default=0,
        help=("0 allows unrestarted GMRES up to the target KKT dimension; "
              "a positive value sets an explicit common iteration budget"),
    )
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.instances <= 32 or args.repeats <= 0:
        raise ValueError("instances must be in [1,32] and repeats must be positive")
    if args.tolerance <= 0.0 or args.max_iterations < 0:
        raise ValueError("tolerance must be positive and max-iterations nonnegative")

    cases_dir = args.cases_dir.resolve()
    output = args.output_dir.resolve()
    raw_path = output / "gmres_real_ocp_ablation_raw.csv"
    rows = [] if args.rerun else _read(raw_path)
    if any(row.get("protocol") != PROTOCOL or
           row.get("timing_boundary") != TIMING_BOUNDARY for row in rows):
        parser.error("Old timing data cannot be mixed with solve-only results; use --rerun or a new output directory")
    completed = {(row["problem"], int(row["seed"]), row["method"])
                 for row in rows}

    with threadpool_limits(limits=1, user_api="blas"):
        for problem in args.problems:
            for seed in range(args.instances):
                if all((problem, seed, method) in completed for method in args.methods):
                    continue
                case_path = cases_dir / problem / f"seed_{seed:03d}.npz"
                if not case_path.exists():
                    raise FileNotFoundError(case_path)
                print(f"{problem:9s} seed={seed:02d}", flush=True)
                case = load(case_path)
                effective_max_iterations = (
                    case.total_dimension
                    if args.max_iterations == 0 else args.max_iterations
                )
                rows.extend(_run_case(
                    case, case_path, args.repeats, args.tolerance,
                    args.max_iterations, effective_max_iterations, args.methods,
                ))
                _write(raw_path, rows)

    summary = _summarize(rows)
    _write(output / "gmres_real_ocp_ablation_summary.csv", summary,
           list(summary[0]) if summary else [])
    if summary:
        _write_formatted_tables(output, summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gmres_real_ocp_ablation_manifest.json").write_text(
        json.dumps({
            "protocol": PROTOCOL,
            "problems": args.problems,
            "instances_per_problem": args.instances,
            "methods": list(args.methods),
            "cases_dir": str(cases_dir),
            "timing_warmups": 0,
            "repeats": args.repeats,
            "tolerance": args.tolerance,
            "requested_max_iterations": args.max_iterations,
            "iteration_policy": (
                "target KKT dimension" if args.max_iterations == 0
                else "explicit common iteration budget"
            ),
            "effective_max_iterations": {
                problem: (
                    load(cases_dir / problem / "seed_000.npz").total_dimension
                    if args.max_iterations == 0 else args.max_iterations
                )
                for problem in args.problems
            },
            "threads": 1,
            "timing_boundary": TIMING_BOUNDARY,
            "accuracy_reference": (
                f"CasADi {ca.__version__} sparse MUMPS direct solve; untimed"
            ),
            "accuracy_metrics": [
                "independently certified relative target-KKT residual",
                "relative adjoint error against CasADi--MUMPS",
            ],
            "comparison_policy": (
                "summaries are emitted only if all timed methods reach the "
                "same requested relative-residual tolerance on every case"
            ),
            "timing_alignment": (
                "post-forward adjoint linear algebra with zero timing warmups; "
                "Target refactor includes its required exact-target factorization"
            ),
            "excluded_from_timing": [
                "forward", "case IO",
                "CasADi--MUMPS accuracy reference", "startup",
                "target construction because frozen target arrays are inputs",
                "one-time Ours and No-reuse handle/workspace construction",
            ],
            "included_in_timing": [
                "cached initialization", "preconditioner applications", "GMRES",
                "target residual checks", "numerical vector operations",
                "Target-refactor handle construction, exact-target generalized "
                "Riccati factorization, and direct solve",
            ],
        }, indent=2), encoding="utf-8",
    )
    print(f"Wrote real-OCP GMRES ablation to {output}")


if __name__ == "__main__":
    main()
