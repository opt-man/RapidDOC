"""Native factor solves and independent KKT checks for Experiment 1."""
from __future__ import annotations

import ctypes as ct
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rapidoc import solver as native
from frozen_cases import FrozenKKT

METHODS = ("Ours", "No-reuse GMRES")


def _array(value):
    return np.ascontiguousarray(value, dtype=np.float64)


class NativeFactor:
    def __init__(self, case: FrozenKKT, forward: bool):
        prefix = "forward" if forward else "target"
        self.blocks = tuple(_array(getattr(case, f"{prefix}_{name}")) for name in (
            "hessian", "terminal_hessian", "dynamics"
        ))
        self.handle = ct.c_void_p()
        cache = tuple(_array(value) for value in (
            case.value_hessians, case.control_cholesky, case.cross_factors
        ))
        self.from_fatrop_cache = forward and all(value.size for value in cache)
        if self.from_fatrop_cache:
            native._check(native.library().frs_create_from_factors(
                case.horizon, case.nx, case.nu,
                *(native._pointer(block) for block in (*self.blocks, *cache)),
                ct.byref(self.handle)
            ))
        else:
            native._check(native.library().frs_create(
                case.horizon, case.nx, case.nu,
                *(native._pointer(block) for block in self.blocks), ct.byref(self.handle)
            ))

    def solve(self, target, rhs, tolerance, max_iterations):
        target_blocks = tuple(_array(block) for block in target)
        rhs = _array(rhs)
        output = np.empty_like(rhs)
        stats = native.FrsStats()
        started = time.perf_counter()
        native._check(native.library().frs_solve(
            self.handle, 1, *(native._pointer(block) for block in target_blocks),
            native._pointer(rhs), tolerance, max_iterations, 1, 0, 0,
            native._pointer(output), ct.byref(stats)
        ))
        elapsed = time.perf_counter() - started
        if stats.used_fallback:
            raise RuntimeError("unexpected fresh-factor fallback")
        return output, stats, elapsed

    def close(self):
        if self.handle:
            native.library().frs_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def _kkt_apply(case, vector):
    sd, T, nx = case.stage_dimension, case.horizon, case.nx
    primal = vector[:case.primal_dimension, 0]
    dual = vector[case.primal_dimension:, 0].reshape(T+1, nx)
    out_p = np.empty_like(primal)
    for stage in range(T):
        z = primal[stage*sd:(stage+1)*sd]
        value = case.target_hessian[stage] @ z
        value[:nx] += dual[stage]
        value += case.target_dynamics[stage].T @ dual[stage+1]
        out_p[stage*sd:(stage+1)*sd] = value
    out_p[T*sd:] = case.target_terminal_hessian @ primal[T*sd:] + dual[T]
    out_d = np.empty((T+1, nx))
    out_d[0] = primal[:nx]
    for stage in range(T):
        z = primal[stage*sd:(stage+1)*sd]
        out_d[stage+1] = case.target_dynamics[stage] @ z + primal[(stage+1)*sd:(stage+1)*sd+nx]
    return np.concatenate((out_p, out_d.reshape(-1)))[:, None]


def _residual(case, solution):
    return float(np.linalg.norm(case.rhs-_kkt_apply(case, solution)) /
                 max(np.linalg.norm(case.rhs), np.finfo(float).tiny))


def _error(solution, reference):
    return float(np.linalg.norm(solution-reference) /
                 max(np.linalg.norm(reference), np.finfo(float).tiny))


def _native_message() -> str:
    raw = native.library().frs_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "native solver error"


def _solve(
    factor: NativeFactor,
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    rhs: np.ndarray,
    tolerance: float,
    max_iterations: int,
    method: str,
) -> tuple[np.ndarray, native.FrsStats, float, int, str]:
    target_blocks = tuple(_array(block) for block in target)
    rhs_array = _array(rhs)
    output = np.empty_like(rhs_array)
    stats = native.FrsStats()
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    entry = getattr(native.library(), "frs_solve_ablation", None)
    if entry is None:
        raise RuntimeError(
            "Rebuild factor_reuse_solver: the loaded DLL lacks solve-only ablation timing")
    dp = ct.POINTER(ct.c_double)
    entry.argtypes = [ct.c_void_p, dp, dp, dp, dp, ct.c_double, ct.c_int,
                      ct.c_int, dp, ct.POINTER(native.FrsStats), dp]
    entry.restype = ct.c_int
    call_seconds = ct.c_double()
    status = entry(
        factor.handle, *(native._pointer(block) for block in target_blocks),
        native._pointer(rhs_array), tolerance, max_iterations, int(method == "Ours"),
        native._pointer(output), ct.byref(stats), ct.byref(call_seconds),
    )
    stats.call_seconds = float(call_seconds.value)
    stats.excluded_preparation_seconds = max(
        0.0, stats.call_seconds - float(stats.elapsed_seconds))
    elapsed_ms = 1e3 * float(stats.elapsed_seconds)
    return output, stats, elapsed_ms, int(status), "" if status == 0 else _native_message()


def _trace(stats: native.FrsStats) -> str:
    count = min(int(stats.residual_history_count), native.MAX_RESIDUAL_HISTORY)
    return json.dumps(
        [float(stats.residual_history[index]) for index in range(count)],
        separators=(",", ":"),
    )
