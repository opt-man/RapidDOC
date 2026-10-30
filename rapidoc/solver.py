"""Minimal ctypes binding shared by the archived RapidDOC experiments.

Set FRS_SOLVER_DLL to the locally built factor-reuse library when it is not in
the archive's build/bin directory. Native dependencies are installed separately.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_NAME = ("factor_reuse_solver.dll" if os.name == "nt" else
                 "libfactor_reuse_solver.dylib" if sys.platform == "darwin" else
                 "libfactor_reuse_solver.so")
DLL = Path(os.environ.get("FRS_SOLVER_DLL", ROOT / "build" / "bin" / _LIBRARY_NAME))
MAX_RESIDUAL_HISTORY = 256
_dll_directory = None
_library = None


class FrsStats(ct.Structure):
    _fields_ = [
        ("elapsed_seconds", ct.c_double),
        ("mean_iterations", ct.c_double),
        ("relative_residual", ct.c_double),
        ("used_fallback", ct.c_int),
        ("rhs_pack_seconds", ct.c_double),
        ("backward_seconds", ct.c_double),
        ("trsm_seconds", ct.c_double),
        ("forward_seconds", ct.c_double),
        ("output_unpack_seconds", ct.c_double),
        ("arnoldi_seconds", ct.c_double),
        ("residual_seconds", ct.c_double),
        ("residual_pack_seconds", ct.c_double),
        ("residual_hx_seconds", ct.c_double),
        ("residual_at_lambda_seconds", ct.c_double),
        ("residual_ax_seconds", ct.c_double),
        ("residual_unpack_seconds", ct.c_double),
        ("residual_subtract_norm_seconds", ct.c_double),
        ("backward_gemm_seconds", ct.c_double),
        ("backward_copy_seconds", ct.c_double),
        ("backward_scaled_copy_seconds", ct.c_double),
        ("backward_trsm_kernel_seconds", ct.c_double),
        ("backward_gemm_calls", ct.c_int),
        ("backward_copy_calls", ct.c_int),
        ("backward_scaled_copy_calls", ct.c_int),
        ("backward_trsm_calls", ct.c_int),
        ("forward_gemm_seconds", ct.c_double),
        ("forward_copy_seconds", ct.c_double),
        ("forward_scaled_copy_seconds", ct.c_double),
        ("forward_trsm_kernel_seconds", ct.c_double),
        ("forward_gemm_calls", ct.c_int),
        ("forward_copy_calls", ct.c_int),
        ("forward_scaled_copy_calls", ct.c_int),
        ("forward_trsm_calls", ct.c_int),
        ("setup_seconds", ct.c_double),
        ("difference_matvec_seconds", ct.c_double),
        ("projection_seconds", ct.c_double),
        ("input_copy_seconds", ct.c_double),
        ("workspace_seconds", ct.c_double),
        ("control_seconds", ct.c_double),
        ("final_update_seconds", ct.c_double),
        ("mean_cached_backsolves", ct.c_double),
        ("residual_history_count", ct.c_int),
        ("residual_history", ct.c_double * MAX_RESIDUAL_HISTORY),
        ("target_build_seconds", ct.c_double),
        ("sensitivity_linearization_seconds", ct.c_double),
    ]

def library():
    """Load the native library lazily, so data and plotting need no native build."""
    global _dll_directory, _library
    if _library is not None:
        return _library
    if not DLL.is_file():
        raise FileNotFoundError(
            f"RapidDOC library not found: {DLL}. Build it first or set FRS_SOLVER_DLL."
        )
    if hasattr(os, "add_dll_directory"):
        _dll_directory = os.add_dll_directory(str(DLL.resolve().parent))
    lib = ct.CDLL(str(DLL.resolve()))
    ptr = ct.POINTER(ct.c_double)
    lib.frs_last_error.argtypes = []
    lib.frs_last_error.restype = ct.c_char_p
    lib.frs_create.argtypes = [
        ct.c_int, ct.c_int, ct.c_int, ptr, ptr, ptr, ct.POINTER(ct.c_void_p),
    ]
    lib.frs_create.restype = ct.c_int
    lib.frs_create_from_factors.argtypes = [
        ct.c_int, ct.c_int, ct.c_int,
        ptr, ptr, ptr, ptr, ptr, ptr, ct.POINTER(ct.c_void_p),
    ]
    lib.frs_create_from_factors.restype = ct.c_int
    lib.frs_destroy.argtypes = [ct.c_void_p]
    lib.frs_destroy.restype = None
    lib.frs_solve.argtypes = [
        ct.c_void_p, ct.c_int, ptr, ptr, ptr, ptr,
        ct.c_double, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ptr, ct.POINTER(FrsStats),
    ]
    lib.frs_solve.restype = ct.c_int
    _library = lib
    return lib


def _pointer(values: np.ndarray):
    return values.ctypes.data_as(ct.POINTER(ct.c_double))


def _check(status: int) -> None:
    if status == 0:
        return
    raw = library().frs_last_error()
    message = raw.decode("utf-8", errors="replace") if raw else "unknown solver error"
    raise RuntimeError(message)
