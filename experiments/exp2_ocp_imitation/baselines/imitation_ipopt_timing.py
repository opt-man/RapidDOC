"""Strict forward numerical timing for the private CasADi/IPOPT plugin.

The private library is the same IPOPT 3.14.19 algorithm, linked to the stock
wheel's MUMPS and BLAS. Its hooks cover MUMPS numerical jobs 2/3 and the
primal-dual solve's reduction/residual arithmetic, excluding all derivative,
KKT assembly, symbolic analysis, cache and independent copy/allocation work.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path

import casadi as ca

ROOT = Path(__file__).resolve().parent
TIMING_PROTOCOL = "solve-kernels-v4"
KERNEL_PROTOCOL = "mumps-numeric-and-pd-arithmetic-v4"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeIpoptTimer:
    """Load a verified private IPOPT plugin; never replace the global wheel."""

    plugin = "ipopt_frs_v4"
    _runtime = None
    _directories = []

    @classmethod
    def _load(cls):
        if cls._runtime is not None:
            return cls._runtime
        if os.name != "nt":
            raise RuntimeError("Strict CasADi/SafePDP forward timing currently requires the private Windows IPOPT build")
        runtime_root = os.environ.get("IMITATION_IPOPT_V4_ROOT")
        source_root = os.environ.get("IMITATION_IPOPT_SOURCE_ROOT")
        if not runtime_root or not source_root:
            raise RuntimeError("Set IMITATION_IPOPT_V4_ROOT to the instrumented install and "
                               "IMITATION_IPOPT_SOURCE_ROOT to the source root recorded by its "
                               "build manifest; source and binary verification are mandatory")
        prefix = Path(runtime_root).expanduser().resolve()
        source_root = Path(source_root).expanduser().resolve()
        manifest_path = prefix/"build_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Strict IPOPT forward timing runtime missing: {manifest_path}; "
                "provide the instrumented external runtime before constructing this baseline. "
                "Whole-NLP wall time cannot substitute for numerical kernel time.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        package = Path(ca.__file__).resolve().parent
        if (manifest.get("timing_protocol") != TIMING_PROTOCOL
                or manifest.get("kernel_protocol") != KERNEL_PROTOCOL
                or manifest.get("casadi_version") != ca.__version__
                or manifest.get("casadi_git_revision") != ca.CasadiMeta.git_revision()
                or manifest.get("casadi_core_sha256") != _digest(package/"libcasadi.dll")):
            raise RuntimeError("Private IPOPT timer build does not match this CasADi wheel or v4 protocol; rebuild it")
        for name, expected in manifest["library_sha256"].items():
            path = prefix/"bin"/name
            if not path.is_file() or _digest(path) != expected:
                raise RuntimeError(f"Private IPOPT timer library changed or is missing: {path}; rebuild it")
        for name, expected in manifest["source_sha256"].items():
            path = source_root/name
            if not path.is_file() or _digest(path) != expected:
                raise RuntimeError(f"Private IPOPT timer source changed: {path}; rebuild it")
        for name, expected in manifest["numerical_dependency_sha256"].items():
            if _digest(package/name) != expected:
                raise RuntimeError(f"CasADi numerical dependency changed: {name}; rebuild the private timer")
        cls._directories.extend([os.add_dll_directory(str(prefix/"bin")),
                                 os.add_dll_directory(str(package))])
        library = ct.CDLL(str(prefix/"bin"/"libfrs_casadi_ipopt_v4.dll"))
        protocol = library.frs_ipopt_solve_timer_protocol_v4
        protocol.argtypes, protocol.restype = [], ct.c_char_p
        if protocol().decode() != KERNEL_PROTOCOL:
            raise RuntimeError("Private IPOPT library has an unsupported numerical timing boundary")
        plugin = ct.CDLL(str(prefix/"bin"/"libcasadi_nlpsol_ipopt_frs_v4.dll"))
        register = plugin.casadi_load_nlpsol_ipopt_frs_v4
        register.argtypes, register.restype = [], None
        register()
        manifest = dict(manifest, verified_runtime_root=str(prefix),
                        verified_source_root=str(source_root))
        cls._runtime = (library, plugin, manifest)
        return cls._runtime

    def __init__(self):
        self.library, self.plugin_library, self.build_manifest = self._load()
        self.runtime_details = self.build_manifest
        self._reset = self.library.frs_ipopt_solve_timer_reset_v4
        self._reset.argtypes, self._reset.restype = [], None
        self._read = self.library.frs_ipopt_solve_timer_read_v4
        self._read.argtypes, self._read.restype = [], ct.c_double
        self._seconds = self.library.frs_ipopt_solve_timer_seconds_v4
        self._seconds.argtypes, self._seconds.restype = [ct.c_int], ct.c_double
        self._count = self.library.frs_ipopt_solve_timer_count_v4
        self._count.argtypes, self._count.restype = [ct.c_int], ct.c_ulonglong
        self._armed, self.last_counts, self.last_seconds = False, None, None

    def reset(self):
        if self._armed:
            raise RuntimeError("IPOPT kernel timer already awaits a matching solve")
        self._reset()
        self._armed = True

    def cancel(self):
        self._armed = False

    def read(self):
        if not self._armed:
            raise RuntimeError("IPOPT kernel timer requires a matching solve")
        self._armed = False
        total = float(self._read())
        self.last_counts = tuple(int(self._count(i)) for i in range(4))
        self.last_seconds = tuple(float(self._seconds(i)) for i in range(4))
        if (not math.isfinite(total) or total <= 0
                or any(not math.isfinite(value) or value < 0 for value in self.last_seconds)
                or self.last_counts[0] == 0 or self.last_counts[1] == 0
                or not math.isclose(total, sum(self.last_seconds), rel_tol=1e-12, abs_tol=1e-15)):
            raise RuntimeError("IPOPT produced no valid v4 numerical factorization/backsolve timing; refusing a result row")
        return total
