"""Open- and closed-loop nonlinear imitation learning with real updates.

Commands: generate, prepare, run, plot, closed-loop, plot-closed-loop.
No synthetic KKT cache, trajectory Jacobian, timing warmup, or fallback solve.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
# The private sIPOPT BLAS/LAPACK DLLs forward to MKL. Its runtime also loads
# dependencies on demand, so add_dll_directory alone is insufficient here.
# Set this before importing numerical backends and before their first BLAS call.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
if os.name == "nt" and os.environ.get("IMITATION_SIPOPT_RUNTIME"):
    _sipopt_runtime = Path(os.environ["IMITATION_SIPOPT_RUNTIME"])
    os.environ["PATH"] = str(_sipopt_runtime)+os.pathsep+os.environ.get("PATH", "")

import casadi as ca
import numpy as np
from threadpoolctl import threadpool_limits

import imitation_models as models

ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = ROOT.parents[1]
if str(ARCHIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_ROOT))
DEFAULT_METHODS = ("Ours", "SafePDP", "CasADi", "acados")
METHODS = DEFAULT_METHODS
METHOD_COLORS = {
    "Ours": "#2457FF",
    "SafePDP": "#F03A3A",
    "sIPOPT": "#21C95B",
    "CasADi": "#21C95B",
    "acados": "#7667C9",
}
METHOD_LINES = {"Ours": "-", "SafePDP": "--", "sIPOPT": ":", "CasADi": "-.", "acados": "--"}
METHOD_MARKERS = {"Ours": "o", "SafePDP": "s", "sIPOPT": "^", "CasADi": "D", "acados": "v"}
METHOD_DISPLAY = {"Ours": "RapidDOC"}


def method_display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)
# Report only each backend's numerical solver invocation.  Per-sample graph or
# auxiliary-system construction, cache preparation, independent certification,
# and gradient contraction remain diagnostics and are excluded from the bars.
TIMING_PROTOCOL = "solver-call-only-v1"
NATIVE_TIMING_PROTOCOL = "solve-kernels-v4"
FROZEN_FAR_MODEL = (
    "nonlinear-imitation-v4-native-constraints",
    "25e9af53dff959f974301fb6b746842cf7c22e7075c18185699c16a35e31c03d",
)
FROZEN_FAR_COMPATIBLE_MODEL = (
    "nonlinear-imitation-v5-moderate-initials",
    "1ffafb17810677b52635d8fcc609fe58ea3b5bec04c5f08ada60c0e43195eb86",
)
FIELDS = ("timing_protocol", "problem", "method", "seed", "epoch", "train_loss", "validation_loss",
          "train_seconds", "forward_ms", "backward_ms", "total_ms", "target_ms", "cache_ms",
          "linear_ms", "contraction_ms", "validation_seconds", "mean_iterations",
          "max_relative_residual", "min_margin", "gradient_norm", "theta")
CHECKPOINT_VERSION = 1


def array(v):
    return np.ascontiguousarray(v, dtype=np.float64)


def ptr(v):
    return v.ctypes.data_as(ct.POINTER(ct.c_double))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(methods):
    """Hash the actual relocated implementation and selected external source."""
    files = [ROOT/"imitation_learning.py", ROOT/"imitation_models.py"]
    if "Ours" in methods:
        files.extend([ROOT/"native"/"imitation_native.cpp",
                      ROOT/"native"/"generated"/"imitation_models.c",
                      ROOT/"native"/"generated"/"imitation_models.h",
                      ARCHIVE_ROOT/"rapidoc"/"solver.py",
                      ARCHIVE_ROOT/"rapidoc"/"factor_reuse_solver.cpp",
                      ARCHIVE_ROOT/"rapidoc"/"factor_reuse_solver.h"])
    for method, filename in (("CasADi", "imitation_casadi.py"),
                             ("acados", "imitation_acados.py")):
        if method in methods:
            files.append(ROOT/"baselines"/filename)
    if set(methods) & {"SafePDP", "CasADi"}:
        files.append(ROOT/"baselines"/"imitation_ipopt_timing.py")
    hashes = {str(path.relative_to(ARCHIVE_ROOT)): digest(path) for path in files}
    if "SafePDP" in methods:
        hashes["external/SafePDP/SafePDP.py"] = digest(safepdp_root()/"SafePDP"/"SafePDP.py")
    return hashes


def safepdp_root():
    value = os.environ.get("IMITATION_SAFEPDP_ROOT")
    if not value:
        raise RuntimeError("Set IMITATION_SAFEPDP_ROOT to the external Safe-PDP checkout "
                           "with patches/safepdp_imitation.patch applied; see EXPERIMENT.md")
    return Path(value).expanduser().resolve()


def _atomic_savez(path, **values):
    """Write a restart checkpoint without exposing a partially written file."""
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def _trim_history(path, start_epoch, fieldnames):
    """Keep only rows preceding a restored checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["epoch"]) < start_epoch]
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _checkpoint_identity(ev, method, seed, args, mode):
    identity = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": mode,
        "problem": ev.model.name,
        "method": method,
        "seed": int(seed),
        "N": int(ev.N),
        "dt": float(ev.dt),
        "learning_rate": float(args.learning_rate if mode == "open-loop"
                               else args.closed_loop_learning_rate),
        "tolerance": float(args.tolerance),
        "forward_tolerance": float(args.forward_tolerance),
        "max_iterations": int(args.max_iterations),
        "forward_max_iterations": int(args.forward_max_iterations),
    }
    if mode == "open-loop":
        identity.update(train_demos=int(args.train_demos),
                        validation_demos=int(args.validation_demos))
    else:
        identity.update(closed_loop_steps=int(args.closed_loop_steps))
    return identity


def _load_checkpoint(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity"].item()))
        if identity != expected:
            raise RuntimeError(
                f"checkpoint configuration differs: {path}; use a new output directory")
        return {name: np.asarray(saved[name]).copy() for name in saved.files
                if name != "identity"}


def _handle_legacy_resume(folder, output, checkpoint, final_epoch):
    """Adopt completed pre-checkpoint runs or preserve and restart partial ones."""
    if checkpoint.exists() or not output.exists():
        return None
    status_path = folder/"status.json"
    try:
        legacy = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        legacy = {}
    if (legacy.get("status") == "ok"
            and int(legacy.get("last_epoch", -1)) >= final_epoch):
        legacy["skipped_completed"] = True
        legacy["legacy_without_checkpoint"] = True
        return legacy
    backup = output.with_name(
        f"{output.stem}.legacy_partial_{time.time_ns()}{output.suffix}")
    output.replace(backup)
    print(f"Preserved incomplete legacy history as {backup.name}; restarting this job", flush=True)
    return None


def pack(X, U):
    return array(np.concatenate((np.concatenate((X[:-1], U), axis=1).ravel(), X[-1])))


def unpack(w, model, N):
    stages = np.asarray(w[:N*(model.nx+model.nu)]).reshape(N, model.nx+model.nu)
    return np.vstack((stages[:, :model.nx], w[-model.nx:])), stages[:, model.nx:].copy()


class Evaluator:
    def __init__(self, model, args):
        self.model, self.N, self.dt = model, args.N, args.dt
        self.functions = model.functions()
        self.zero = np.zeros(model.nx)
        self.plant_parameter_jacobian = ca.Function(
            f"plant_parameter_jacobian_{model.name}",
            [model.x, model.u, model.parameters],
            [ca.jacobian(model.dynamics, model.parameters)[:, :model.d]])

    def parameters(self, theta):
        return np.r_[theta, self.dt]

    def value(self, key, x, u, theta, lam=None):
        return np.asarray(self.functions[key](x, u, self.parameters(theta),
                                             self.zero if lam is None else lam, 1), dtype=float)

    def guess(self, initial):
        # Common multiple-shooting guess, not a demonstration or learned solution.
        X = np.linspace(initial, self.model.goal, self.N+1)
        U = np.tile(self.model.resting_control, (self.N, 1))
        return pack(X, U)

    def margin(self, X, U, theta):
        values = [float(self.value("margin", x, u, theta).item()) for x, u in zip(X[:-1], U)]
        values.append(float(self.value("marginT", X[-1], [], theta).item()))
        minimum = min(values)
        if not np.isfinite(minimum) or minimum < -1e-6:
            raise RuntimeError(f"solution violates an inequality: margin={minimum}")
        return minimum

    def target(self, X, U, theta):
        nx, nu = self.model.nx, self.model.nu
        HT = self.value("hessT", X[-1], [], theta)
        costate = self.value("gradT", X[-1], [], theta).ravel()
        H = np.empty((self.N, nx+nu, nx+nu))
        A = np.empty((self.N, nx, nx+nu))
        mixed = np.empty((self.N, nx+nu, self.model.d))
        # Recover PMP costates from the returned trajectory. This is a vector
        # recursion, not a factorization. Dynamics and constraints do not depend
        # on the learnable weights, so G_B has only these cost-derivative blocks.
        for k in range(self.N-1, -1, -1):
            J = self.value("jac", X[k], U[k], theta)
            H[k] = self.value("hess", X[k], U[k], theta, costate)
            A[k] = -J
            mixed[k] = self.value("mixed", X[k], U[k], theta)
            costate = self.value("grad", X[k], U[k], theta).ravel()[:nx] + J[:, :nx].T@costate
        mixedT = self.value("mixedT", X[-1], [], theta)
        return tuple(array(v) for v in (H, HT, A, mixed, mixedT))

    def parameter_derivatives(self, X, U, theta):
        """Build only the mixed cost derivatives needed after an adjoint solve."""
        mixed = np.asarray([
            self.value("mixed", x, u, theta) for x, u in zip(X[:-1], U)
        ])
        mixedT = self.value("mixedT", X[-1], [], theta)
        return array(mixed), array(mixedT)


class IpoptForward:
    """Persistent explicitly constrained NLP; construction is outside timing."""
    def __init__(self, ev, tolerance, max_iter):
        self.ev = ev
        m, N = ev.model, ev.N
        w = ca.MX.sym("w", N*(m.nx+m.nu)+m.nx)
        par = ca.MX.sym("par", m.d+1+m.nx)
        p, x0 = par[:m.d+1], par[m.d+1:]
        objective, constraints = 0, [w[:m.nx]-x0]
        lbg, ubg = [0.0]*m.nx, [0.0]*m.nx
        for k in range(N):
            start = k*(m.nx+m.nu)
            x, u = w[start:start+m.nx], w[start+m.nx:start+m.nx+m.nu]
            next_x = w[start+m.nx+m.nu:start+2*m.nx+m.nu]
            objective += ev.functions["cost"](x, u, p, ev.zero, 1)
            constraints.append(next_x-ev.functions["f"](x, u, p, ev.zero, 1))
            lbg.extend([0.0]*m.nx)
            ubg.extend([0.0]*m.nx)
            constraints.append(ev.functions["ineq"](x, u, p, ev.zero, 1))
            count = int(m.margins.numel())
            lbg.extend([0.0]*count)
            ubg.extend([np.inf]*count)
        objective += ev.functions["costT"](w[-m.nx:], [], p, ev.zero, 1)
        constraints.append(ev.functions["ineqT"](w[-m.nx:], [], p, ev.zero, 1))
        terminal_count = int(m.terminal_margins.numel())
        lbg.extend([0.0]*terminal_count)
        ubg.extend([np.inf]*terminal_count)
        nlp = {"x": w, "p": par, "f": objective, "g": ca.vertcat(*constraints)}
        self.lbg, self.ubg = lbg, ubg
        self.solver = ca.nlpsol("imitation_ipopt", "ipopt", nlp, {
            "print_time": False, "error_on_fail": False,
            "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.tol": tolerance,
            "ipopt.acceptable_tol": tolerance, "ipopt.max_iter": max_iter,
        })

    def forward(self, theta, initial, guess):
        result = self.solver(
            x0=guess, p=np.r_[self.ev.parameters(theta), initial],
            lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise RuntimeError("IPOPT forward: "+str(stats.get("return_status")))
        return array(result["x"]).ravel()

    def close(self):
        pass


class Ours:
    _lib = None
    _directory = None

    @classmethod
    def library(cls):
        if cls._lib is not None:
            return cls._lib
        path = Path(os.environ.get(
            "IMITATION_NATIVE_DLL", ARCHIVE_ROOT/"build"/"bin"/"imitation_native.dll"))
        if not path.exists():
            raise FileNotFoundError("Build imitation_native or set IMITATION_NATIVE_DLL; see EXPERIMENT.md")
        if hasattr(os, "add_dll_directory"):
            cls._directory = os.add_dll_directory(str(path.parent))
        from rapidoc.solver import FrsStats
        lib = ct.CDLL(str(path))
        dp = ct.POINTER(ct.c_double)
        lib.il_last_error.restype = ct.c_char_p
        lib.il_create.argtypes = [
            ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_int]
        lib.il_create.restype = ct.c_void_p
        lib.il_destroy.argtypes = [ct.c_void_p]
        lib.il_destroy.restype = None
        lib.il_timing_protocol.restype = ct.c_char_p
        if lib.il_timing_protocol().decode() != NATIVE_TIMING_PROTOCOL:
            raise RuntimeError("Rebuild imitation_native for solve-kernels-v4 timing")
        lib.il_forward.argtypes = [ct.c_void_p, dp, dp, dp, dp, dp]
        lib.il_forward.restype = ct.c_int
        lib.il_forward_timing.argtypes = [ct.c_void_p, dp]
        lib.il_forward_timing.restype = ct.c_int
        lib.il_backward.argtypes = [ct.c_void_p, dp, ct.c_double, ct.c_int,
                                    dp, dp, ct.POINTER(FrsStats), dp]
        lib.il_backward.restype = ct.c_int
        cls._lib = lib
        return lib

    def __init__(self, ev, args):
        self.ev, self.args, self.lib = ev, args, self.library()
        native_path = Path(os.environ.get(
            "IMITATION_NATIVE_DLL", ARCHIVE_ROOT/"build"/"bin"/"imitation_native.dll"))
        self.runtime_details = {
            "native_diagnostic_timing_protocol": NATIVE_TIMING_PROTOCOL,
            "forward_timer": "private FATROP numerical call-site counters",
            "binaries": {name: {"path": str(native_path.parent/name),
                                 "sha256": digest(native_path.parent/name)}
                         for name in ("imitation_native.dll", "fatrop.dll", "factor_reuse_solver.dll")},
        }
        self.handle = self.lib.il_create(
            models.NAMES.index(ev.model.name), ev.N, ev.dt,
            args.forward_tolerance, args.forward_max_iterations)
        if not self.handle:
            raise RuntimeError(self.lib.il_last_error().decode())

    def forward(self, theta, initial, guess):
        theta, initial, guess = array(theta), array(initial), array(guess)
        output = np.empty_like(guess)
        elapsed = ct.c_double()
        code = self.lib.il_forward(self.handle, ptr(theta), ptr(initial), ptr(guess), ptr(output), ct.byref(elapsed))
        if code:
            raise RuntimeError(self.lib.il_last_error().decode())
        self.forward_seconds = elapsed.value
        timing = np.empty(7)
        if self.lib.il_forward_timing(self.handle, ptr(timing)):
            raise RuntimeError("FATROP numerical timing diagnostics unavailable")
        self.forward_kernel_counts = timing[1:6].astype(np.int64)
        self.forward_native_seconds = float(timing[6])
        return output

    def adjoint(self, target, rhs):
        from rapidoc.solver import FrsStats
        if target is not None:
            raise RuntimeError("Ours builds the target operator in the native backward call")
        output = np.empty_like(rhs)
        gradient = np.empty(self.ev.model.d)
        stats, cache = FrsStats(), ct.c_double()
        code = self.lib.il_backward(self.handle, ptr(rhs), self.args.tolerance,
                                    self.args.max_iterations, ptr(output), ptr(gradient),
                                    ct.byref(stats), ct.byref(cache))
        if code:
            raise RuntimeError(self.lib.il_last_error().decode())
        if stats.used_fallback:
            raise RuntimeError("unexpected fresh-factor fallback")
        return (output, gradient, stats.mean_iterations, cache.value,
                stats.elapsed_seconds, stats.target_build_seconds,
                stats.relative_residual)

    def close(self):
        if self.handle:
            self.lib.il_destroy(self.handle)
            self.handle = None


class SafePDP:
    """SafePDP COCsys with supplied primal guesses and full sensitivities."""
    def __init__(self, ev, args):
        self.ev, self.args = ev, args
        from baselines.imitation_ipopt_timing import NativeIpoptTimer
        self.forward_timer = NativeIpoptTimer()
        self.runtime_details = self.forward_timer.runtime_details
        upstream = safepdp_root()
        if not upstream.exists():
            raise FileNotFoundError(f"official SafePDP source missing: {upstream}")
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        from SafePDP import SafePDP as constrained_pdp
        m = ev.model
        self.system = constrained_pdp.COCsys(f"official SafePDP {m.name}")
        self.system.setAuxvarVariable(m.parameters)
        self.system.setStateVariable(m.x)
        self.system.setControlVariable(m.u)
        self.system.setDyn(m.dynamics)
        self.system.setPathCost(m.running)
        self.system.setFinalCost(m.terminal)
        self.system.setPathInequCstr(-m.margins)
        self.system.setFinalInequCstr(-m.terminal_margins)
        self.system.diffCPMP()
        self.lqr = constrained_pdp.EQCLQR()
        self.solution = None
        self.forward_native_seconds = self.backward_native_seconds = float("nan")

    def forward(self, theta, initial, guess):
        self.solution = self.system.ocSolver(
            init_state=np.asarray(initial), horizon=self.ev.N,
            auxvar_value=self.ev.parameters(theta), print_level=0,
            initial_guess=guess,
            solver_options={"ipopt.tol": self.args.forward_tolerance,
                            "ipopt.acceptable_tol": self.args.forward_tolerance,
                            "ipopt.max_iter": self.args.forward_max_iterations},
            solve_timer=self.forward_timer,
        )
        if self.solution.get("timing_protocol") != NATIVE_TIMING_PROTOCOL:
            raise RuntimeError("SafePDP forward requires strict numerical-kernel timing")
        self.forward_seconds = float(self.solution["solve_seconds"])
        # CasADi's t_wall_total measures the nlpsol invocation itself and
        # excludes the Python-side NLP construction and nlpsol creation above.
        self.forward_native_seconds = float(self.solution["native_solve_seconds"])
        return pack(np.asarray(self.solution["state_traj_opt"]),
                    np.asarray(self.solution["control_traj_opt"]))

    def gradient(self, rhs, state_jacobian=None):
        if self.solution is None:
            raise RuntimeError("SafePDP requires a matching official forward solve")
        m, N = self.ev.model, self.ev.N
        target_start = time.perf_counter()
        auxiliary = self.system.getAuxSys(self.solution)
        target_seconds = time.perf_counter()-target_start

        self.lqr.setDyn(
            auxiliary["dynFx_t"], auxiliary["dynFu_t"], auxiliary["dynFe_t"])
        self.lqr.setPathCost(
            auxiliary["Lxx_t"], auxiliary["Lxu_t"], auxiliary["Lxe_t"],
            auxiliary["Lux_t"], auxiliary["Luu_t"], auxiliary["Lue_t"])
        self.lqr.setFinalCost(auxiliary["Lxx_T"], auxiliary["Lxe_T"])
        self.lqr.setPathConstraints(
            auxiliary["GbarHx_t"], auxiliary["GbarHu_t"],
            auxiliary["GbarHe_t"])
        self.lqr.setFinalConstraints(
            auxiliary["GbarHx_T"], auxiliary["GbarHe_T"])
        initial_sensitivity = np.zeros((m.nx, m.parameters.numel()))
        if state_jacobian is not None:
            initial_sensitivity[:, :m.d] = state_jacobian
        solve_started = time.perf_counter()
        sensitivity = self.lqr.eqctlqrSolver(initial_sensitivity, N)
        self.backward_native_seconds = time.perf_counter()-solve_started
        if sensitivity.get("timing_protocol") != NATIVE_TIMING_PROTOCOL:
            raise RuntimeError("SafePDP requires the segmented solve timer")
        linear = float(sensitivity["solve_seconds"])

        contraction_start = time.perf_counter()
        primal_size = N*(m.nx+m.nu)+m.nx
        gx, gu = unpack(rhs[:primal_size], m, N)
        dx = np.asarray(sensitivity["state_traj_opt"])
        du = np.asarray(sensitivity["control_traj_opt"])
        gradient = (np.einsum("ki,kij->j", gx, dx)
                    + np.einsum("ki,kij->j", gu, du))[:m.d]
        self.control_parameter_jacobian = du[0, :, :m.d].copy()
        contraction = time.perf_counter()-contraction_start
        return array(gradient), target_seconds, linear, contraction

    def close(self):
        self.solution = None


class SIPOPT:
    """Native Ipopt nominal + one official sIPOPT auxiliary loss direction."""
    _lib = None
    _directories = []

    @classmethod
    def library(cls):
        if cls._lib is not None:
            return cls._lib
        path = ROOT/"bin"/"imitation_sipopt.dll"
        if not path.exists():
            raise FileNotFoundError("Build the imitation_sipopt target first; see IMITATION_LEARNING.md")
        if hasattr(os, "add_dll_directory"):
            for directory in (ROOT/"bin", ROOT/".sipopt"/"Library"/"bin"):
                cls._directories.append(os.add_dll_directory(str(directory)))
        lib, dp = ct.CDLL(str(path)), ct.POINTER(ct.c_double)
        lib.ils_last_error.restype = ct.c_char_p
        lib.ils_timing_protocol.restype = ct.c_char_p
        if lib.ils_timing_protocol().decode() != NATIVE_TIMING_PROTOCOL:
            raise RuntimeError("Rebuild imitation_sipopt for solve-kernels-v4 timing")
        lib.ils_create.argtypes = [
            ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_int]
        lib.ils_create.restype = ct.c_void_p
        lib.ils_create_closed_loop.argtypes = lib.ils_create.argtypes
        lib.ils_create_closed_loop.restype = ct.c_void_p
        lib.ils_destroy.argtypes = [ct.c_void_p]
        lib.ils_destroy.restype = None
        lib.ils_forward.argtypes = [ct.c_void_p, dp, dp, dp, dp, dp, dp,
                                    ct.POINTER(ct.c_double)]
        lib.ils_forward.restype = ct.c_int
        lib.ils_forward_timing.argtypes = [ct.c_void_p, dp]
        lib.ils_forward_timing.restype = ct.c_int
        lib.ils_backward.argtypes = [ct.c_void_p, dp, dp]
        lib.ils_backward.restype = ct.c_int
        cls._lib = lib
        return lib

    def __init__(self, ev, args):
        self.ev, self.lib = ev, self.library()
        self.runtime_details = {
            "native_diagnostic_timing_protocol": NATIVE_TIMING_PROTOCOL,
            "forward_timer": "private MUMPS numerical and primal-dual arithmetic counters",
            "binaries": {name: {"path": str(ROOT/"bin"/name),
                                 "sha256": digest(ROOT/"bin"/name)}
                         for name in ("imitation_sipopt.dll", "frs_ipopt_highres.dll", "frs_sipopt_highres.dll")},
        }
        self.closed_loop = args.command == "closed-loop"
        create = self.lib.ils_create_closed_loop if self.closed_loop else self.lib.ils_create
        self.handle = create(
            models.NAMES.index(ev.model.name), ev.N, ev.dt,
            args.forward_tolerance, args.forward_max_iterations)
        if not self.handle:
            raise RuntimeError(self.lib.ils_last_error().decode())
        m = ev.model
        self.loss_diagonal = pack(
            np.ones((ev.N+1, m.nx)),
            np.ones((ev.N, m.nu)))
        self.demo = None

    def set_demo(self, X, U=None):
        self.demo = array(X if U is None else pack(X, U))

    def forward(self, theta, initial, guess):
        if self.demo is None:
            raise RuntimeError("sIPOPT needs the current demonstration before the nominal solve")
        theta, initial, guess = array(theta), array(initial), array(guess)
        output = np.empty_like(guess)
        elapsed = ct.c_double()
        code = self.lib.ils_forward(self.handle, ptr(theta), ptr(initial), ptr(guess),
                                    ptr(self.demo), ptr(self.loss_diagonal), ptr(output),
                                    ct.byref(elapsed))
        if code:
            raise RuntimeError(self.lib.ils_last_error().decode())
        self.forward_seconds = elapsed.value
        timing = np.empty(9)
        if self.lib.ils_forward_timing(self.handle, ptr(timing)):
            raise RuntimeError("sIPOPT numerical timing diagnostics unavailable")
        self.forward_kernel_seconds = timing[:4].copy()
        self.forward_kernel_counts = timing[4:8].astype(np.int64)
        self.forward_native_seconds = float(timing[8])
        return output

    def adjoint(self, target, rhs):
        output = (np.empty((1+self.ev.model.nu, rhs.size))
                  if self.closed_loop else np.empty_like(rhs))
        elapsed = ct.c_double()
        code = self.lib.ils_backward(self.handle, ptr(output), ct.byref(elapsed))
        if code:
            raise RuntimeError(self.lib.ils_last_error().decode())
        # Native sensitivity uses its own retained Ipopt state. `target` is not
        # substituted into that state; the common caller independently certifies
        # its returned adjoint and contracts the same parameter derivative.
        return output, 0.0, 0.0, elapsed.value

    def close(self):
        if self.handle:
            self.lib.ils_destroy(self.handle)
            self.handle = None


def apply_target(target, value, model, N):
    H, HT, A = target[:3]
    size = N*(model.nx+model.nu)+model.nx
    ax, au = unpack(value[:size], model, N)
    lam = value[size:].reshape(N+1, model.nx)
    primal, equalities = [], [ax[0]]
    for k in range(N):
        z = np.r_[ax[k], au[k]]
        out = H[k]@z+A[k].T@lam[k+1]
        out[:model.nx] += lam[k]
        primal.append(out)
        equalities.append(A[k]@z+ax[k+1])
    return np.concatenate((*primal, HT@ax[-1]+lam[-1], *equalities))


def imitation_seed(X, U, demoX, demoU, model):
    # LAPANDA-style trajectory loss: sum over the complete state-control rollout.
    # Averaging over demonstrations is performed by the outer training loop.
    dx, du = X-demoX, U-demoU
    loss = 0.5*(np.sum(dx*dx)+np.sum(du*du))
    gx, gu = dx, du
    primal = pack(gx, gu)
    return float(loss), array(np.r_[primal, np.zeros((U.shape[0]+1)*model.nx)])


def make_adapter(method, ev, args):
    if method == "CasADi":
        # Optional dependency instrumentation must not affect the original three.
        from baselines.imitation_casadi import CasADi
        return CasADi(ev, args)
    if method == "acados":
        from baselines.imitation_acados import Acados
        return Acados(ev, args)
    return {"Ours": Ours, "SafePDP": SafePDP, "sIPOPT": SIPOPT}[method](ev, args)


def adapter_runtime_details(adapter, method):
    details = dict(getattr(adapter, "runtime_details", {
        "derivative_backend": getattr(adapter, "derivative_backend", method),
        "casadi_version": ca.__version__,
        "casadi_package": str(Path(ca.__file__).resolve().parent),
    }))
    if "timing_protocol" in details:
        details.setdefault("native_diagnostic_timing_protocol",
                           details.pop("timing_protocol"))
    details["reported_timing_protocol"] = TIMING_PROTOCOL
    return details


def closed_loop_derivatives(ev, adapter, theta, X, U, rhs, state_jacobian):
    """Differentiate the local loss and first control through the incoming state."""
    model = ev.model
    target = cache = linear = contraction = 0.0
    iterations, residual = 0.0, float("nan")
    if callable(getattr(adapter, "gradient", None)):
        gradient, target, linear, contraction = adapter.gradient(rhs, state_jacobian)
        control_jacobian = adapter.control_parameter_jacobian
    else:
        primal_size = ev.N*(model.nx+model.nu)+model.nx
        if isinstance(adapter, Ours):
            adjoints, parameter_gradients, residuals = [], [], []
            for direction in range(1+model.nu):
                seed = rhs if direction == 0 else np.zeros_like(rhs)
                if direction:
                    seed[model.nx+direction-1] = 1.0
                adjoint, grad, count, prep, solve, build, error = adapter.adjoint(None, seed)
                adjoints.append(adjoint)
                parameter_gradients.append(grad)
                iterations += count
                cache += prep
                linear += solve
                target += build
                residuals.append(error)
            residual = max(residuals)
            adjoints, parameter_gradients = np.asarray(adjoints), np.asarray(parameter_gradients)
        else:
            if not adapter.closed_loop:
                raise RuntimeError("closed-loop sIPOPT requires controller sensitivity directions")
            adjoints, iterations, cache, linear = adapter.adjoint(None, rhs)
            started = time.perf_counter()
            mixed, mixedT = ev.parameter_derivatives(X, U, theta)
            target = time.perf_counter()-started
            started = time.perf_counter()
            parameter_gradients = []
            for adjoint in adjoints:
                ax, au = unpack(adjoint[:primal_size], model, ev.N)
                directions = np.concatenate((ax[:-1], au), axis=1)
                parameter_gradients.append(
                    -np.einsum("ki,kid->d", directions, mixed)-ax[-1]@mixedT)
            parameter_gradients = np.asarray(parameter_gradients)
            contraction += time.perf_counter()-started
        started = time.perf_counter()
        # The initial equation is X[0]-state=0; its adjoint block is dL/dstate.
        state_gradients = adjoints[:, primal_size:primal_size+model.nx]
        total_gradients = parameter_gradients + state_gradients @ state_jacobian
        gradient, control_jacobian = total_gradients[0], total_gradients[1:]
        contraction += time.perf_counter()-started
    return gradient, control_jacobian, target, cache, linear, contraction, iterations, residual


def evaluate(ev, adapter, theta, initial, guess, demoX, demoU, args,
             gradient=True, packed_demo=None, state_jacobian=None):
    if isinstance(adapter, SIPOPT):
        adapter.set_demo(packed_demo if packed_demo is not None else pack(demoX, demoU))
    forward_started = time.perf_counter()
    w = adapter.forward(theta, initial, guess)
    forward_wall = time.perf_counter()-forward_started
    forward = float(getattr(adapter, "forward_native_seconds", forward_wall))
    if not np.isfinite(forward) or forward <= 0 or forward > forward_wall*1.05+1e-6:
        raise RuntimeError("invalid native forward-solver timing")
    X, U = unpack(w, ev.model, ev.N)
    margin = ev.margin(X, U, theta)
    backward_default = 0.0 if gradient else float("nan")
    metrics = dict(forward_ms=1e3*forward, backward_ms=backward_default,
                   total_ms=(1e3*forward if gradient else float("nan")),
                   target_ms=0.0,
                    cache_ms=0.0, linear_ms=0.0, contraction_ms=0.0,
                   mean_iterations=0.0, max_relative_residual=0.0, min_margin=margin)
    grad = np.zeros(ev.model.d)
    if gradient:
        # The outer wall clock is retained only to validate the narrower native
        # solver-call measurement reported below.
        backward_started = time.perf_counter()
        loss, rhs = imitation_seed(X, U, demoX, demoU, ev.model)
        contraction = target_seconds = cache = linear = 0.0
        if state_jacobian is not None:
            (grad, control_jacobian, target_seconds, cache, linear, contraction,
             iterations, residual) = closed_loop_derivatives(
                ev, adapter, theta, X, U, rhs, state_jacobian)
            metrics["control_parameter_jacobian"] = control_jacobian
        elif isinstance(adapter, Ours):
            (adjoint, grad, iterations, cache, linear,
             target_seconds, residual) = adapter.adjoint(None, rhs)
        elif callable(getattr(adapter, "gradient", None)):
            grad, target_seconds, linear, contraction = adapter.gradient(rhs)
            iterations, residual = 0.0, float("nan")
        else:
            adjoint, iterations, cache, linear = adapter.adjoint(None, rhs)
            target_start = time.perf_counter()
            mixed, mixedT = ev.parameter_derivatives(X, U, theta)
            target_seconds = time.perf_counter()-target_start
            t = time.perf_counter()
            ax, au = unpack(adjoint[:w.size], ev.model, ev.N)
            directions = np.concatenate((ax[:-1], au), axis=1)
            grad = -np.einsum("ki,kid->d", directions, mixed)-ax[-1]@mixedT
            contraction = time.perf_counter()-t
            residual = float("nan")
        if isinstance(adapter, Ours) and (
                not np.isfinite(residual) or residual > args.tolerance):
            raise RuntimeError(f"target adjoint residual {residual:.3e} exceeds {args.tolerance:.3e}")
        if not np.all(np.isfinite(grad)):
            raise RuntimeError("nonfinite loss gradient")
        backward_wall = time.perf_counter()-backward_started
        backward = float(getattr(adapter, "backward_native_seconds", linear))
        if not np.isfinite(backward) or backward <= 0 or backward > backward_wall*1.05+1e-6:
            raise RuntimeError("invalid native backward-solver timing")
        metrics.update(target_ms=1e3*target_seconds, contraction_ms=1e3*contraction,
                         backward_ms=1e3*backward,
                        cache_ms=1e3*cache, linear_ms=1e3*linear,
                       mean_iterations=iterations, max_relative_residual=float(residual))
    else:
        loss, _ = imitation_seed(X, U, demoX, demoU, ev.model)
    metrics["total_ms"] = metrics["forward_ms"]+metrics["backward_ms"]
    return loss, grad, w, metrics


def dataset_config(model, args):
    return dict(version=models.VERSION, model_sha256=digest(Path(models.__file__)),
                problem=model.name, N=args.N, dt=args.dt,
                train_demos=args.train_demos, validation_demos=args.validation_demos,
                data_seed=args.data_seed, teacher_tolerance=args.teacher_tolerance,
                teacher_theta=model.teacher_theta.tolist(),
                teacher="IPOPT, identical explicitly constrained OCP",
                imitation_loss="half squared state-control trajectory error; no horizon or dimension normalization")


def matches_frozen_far_dataset(stored, expected):
    # The recorded v4 -> v5 patch changes only the initial-state sampler and
    # VERSION. Frozen files already contain their initial states and teacher
    # trajectories. Pin both source hashes so later model edits fail closed.
    if (stored.get("version"), stored.get("model_sha256")) != FROZEN_FAR_MODEL:
        return False
    if (expected.get("version"), expected.get("model_sha256")) != FROZEN_FAR_COMPATIBLE_MODEL:
        return False
    comparable = dict(stored)
    comparable.update(version=expected["version"], model_sha256=expected["model_sha256"])
    return comparable == expected


def dataset(ev, args):
    path = args.dataset_dir/(ev.model.name+".npz")
    expected = dataset_config(ev.model, args)
    reuse_far = getattr(args, "reuse_frozen_far", False)
    if reuse_far and (args.command != "run" or args.regenerate):
        raise ValueError("--reuse-frozen-far requires run without --regenerate")
    if path.exists() and not args.regenerate:
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(data["metadata"]))
            if reuse_far:
                if not matches_frozen_far_dataset(stored, expected):
                    raise RuntimeError(
                        f"frozen far compatibility check failed: {path}; "
                        "both model hashes and all other dataset settings must match")
                print(f"Reusing unchanged frozen far data for {', '.join(args.methods)}: {path}", flush=True)
            elif stored != expected:
                raise RuntimeError(f"dataset configuration differs: {path}; choose a new dataset directory")
            return {key: np.array(data[key]) for key in ("initial", "X", "U", "split")}
    if reuse_far:
        raise FileNotFoundError(f"frozen far data must already exist; no data will be generated: {path}")
    count = args.train_demos+args.validation_demos
    initial = ev.model.initial_conditions(count, args.data_seed+1009*models.NAMES.index(ev.model.name))
    teacher = IpoptForward(ev, args.teacher_tolerance, args.forward_max_iterations)
    X, U = [], []
    # No rejection sampling to retain only easy/successful teacher instances.
    for index, x0 in enumerate(initial):
        try:
            w = teacher.forward(ev.model.teacher_theta, x0, ev.guess(x0))
            x, u = unpack(w, ev.model, ev.N)
            ev.margin(x, u, ev.model.teacher_theta)
        except Exception as error:
            raise RuntimeError(f"teacher {ev.model.name} instance {index}: {error}") from error
        X.append(x); U.append(u)
    result = dict(initial=initial, X=np.asarray(X), U=np.asarray(U),
                  split=np.r_[np.zeros(args.train_demos, dtype=int), np.ones(args.validation_demos, dtype=int)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result, metadata=np.asarray(json.dumps(expected, sort_keys=True)))
    print(f"Frozen teacher data: {path}", flush=True)
    return result


def train(ev, data, method, seed, args):
    folder = args.output_dir/ev.model.name/method/f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/"history.csv"
    checkpoint = folder/"checkpoint.npz"
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"not overwriting {output}; use --overwrite or a new output directory")
    adapter = None
    status = dict(problem=ev.model.name, method=method, seed=seed, status="failed", last_epoch=-1)
    if args.resume:
        legacy = _handle_legacy_resume(folder, output, checkpoint, args.epochs)
        if legacy is not None:
            print(f"Skipping completed legacy {ev.model.name}/{method}/seed {seed}", flush=True)
            return legacy
    packed_demos = ([pack(x, u) for x, u in zip(data["X"], data["U"])]
                    if method == "sIPOPT" else None)
    identity = _checkpoint_identity(ev, method, seed, args, "open-loop")
    start_epoch = 0
    if args.resume and checkpoint.exists():
        saved = _load_checkpoint(checkpoint, identity)
        start_epoch = int(saved["next_epoch"].item())
        theta = saved["theta"]
        m, v = saved["adam_m"], saved["adam_v"]
        previous = [row.copy() for row in saved["previous"]]
        train_seconds = float(saved["train_seconds"].item())
        if len(previous) != len(data["initial"]):
            raise RuntimeError(f"checkpoint sample count differs: {checkpoint}")
        _trim_history(output, start_epoch, FIELDS)
        print(f"Resuming {ev.model.name}/{method}/seed {seed} at epoch {start_epoch}", flush=True)
    else:
        if args.resume and output.exists():
            raise RuntimeError(
                f"cannot resume {output}: checkpoint.npz is missing; use --overwrite to restart")
        rng = np.random.default_rng(60013+seed)
        theta = rng.normal(0, 0.15, ev.model.d)
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        previous = [ev.guess(x0) for x0 in data["initial"]]
        train_seconds = 0.0
        np.save(folder/"initial_theta.npy", theta)
    if start_epoch > args.epochs:
        status.update(status="ok", last_epoch=args.epochs, train_seconds=train_seconds,
                      skipped_completed=True)
        print(f"Skipping completed {ev.model.name}/{method}/seed {seed}", flush=True)
        (folder/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status
    try:
        adapter = make_adapter(method, ev, args)
        status["runtime_details"] = adapter_runtime_details(adapter, method)
        mode = "a" if start_epoch else "w"
        with output.open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not start_epoch:
                writer.writeheader()
            for epoch in range(start_epoch, args.epochs+1):
                start = time.perf_counter()
                losses, grads, records = [], [], []
                for index in range(args.train_demos):
                    value, gradient, w, record = evaluate(
                        ev, adapter, theta, data["initial"][index], previous[index],
                        data["X"][index], data["U"][index], args,
                        gradient=epoch<args.epochs,
                        packed_demo=None if packed_demos is None else packed_demos[index])
                    previous[index] = w
                    losses.append(value); grads.append(gradient); records.append(record)
                grad = np.mean(grads, axis=0)
                next_theta = theta.copy()
                if epoch < args.epochs:
                    # Same Adam update, initial parameters and projection for every method.
                    m = 0.9*m+0.1*grad
                    v = 0.999*v+0.001*grad*grad
                    next_theta -= args.learning_rate*(m/(1-0.9**(epoch+1)))/(np.sqrt(v/(1-0.999**(epoch+1)))+1e-8)
                    next_theta = np.clip(next_theta, -2, 2)
                train_seconds += time.perf_counter()-start
                validation, validation_seconds = "", 0.0
                if epoch % args.validation_every == 0 or epoch==args.epochs:
                    start = time.perf_counter()
                    heldout = []
                    for index in range(args.train_demos, len(previous)):
                        loss, _, w, _ = evaluate(ev, adapter, theta, data["initial"][index], previous[index],
                                                 data["X"][index], data["U"][index], args,
                                                 gradient=False,
                                                 packed_demo=None if packed_demos is None else packed_demos[index])
                        previous[index] = w
                        heldout.append(loss)
                    validation = float(np.mean(heldout))
                    validation_seconds = time.perf_counter()-start
                row = dict(problem=ev.model.name, method=method, seed=seed, epoch=epoch,
                           train_loss=float(np.mean(losses)), validation_loss=validation,
                           train_seconds=train_seconds, validation_seconds=validation_seconds,
                           gradient_norm=float(np.linalg.norm(grad)), theta=json.dumps(theta.tolist()))
                for key in ("forward_ms", "backward_ms", "total_ms", "target_ms",
                            "cache_ms", "linear_ms", "contraction_ms"):
                    row[key] = float(np.mean([r[key] for r in records]))
                row["mean_iterations"] = float(np.mean([r["mean_iterations"] for r in records]))
                row["max_relative_residual"] = max(r["max_relative_residual"] for r in records)
                row["min_margin"] = min(r["min_margin"] for r in records)
                row["timing_protocol"] = TIMING_PROTOCOL
                writer.writerow(row); stream.flush()
                status["last_epoch"] = epoch
                if epoch % args.print_every == 0 or epoch==args.epochs:
                    print(f"{ev.model.name:10s} {method:7s} seed={seed:02d} epoch={epoch:04d} "
                          f"loss={row['train_loss']:.4e} time={train_seconds:.2f}s", flush=True)
                theta = next_theta
                np.save(folder/"theta.npy", theta)
                if (epoch == 0 or (epoch+1) % args.checkpoint_every == 0
                        or epoch == args.epochs):
                    _atomic_savez(
                        checkpoint,
                        identity=np.asarray(json.dumps(identity, sort_keys=True)),
                        next_epoch=np.asarray(epoch+1),
                        theta=theta,
                        adam_m=m,
                        adam_v=v,
                        previous=np.asarray(previous),
                        train_seconds=np.asarray(train_seconds),
                    )
        np.savez_compressed(folder/"final_predictions.npz", trajectory=np.asarray(previous),
                            theta=theta, initial=data["initial"], split=data["split"])
        status.update(status="ok", train_seconds=train_seconds)
    except Exception as error:
        status["message"] = str(error)
        print(f"FAILED {ev.model.name} {method} seed={seed}: {error}", flush=True)
    finally:
        if adapter is not None:
            adapter.close()
        (folder/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def shifted_guess(w, ev, next_state):
    """Shift one MPC solution by one stage without introducing a new solve."""
    X, U = unpack(w, ev.model, ev.N)
    shifted_X = np.vstack((np.asarray(next_state), X[2:], X[-1]))
    shifted_U = np.vstack((U[1:], U[-1]))
    return pack(shifted_X, shifted_U)


def closed_loop_metric(problem, state, planned_controls):
    """Task quantity displayed beside the closed-loop learning curve."""
    state = np.asarray(state, dtype=float)
    if problem == "cartpole":
        return float(np.sum(np.asarray(planned_controls)[:, 0]**2))
    if problem == "quadrotor":
        return float(state[1])
    q1, q2 = state
    end_effector = np.array([
        np.cos(q1) + 0.8*np.cos(q1+q2),
        np.sin(q1) + 0.8*np.sin(q1+q2),
    ])
    return float(np.linalg.norm(end_effector-np.array([1.3, 0.3])))


def closed_loop_reference(ev, data, args):
    """Create one frozen teacher MPC rollout shared by every method and seed."""
    path = args.dataset_dir/f"{ev.model.name}_closed_loop_{args.closed_loop_steps}.npz"
    expected = dataset_config(ev.model, args)
    expected.update(protocol="closed-loop-v2-lapanda-scale", steps=args.closed_loop_steps,
                    initial_source="open-loop training instance 0")
    if path.exists() and not args.regenerate:
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored["metadata"])) != expected:
                raise RuntimeError(f"closed-loop dataset configuration differs: {path}")
            return {key: np.array(stored[key]) for key in
                    ("initial", "states", "X", "U", "metric")}

    teacher = IpoptForward(ev, args.teacher_tolerance, args.forward_max_iterations)
    state = np.asarray(data["initial"][0], dtype=float).copy()
    initial = state.copy()
    guess = ev.guess(state)
    states, predictions_X, predictions_U, metric = [state.copy()], [], [], []
    try:
        for step in range(args.closed_loop_steps):
            try:
                w = teacher.forward(ev.model.teacher_theta, state, guess)
                X, U = unpack(w, ev.model, ev.N)
                ev.margin(X, U, ev.model.teacher_theta)
            except Exception as error:
                raise RuntimeError(
                    f"closed-loop teacher {ev.model.name} step {step}: {error}"
                ) from error
            predictions_X.append(X.copy())
            predictions_U.append(U.copy())
            metric.append(closed_loop_metric(ev.model.name, state, U))
            next_state = ev.value("f", state, U[0], ev.model.teacher_theta).ravel()
            guess = shifted_guess(w, ev, next_state)
            state = next_state
            states.append(state.copy())
    finally:
        close = getattr(teacher, "close", None)
        if close is not None:
            close()

    result = dict(initial=initial, states=np.asarray(states),
                  X=np.asarray(predictions_X), U=np.asarray(predictions_U),
                  metric=np.asarray(metric))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result,
                        metadata=np.asarray(json.dumps(expected, sort_keys=True)))
    print(f"Frozen closed-loop teacher data: {path}", flush=True)
    return result


def closed_loop_rollout(ev, reference, adapter, theta, args, gradient,
                        packed_reference=None, epoch=None):
    """Differentiate the mean prediction loss through the complete MPC rollout."""
    state = np.asarray(reference["initial"], dtype=float).copy()
    guess = ev.guess(state)
    states, controls, metric = [state.copy()], [], []
    losses, gradients, records = [], [], []
    state_jacobian = np.zeros((ev.model.nx, ev.model.d)) if gradient else None
    for step in range(args.closed_loop_steps):
        try:
            value, grad, w, record = evaluate(
                ev, adapter, theta, state, guess,
                reference["X"][step], reference["U"][step], args,
                gradient=gradient,
                packed_demo=None if packed_reference is None else packed_reference[step],
                state_jacobian=state_jacobian,
            )
        except Exception as error:
            raise RuntimeError(
                f"epoch={epoch}, rollout_step={step}: {error}"
            ) from error
        X, U = unpack(w, ev.model, ev.N)
        losses.append(value)
        gradients.append(grad)
        records.append(record)
        controls.append(U[0].copy())
        metric.append(closed_loop_metric(ev.model.name, state, U))
        next_state = ev.value("f", state, U[0], theta).ravel()
        if gradient:
            plant_jacobian = ev.value("jac", state, U[0], theta)
            explicit_parameters = np.asarray(ev.plant_parameter_jacobian(
                state, U[0], ev.parameters(theta)), dtype=float)
            state_jacobian = (
                plant_jacobian[:, :ev.model.nx] @ state_jacobian
                + plant_jacobian[:, ev.model.nx:] @ record["control_parameter_jacobian"]
                + explicit_parameters)
            if not np.all(np.isfinite(state_jacobian)):
                raise RuntimeError(f"epoch={epoch}, rollout_step={step}: nonfinite rollout sensitivity")
        guess = shifted_guess(w, ev, next_state)
        state = next_state
        states.append(state.copy())
    return dict(
        loss=float(np.mean(losses)),
        gradient=np.mean(gradients, axis=0),
        states=np.asarray(states),
        controls=np.asarray(controls),
        metric=np.asarray(metric),
        records=records,
    )


def train_closed_loop(ev, reference, method, seed, args):
    """Train one parameter vector through a 50-step MPC rollout."""
    folder = args.output_dir/ev.model.name/method/f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/"history.csv"
    checkpoint = folder/"checkpoint.npz"
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"not overwriting {output}; use --overwrite or a new output directory")

    fields = ("timing_protocol", "problem", "method", "seed", "epoch", "loss", "train_seconds",
              "forward_ms", "backward_ms", "total_ms", "mean_iterations",
              "target_ms", "cache_ms", "linear_ms", "contraction_ms",
              "max_relative_residual", "min_margin", "gradient_norm", "theta")
    packed_reference = ([pack(x, u) for x, u in zip(reference["X"], reference["U"])]
                        if method == "sIPOPT" else None)
    adapter = None
    status = dict(problem=ev.model.name, method=method, seed=seed,
                  status="failed", last_epoch=-1)
    if args.resume:
        legacy = _handle_legacy_resume(
            folder, output, checkpoint, args.closed_loop_epochs)
        if legacy is not None:
            print(f"Skipping completed legacy closed-loop "
                  f"{ev.model.name}/{method}/seed {seed}", flush=True)
            return legacy
    identity = _checkpoint_identity(ev, method, seed, args, "closed-loop")
    start_epoch = 0
    if args.resume and checkpoint.exists():
        saved = _load_checkpoint(checkpoint, identity)
        start_epoch = int(saved["next_epoch"].item())
        theta = saved["theta"]
        m, v = saved["adam_m"], saved["adam_v"]
        elapsed = float(saved["train_seconds"].item())
        _trim_history(output, start_epoch, fields)
        print(f"Resuming closed-loop {ev.model.name}/{method}/seed {seed} "
              f"at epoch {start_epoch}", flush=True)
    else:
        if args.resume and output.exists():
            raise RuntimeError(
                f"cannot resume {output}: checkpoint.npz is missing; use --overwrite to restart")
        rng = np.random.default_rng(60013+seed)
        theta = rng.normal(0, 0.15, ev.model.d)
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        elapsed = 0.0
        np.save(folder/"initial_theta.npy", theta)
    if start_epoch > args.closed_loop_epochs:
        status.update(status="ok", last_epoch=args.closed_loop_epochs,
                      train_seconds=elapsed, skipped_completed=True)
        print(f"Skipping completed closed-loop {ev.model.name}/{method}/seed {seed}", flush=True)
        (folder/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status
    try:
        adapter = make_adapter(method, ev, args)
        status["runtime_details"] = adapter_runtime_details(adapter, method)
        mode = "a" if start_epoch else "w"
        with output.open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not start_epoch:
                writer.writeheader()
            for epoch in range(start_epoch, args.closed_loop_epochs+1):
                start = time.perf_counter()
                result = closed_loop_rollout(
                    ev, reference, adapter, theta, args,
                    gradient=epoch < args.closed_loop_epochs,
                    packed_reference=packed_reference,
                    epoch=epoch,
                )
                elapsed += time.perf_counter()-start
                grad = result["gradient"]
                next_theta = theta.copy()
                if epoch < args.closed_loop_epochs:
                    m = 0.9*m+0.1*grad
                    v = 0.999*v+0.001*grad*grad
                    next_theta -= args.closed_loop_learning_rate*\
                        (m/(1-0.9**(epoch+1)))/(np.sqrt(v/(1-0.999**(epoch+1)))+1e-8)
                    next_theta = np.clip(next_theta, -2, 2)

                records = result["records"]
                row = dict(
                    problem=ev.model.name, method=method, seed=seed, epoch=epoch,
                    loss=result["loss"], train_seconds=elapsed,
                    forward_ms=float(np.mean([r["forward_ms"] for r in records])),
                    backward_ms=float(np.mean([r["backward_ms"] for r in records])),
                    total_ms=float(np.mean([r["total_ms"] for r in records])),
                    target_ms=float(np.mean([r["target_ms"] for r in records])),
                    cache_ms=float(np.mean([r["cache_ms"] for r in records])),
                    linear_ms=float(np.mean([r["linear_ms"] for r in records])),
                    contraction_ms=float(np.mean([r["contraction_ms"] for r in records])),
                    mean_iterations=float(np.mean([r["mean_iterations"] for r in records])),
                    max_relative_residual=max(r["max_relative_residual"] for r in records),
                    min_margin=min(r["min_margin"] for r in records),
                    gradient_norm=float(np.linalg.norm(grad)),
                    theta=json.dumps(theta.tolist()),
                )
                row["timing_protocol"] = TIMING_PROTOCOL
                writer.writerow(row)
                stream.flush()
                status["last_epoch"] = epoch
                if epoch == 0:
                    np.savez_compressed(folder/"policy_initial.npz",
                                        states=result["states"], controls=result["controls"],
                                        metric=result["metric"], theta=theta)
                if epoch == args.closed_loop_epochs:
                    np.savez_compressed(folder/"policy_final.npz",
                                        states=result["states"], controls=result["controls"],
                                        metric=result["metric"], theta=theta)
                if epoch % args.print_every == 0 or epoch == args.closed_loop_epochs:
                    print(f"{ev.model.name:10s} {method:7s} closed-loop seed={seed:02d} "
                          f"epoch={epoch:04d} loss={row['loss']:.4e} "
                          f"time={elapsed:.2f}s", flush=True)
                theta = next_theta
                np.save(folder/"theta.npy", theta)
                if (epoch == 0 or (epoch+1) % args.checkpoint_every == 0
                        or epoch == args.closed_loop_epochs):
                    _atomic_savez(
                        checkpoint,
                        identity=np.asarray(json.dumps(identity, sort_keys=True)),
                        next_epoch=np.asarray(epoch+1),
                        theta=theta,
                        adam_m=m,
                        adam_v=v,
                        train_seconds=np.asarray(elapsed),
                    )
        status.update(status="ok", train_seconds=elapsed)
    except Exception as error:
        if "epoch" in locals():
            status["failed_epoch"] = epoch
        status["message"] = str(error)
        print(f"FAILED closed-loop {ev.model.name} {method} seed={seed}: {error}", flush=True)
    finally:
        if adapter is not None:
            adapter.close()
        (folder/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _apply_ablation_rc(plt):
    """Match the typography and neutral background of the ablation figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _apply_ablation_axes(fig, axes):
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(False)
        axis.grid(axis="y", which="major", color="#B8B8B8",
                  linestyle="--", linewidth=0.55, alpha=0.45)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.75)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D

    _apply_ablation_rc(plt)

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    # Keep the paper palette fixed across every experiment. The blue and
    # orange are the previously selected third and eighth palette colors.
    colors, line_styles, markers = METHOD_COLORS, METHOD_LINES, METHOD_MARKERS
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.8,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(
        2, len(args.problems), figsize=(7.14, 4.05),
        squeeze=False,
    )
    loss_axes, timing_axes = axes[0], axes[1]
    summaries = []
    timing_data = {}
    timing_label_groups = []
    for col, problem in enumerate(args.problems):
        ax = loss_axes[col]
        for method_index, method in enumerate(args.methods):
            runs, statuses = [], []
            for seed in args.seeds:
                folder = args.output_dir/problem/method/f"seed_{seed:03d}"
                if not (folder/"status.json").exists():
                    continue
                status = json.loads((folder/"status.json").read_text(encoding="utf-8"))
                statuses.append(status)
                if not (folder/"history.csv").exists():
                    continue
                with (folder/"history.csv").open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    if any(row.get("timing_protocol") != TIMING_PROTOCOL for row in rows):
                        raise RuntimeError(
                            f"Old/mixed timing in {folder}; rerun the selected methods "
                            "in a new results directory before plotting solve times")
                    runs.append(rows)
                    summaries.append(dict(problem=problem, method=method, seed=seed,
                                          status=status["status"], epoch=rows[-1]["epoch"],
                                          train_loss=rows[-1]["train_loss"], validation_loss=rows[-1]["validation_loss"],
                                          train_seconds=rows[-1]["train_seconds"]))
            if not runs:
                continue
            # Each history row stores the mean solver time over the training
            # batch at one iteration.  Keep these iteration-level observations
            # so a single-seed run can still show its temporal timing spread.
            timing_data[(problem, method)] = {
                key: np.asarray([
                    value
                    for rows in runs
                    for row in rows
                    if np.isfinite(value := float(row[key]))
                ])
                for key in ("forward_ms", "backward_ms")
            }
            # Aggregate only the common observed prefix. Do not carry a failed
            # curve forward or silently treat its disappearance as convergence.
            length = min(map(len, runs))
            losses = np.array([[float(r["train_loss"]) for r in rows[:length]] for rows in runs])
            med, low, high = np.percentile(losses, [50,25,75], axis=0)
            epochs = np.asarray([int(row["epoch"]) for row in runs[0][:length]])
            ax.plot(
                epochs, med, color=colors[method],
                linewidth=2.0, linestyle=line_styles[method],
                marker=markers[method], markevery=(15*method_index, 60),
                markersize=3.8, markerfacecolor="white",
                markeredgecolor=colors[method], markeredgewidth=0.9,
                solid_capstyle="round", alpha=0.82,
                label=method_display_name(method),
            )
            ax.fill_between(
                epochs, low, high,
                color=colors[method], alpha=0.07, linewidth=0,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylim(bottom=0)
        ax.set_facecolor("white")
        ax.grid(True, color="#E3E3E3", linewidth=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        timing_axis = timing_axes[col]
        label_groups = [[], []]
        locations = np.arange(2)
        width = 0.92/max(1, len(args.methods))
        for method_index, method in enumerate(args.methods):
            values = timing_data.get((problem, method))
            if values is None:
                continue
            distribution = [values[key] for key in ("forward_ms", "backward_ms")]
            median = np.asarray([np.nanmedian(value) for value in distribution])
            lower = np.asarray([np.nanpercentile(value, 25) for value in distribution])
            upper = np.asarray([np.nanpercentile(value, 75) for value in distribution])
            offset = (method_index-(len(args.methods)-1)/2)*width
            errorbar_options = {}
            if all(value.size > 1 for value in distribution):
                errorbar_options = {
                    "yerr": np.vstack((median-lower, upper-median)),
                    "capsize": 2.5,
                    "error_kw": {
                        "ecolor": colors[method],
                        "elinewidth": 0.9,
                        "capthick": 0.9,
                    },
                }
            bars = timing_axis.bar(
                locations+offset, median, width=width*0.72,
                color=mcolors.to_rgba(colors[method], 0.16),
                linewidth=1.0, edgecolor=colors[method],
                **errorbar_options,
            )
            labels = [f"{value:.2f}" for value in median]
            for group_index, (bar, label, top) in enumerate(zip(bars, labels, upper)):
                annotation = timing_axis.annotate(
                    label,
                    (bar.get_x()+bar.get_width()/2, top),
                    xytext=(0.0, 2.0),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.3,
                    color="#222222", zorder=5, clip_on=False,
                )
                label_groups[group_index].append(annotation)
        timing_axis.margins(x=0.12, y=0.30)
        timing_axis.set_ylim(bottom=0)
        timing_axis.set_xticks(locations, ("Fwd", "Bwd"))
        timing_axis.set_ylabel("Time (ms)")
        timing_axis.set_facecolor("white")
        timing_axis.grid(True, axis="y", color="#E3E3E3", linewidth=0.65)
        timing_axis.set_axisbelow(True)
        for spine in timing_axis.spines.values():
            spine.set_linewidth(0.8)
        timing_label_groups.append(label_groups)
    loss_axes[0].set_ylabel("Imitation loss")
    _apply_ablation_axes(fig, [*loss_axes, *timing_axes])
    legend_handles = [
        Line2D(
            [0], [0], color=colors[method], linewidth=2.0,
            linestyle=line_styles[method], marker=markers[method],
            markersize=4, markerfacecolor="white",
            label=method_display_name(method),
        )
        for method in args.methods
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        frameon=False, bbox_to_anchor=(0.5, 1.01), handlelength=2.2,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93), w_pad=1.0, h_pad=1.1)
    problem_titles = {
        "cartpole": "CartPole",
        "quadrotor": "Quadrotor",
        "robotarm": "Robot Arm",
    }
    for index, (problem, axis) in enumerate(zip(args.problems, timing_axes)):
        bounds = axis.get_position()
        fig.text(
            0.5 * (bounds.x0 + bounds.x1),
            0.012,
            f"({chr(ord('a') + index)}) {problem_titles[problem]}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    # Keep every value centered on its bar.  Resolve only actual text
    # collisions by moving the later label upward in small vertical steps.
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moved = False
        for label_groups in timing_label_groups:
            for group in label_groups:
                boxes = []
                for annotation in group:
                    box = annotation.get_window_extent(renderer).expanded(1.01, 1.02)
                    while any(box.overlaps(previous) for previous in boxes):
                        offset_x, offset_y = annotation.get_position()
                        annotation.set_position((offset_x, offset_y + 1.5))
                        fig.canvas.draw()
                        renderer = fig.canvas.get_renderer()
                        box = annotation.get_window_extent(renderer).expanded(1.01, 1.02)
                        moved = True
                    boxes.append(box)
        if not moved:
            break
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.figure_dir/f"imitation_learning.{extension}", dpi=300)
    plt.close(fig)
    if summaries:
        with (args.figure_dir/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)


def plot_closed_loop(args):
    """Plot closed-loop learning and policy evolution in the LAPANDA layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _apply_ablation_rc(plt)

    colors, line_styles, markers = METHOD_COLORS, METHOD_LINES, METHOD_MARKERS
    policy_colors = {"Teacher": "#222222", "Epoch 0": "#B7B7B7", "Final": "#F03A3A"}
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.8, "axes.labelsize": 9.0,
        "axes.titlesize": 9.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2*len(args.problems),
                             figsize=(7.14, 2.15), squeeze=False)
    axes = axes[0]
    metric_labels = {"cartpole": "Control energy", "quadrotor": "Altitude",
                     "robotarm": "Clearance"}
    titles = {"cartpole": "Cartpole", "quadrotor": "Quadrotor",
              "robotarm": "Robot Arm"}

    for column, problem in enumerate(args.problems):
        loss_axis, policy_axis = axes[2*column], axes[2*column+1]
        for method_index, method in enumerate(args.methods):
            histories = []
            for seed in args.seeds:
                path = args.output_dir/problem/method/f"seed_{seed:03d}"/"history.csv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    histories.append(rows)
            if not histories:
                continue
            length = min(map(len, histories))
            values = np.asarray([[float(row["loss"]) for row in rows[:length]]
                                 for rows in histories])
            median, lower, upper = np.nanpercentile(values, [50, 25, 75], axis=0)
            epochs = np.asarray([int(row["epoch"]) for row in histories[0][:length]])
            loss_axis.plot(
                epochs, median, color=colors[method], linewidth=1.9,
                linestyle=line_styles[method], marker=markers[method],
                markevery=(10*method_index, 50), markersize=3.4,
                markerfacecolor="white", markeredgecolor=colors[method],
                markeredgewidth=0.8, alpha=0.82,
                label=method_display_name(method),
            )
            loss_axis.fill_between(epochs, lower, upper, color=colors[method],
                                   alpha=0.07, linewidth=0)

        loss_axis.set_ylim(bottom=0)

        reference_path = args.dataset_dir/f"{problem}_closed_loop_{args.closed_loop_steps}.npz"
        if reference_path.exists():
            with np.load(reference_path, allow_pickle=False) as reference:
                target_metric = np.asarray(reference["metric"])
            steps = np.arange(target_metric.size)
            policy_axis.plot(steps, target_metric, color=policy_colors["Teacher"], linewidth=1.8,
                             linestyle="--", label="Teacher", zorder=5)

        initial_metrics, final_metrics = [], []
        for seed in args.seeds:
            folder = args.output_dir/problem/"Ours"/f"seed_{seed:03d}"
            initial_path, final_path = folder/"policy_initial.npz", folder/"policy_final.npz"
            if initial_path.exists() and final_path.exists():
                with np.load(initial_path, allow_pickle=False) as values:
                    initial_metrics.append(np.asarray(values["metric"]))
                with np.load(final_path, allow_pickle=False) as values:
                    final_metrics.append(np.asarray(values["metric"]))
        for values, color, label in ((initial_metrics, policy_colors["Epoch 0"], "Epoch 0"),
                                     (final_metrics, policy_colors["Final"],
                                      f"Epoch {args.closed_loop_epochs}")):
            if not values:
                continue
            values = np.asarray(values)
            median, lower, upper = np.nanpercentile(values, [50, 25, 75], axis=0)
            steps = np.arange(median.size)
            policy_axis.plot(steps, median, color=color, linewidth=2, label=label, zorder=3)
            policy_axis.fill_between(steps, lower, upper, color=color, alpha=0.16, linewidth=0)

        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Imitation loss")
        policy_axis.set_xlabel("MPC step")
        policy_axis.set_ylabel(metric_labels[problem])
        for axis in (loss_axis, policy_axis):
            axis.xaxis.labelpad = 2.0
            axis.yaxis.labelpad = 1.0
            axis.tick_params(axis="both", pad=1.5)
            axis.set_facecolor("white")
            axis.grid(True, color="#E3E3E3", linewidth=0.65)
            axis.set_axisbelow(True)

    method_handles = [Line2D([0], [0], color=colors[m], linewidth=1.9,
                             linestyle=line_styles[m], marker=markers[m],
                             markersize=3.8, markerfacecolor="white",
                             label=method_display_name(m))
                      for m in args.methods]
    policy_handles = [
        Line2D([0], [0], color=policy_colors["Teacher"], linestyle="--", linewidth=1.8, label="Teacher"),
        Line2D([0], [0], color=policy_colors["Epoch 0"], linewidth=2, label="Epoch 0"),
        Line2D([0], [0], color=policy_colors["Final"], linewidth=2,
               label=f"Epoch {args.closed_loop_epochs}"),
    ]
    _apply_ablation_axes(fig, np.ravel(axes))
    fig.legend(handles=method_handles+policy_handles, loc="upper center",
               ncol=len(method_handles)+len(policy_handles), frameon=False,
               bbox_to_anchor=(0.5, 1.03), columnspacing=1.0, handlelength=1.8)
    fig.subplots_adjust(
        left=0.052, right=0.995, bottom=0.30, top=0.72, wspace=0.55,
    )
    for index, problem in enumerate(args.problems):
        left_bounds = axes[2 * index].get_position()
        right_bounds = axes[2 * index + 1].get_position()
        fig.text(
            0.5 * (left_bounds.x0 + right_bounds.x1),
            0.015,
            f"({chr(ord('a') + index)}) {titles[problem]}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.figure_dir/f"closed_loop_imitation_learning.{extension}",
                    dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "prepare", "run", "plot", "check-backends",
                                            "closed-loop", "plot-closed-loop"))
    parser.add_argument("--problems", nargs="+", choices=models.NAMES, default=list(models.NAMES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS),
                        help="The four paper methods; native methods require external instrumented runtimes")
    parser.add_argument("--casadi-active-tolerance", type=float, default=1e-6,
                        help="CasADi native reverse inactive-multiplier slack threshold")
    parser.add_argument("--acados-root", type=Path, help="Private instrumented acados install prefix")
    parser.add_argument("--acados-template-dir", type=Path, help="Matching acados_template source directory")
    parser.add_argument("--acados-cache-dir", type=Path, help="Generated acados solver cache; prefer a path without spaces")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--train-demos", type=int, default=32)
    parser.add_argument("--validation-demos", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2])
    parser.add_argument("--data-seed", type=int, default=20260905)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--closed-loop-epochs", type=int, default=500)
    parser.add_argument("--closed-loop-steps", type=int, default=50)
    parser.add_argument("--closed-loop-learning-rate", type=float, default=5e-3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--forward-tolerance", type=float, default=1e-8)
    parser.add_argument("--teacher-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--forward-max-iterations", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path,
                        help="Plot and regenerated summary destination; defaults to regenerated/")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--reuse-frozen-far", action="store_true",
                        help="Reuse the known frozen v4 far dataset for the selected open-loop methods")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume incomplete runs and skip completed problem/method/seed combinations")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save restart state every this many epochs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Skip automatic plots after training")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.reuse_frozen_far:
        if args.command != "run" or args.regenerate:
            parser.error("--reuse-frozen-far requires run without --regenerate")
        if args.dataset_dir is None:
            parser.error("--reuse-frozen-far requires an explicit --dataset-dir")
    if args.command=="generate":
        models.generate(ROOT/"native"/"generated")
        print("Generated nonlinear model C/header. Build imitation_native next.")
        return
    if args.quick:
        args.train_demos, args.validation_demos, args.seeds, args.epochs = 4, 2, [0], 10
        args.closed_loop_epochs, args.closed_loop_steps = 10, 10
        args.validation_every, args.print_every = 5, 1
    suffix = "_smoke" if args.quick else ""
    mode = "closed_loop" if args.command in ("closed-loop", "plot-closed-loop") else "open_loop"
    plotting = args.command in ("plot", "plot-closed-loop")
    args.dataset_dir = (args.dataset_dir or (ROOT/"rerun_data_smoke" if args.quick else ROOT/"data")).resolve()
    args.output_dir = (args.output_dir or
                       (ROOT/"results"/mode if plotting else ROOT/("rerun_"+mode+suffix))).resolve()
    args.figure_dir = (args.figure_dir or ROOT/"regenerated").resolve()
    frozen_roots = (ROOT/"data", ROOT/"results", ROOT/"figures")
    if any(args.figure_dir.is_relative_to(path) for path in frozen_roots):
        parser.error("--figure-dir must be outside the frozen data, results, and figures directories")
    if not plotting and any(args.output_dir.is_relative_to(path) for path in frozen_roots):
        parser.error("--output-dir must be outside the frozen data, results, and figures directories")
    if args.regenerate and args.dataset_dir.is_relative_to(ROOT/"data"):
        parser.error("--regenerate requires a new --dataset-dir outside frozen data/")
    for key in ("N", "train_demos", "validation_demos", "epochs", "checkpoint_every",
                "closed_loop_epochs", "closed_loop_steps", "closed_loop_learning_rate",
                "validation_every", "print_every",
                "max_iterations", "forward_max_iterations", "dt", "tolerance",
                "forward_tolerance", "teacher_tolerance", "learning_rate", "casadi_active_tolerance"):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:
            parser.error(f"{key} must be positive and finite")
    if args.command in ("plot", "plot-closed-loop"):
        (plot if args.command == "plot" else plot_closed_loop)(args)
        return
    backend_runtime = {}
    if args.command in ("prepare", "check-backends"):
        # Explicit preparation performs all one-time backend construction.  A
        # formal run creates each backend only once inside its training job.
        preflight_methods = args.methods
        for name in args.problems:
            ev = Evaluator(models.build(name), args)
            for method in preflight_methods:
                adapter = make_adapter(method, ev, args)
                details = adapter_runtime_details(adapter, method)
                backend_runtime[f"{name}/{method}"] = details
                adapter.close()
                print(f"{name}/{method}: backend construction OK", flush=True)
        if args.command == "check-backends":
            return
    with threadpool_limits(limits=1):
        prepared = {}
        for name in args.problems:
            ev = Evaluator(models.build(name), args)
            prepared[name] = (ev, dataset(ev, args))
        if args.command=="prepare":
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
        manifest.update(protocol=models.VERSION, numpy=np.__version__, casadi=ca.__version__,
                        backend_runtime=backend_runtime,
                        python=sys.version, timing_warmups=0, threads=1,
                        cpu_affinity_logical_cpu=0,
                        imitation_loss="half squared state-control trajectory error; mean over demonstrations only",
                        parameter_initialization="identical across methods for each seed",
                        forward_initialization="previous trajectory for the same sample; common first guess",
                        backward="backend-native sensitivities; scalar-loss contraction; closed loop also differentiates first control",
                        dataset_sha256={name:digest(args.dataset_dir/(name+".npz")) for name in args.problems},
                        source_sha256=source_hashes(args.methods))
        manifest.update(
            closed_loop_gradient="mean trajectory loss differentiated through all MPC state transitions; full-rollout-v1",
            closed_loop_warm_start="shifted solution from the previous MPC step",
            safepdp="COCsys.ocSolver, getAuxSys, EQCLQR; rank and constrained-policy algebra fixes",
            sipopt="official native Ipopt and sIPOPT; external certification excluded from training timing",
            casadi_baseline="native nlpsol(ipopt).reverse(1) graph with equivalent QR node rebound for timing; explicit state/control/obstacle inequalities",
            casadi_linear_timer="stock-linsol-record-time: t_wall_nfact + t_wall_solve; one graph solve per reverse seed; numeric kernel boundaries match v3 hooks, clock implementation differs",
            acados_baseline="SQP/HPIPM forward and exact unregularized sensitivity solve; iterate transfer excluded and sensitivity factorization counted in backward_ms",
            acados_forward_configuration="EXACT Hessian, MIRROR regularization, FIXED_STEP; PARTIAL_CONDENSING_HPIPM cond_N=N; unit stage cost scaling",
            acados_private_build="MSVC x64; BLASFEO/HPIPM GENERIC; OpenMP OFF; external runtime build_manifest.json records the actual build",
            safepdp_forward_timer="CasADi nlpsol t_wall_total; excludes per-call NLP construction and nlpsol creation",
            timing_platform=sys.platform,
            timing_protocol=TIMING_PROTOCOL,
            timing_revision="solver-call-only-20260914",
            timing_reference="backend-internal nominal and sensitivity solver invocations",
            timing_forward="native nominal-solver invocation only; CasADi-based methods use nlpsol t_wall_total",
            timing_backward="native adjoint or sensitivity-solver invocation only",
            timing_total="forward_ms + backward_ms; same paired calls",
            timing_excludes="problem/graph construction, nlpsol creation, target and cache preparation, independent residual certification, gradient contraction, optimizer update, validation, teacher generation, disk IO, and plotting",
            safepdp_backward_timer="EQCLQR.eqctlqrSolver wall time; getAuxSys, configuration, and contraction excluded",
            numerical_kernel_diagnostics="target_ms, cache_ms, linear_ms, and contraction_ms remain diagnostic subcomponents; reported backward_ms uses the complete internal solver invocation",
        )
        manifest_name = "closed_loop_manifest.json" if args.command == "closed-loop" else "manifest.json"
        if args.reuse_frozen_far:
            manifest["frozen_dataset_protocol"] = FROZEN_FAR_MODEL[0]
            manifest["frozen_dataset_model_sha256"] = FROZEN_FAR_MODEL[1]
            manifest["dataset_compatibility"] = (
                "Known v4/v5 sampler-only change; stored initial states and teacher trajectories "
                "are reused unchanged; all other dataset settings match")
            manifest["frozen_dataset_reuse_methods"] = list(args.methods)
            manifest_name = ("manifest_safepdp_far_rerun.json" if args.methods == ["SafePDP"]
                             else "manifest_frozen_far_rerun.json")
        manifest_path = args.output_dir/manifest_name
        if args.resume and manifest_path.exists():
            try:
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                backend_runtime.update(previous_manifest.get("backend_runtime", {}))
                manifest["backend_runtime"] = backend_runtime
            except (OSError, ValueError, TypeError):
                pass
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        statuses = []
        if args.command == "run":
            # Alternate method order between seeds; training samples remain serial.
            for name, (ev,data) in prepared.items():
                for index,seed in enumerate(args.seeds):
                    for method in args.methods if index%2==0 else list(reversed(args.methods)):
                        statuses.append(train(ev,data,method,seed,args))
        else:
            references = {name: closed_loop_reference(ev, data, args)
                          for name, (ev, data) in prepared.items()}
            for name, (ev, data) in prepared.items():
                for index, seed in enumerate(args.seeds):
                    for method in args.methods if index%2==0 else list(reversed(args.methods)):
                        statuses.append(train_closed_loop(
                            ev, references[name], method, seed, args))
        for item in statuses:
            details = item.get("runtime_details")
            if details:
                backend_runtime[f"{item['problem']}/{item['method']}"] = details
        manifest["backend_runtime"] = backend_runtime
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        if not args.no_plot:
            (plot if args.command == "run" else plot_closed_loop)(args)
        if any(item["status"]!="ok" for item in statuses):
            raise SystemExit("Some training runs failed; partial traces and status.json are retained.")
    print(f"Wrote imitation-learning results to {args.output_dir}")


def _pin_to_single_p_core() -> None:
    """Pin the benchmark and its native solvers to one Windows CPU thread."""
    import ctypes
    import os

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    if os.name != "nt":
        return

    logical_cpu = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(1 << logical_cpu)):
        raise ctypes.WinError(ctypes.get_last_error())
    print(f"Pinned benchmark process to P-core logical CPU {logical_cpu}.")


if __name__=="__main__":
    _pin_to_single_p_core()
    main()
