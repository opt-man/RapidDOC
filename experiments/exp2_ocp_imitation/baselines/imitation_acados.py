"""Native acados SQP and exact HPIPM sensitivities for imitation learning.

All model inequalities stay hard, including the arm obstacle.  Forward SQP
uses MIRROR regularization; sensitivities use a separate exact, unregularized
solver at the accepted primal AND dual iterate.  No surrogate KKT solver,
finite differences, penalty objective, or fallback backend is used.

The private acados dependency must export solve-kernels-v4 hooks.  Closed-loop
derivatives additionally require the hook covering qp_solver->eval_forw_sens;
the older figure-eight dependency only timed its adjoint counterpart.
Generation/build, parameter uploads, primal/dual transfer, target derivatives,
output extraction and final parameter contraction are outside kernel timing.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIMING_PROTOCOL = "solve-kernels-v4"
KERNEL_KINDS = ("factor", "solve", "residual", "reduction", "condensing",
                "vector", "forward", "adjoint", "ipm_control")
_DLL_DIRECTORIES = []


class NativeSolveTimer:
    """Read the timer in the exact acados module used by the solver."""

    def __init__(self, library):
        self.library = library
        try:
            self._reset = library.frs_acados_solve_timer_reset_v4
            self._read = library.frs_acados_solve_timer_read_v4
            self._count = library.frs_acados_solve_timer_count_v4
            self._qp_calls = library.frs_acados_solve_timer_qp_calls_v4
        except AttributeError as error:
            raise RuntimeError(
                "acados requires a private solve-kernels-v4 runtime with "
                "frs_acados_solve_timer_*_v4 hooks; set IMITATION_ACADOS_ROOT "
                "to an instrumented external runtime. Native API wall time "
                "cannot substitute for the numerical solve timer.") from error
        self._reset.argtypes, self._reset.restype = [], None
        self._read.argtypes, self._read.restype = [], ct.c_double
        self._count.argtypes, self._count.restype = [ct.c_int], ct.c_ulonglong
        self._qp_calls.argtypes, self._qp_calls.restype = [], ct.c_ulonglong
        try:
            marker = library.frs_acados_hpipm_kernel_timer_enabled_v4
            marker.argtypes, marker.restype = [], ct.c_int
            strict_kernels = marker() == 1
        except AttributeError:
            strict_kernels = False
        if not strict_kernels:
            raise RuntimeError(
                "acados solve-kernels-v4 requires the strict HPIPM kernel hooks "
                "and frs_acados_hpipm_kernel_timer_enabled_v4 marker; older "
                "QP wrapper timers include separable workspace/copy operations.")

    def require_forward_sensitivity_hook(self):
        try:
            check = self.library.frs_acados_forward_sens_timer_enabled_v3
            check.argtypes, check.restype = [], ct.c_int
            supported = check() == 1
        except AttributeError:
            supported = False
        if not supported:
            raise RuntimeError(
                "acados closed-loop timing needs the v3 qp_solver->eval_forw_sens "
                "hook and frs_acados_forward_sens_timer_enabled_v3 marker; "
                "the older figure-eight adjoint-only runtime is insufficient.")

    def reset(self):
        self._reset()

    def read(self, allow_no_qp=False):
        value = float(self._read())
        self.last_counts = {name: int(self._count(i)) for i, name in enumerate(KERNEL_KINDS)}
        self.last_qp_calls = int(self._qp_calls())
        # A stationary initial iterate can finish the NLP without entering any
        # QP. Missing hooks are rejected in __init__; a verified empty call is
        # allowed to be zero, never replaced by an invented positive duration.
        if (allow_no_qp and self.last_qp_calls == 0 and not any(self.last_counts.values())
                and (math.isnan(value) or value == 0)):
            return 0.0
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("acados returned no positive finite v4 numerical solve time")
        if not any(self.last_counts.values()):
            raise RuntimeError("acados reported numerical time without any kernel calls")
        return value


def _runtime(args):
    filename = ("acados.dll" if os.name == "nt" else
                "libacados.dylib" if sys.platform == "darwin" else "libacados.so")
    runtime_root = (getattr(args, "acados_root", None)
                    or os.environ.get("IMITATION_ACADOS_ROOT")
                    or os.environ.get("ACADOS_SOURCE_DIR"))
    if not runtime_root:
        raise RuntimeError("Set IMITATION_ACADOS_ROOT or --acados-root to an instrumented external install")
    prefix = Path(runtime_root).expanduser().resolve()
    binary = prefix / ("bin" if os.name == "nt" else "lib")
    library_path = binary / filename
    if not library_path.is_file():
        raise RuntimeError(f"acados private runtime missing: {library_path}; "
                           "build it first or set IMITATION_ACADOS_ROOT")
    os.environ["ACADOS_SOURCE_DIR"] = str(prefix)
    if os.name == "nt":
        os.environ["PATH"] = str(binary) + os.pathsep + os.environ.get("PATH", "")
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(binary)))
    library = ct.CDLL(str(library_path))
    NativeSolveTimer(library)  # fail before generating or compiling anything
    template = (getattr(args, "acados_template_dir", None)
                or os.environ.get("IMITATION_ACADOS_TEMPLATE_DIR"))
    if template is None:
        candidate = prefix / "interfaces" / "acados_template"
        template = candidate if candidate.is_dir() else None
    if template is not None:
        template = Path(template).resolve()
        if not (template / "acados_template" / "__init__.py").is_file():
            raise RuntimeError(f"acados_template source missing: {template}")
        sys.path.insert(0, str(template))
    try:
        import acados_template
    except ImportError as error:
        raise RuntimeError("matching acados_template is required for offline source generation") from error
    return prefix, library_path, library, acados_template


def _build_configuration(manifest_path, binary_hashes):
    """Accept build claims only when the manifest identifies these binaries."""
    if not manifest_path.is_file():
        return {"status": "unavailable", "reason": "no build_manifest.json supplied"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    recorded = manifest.get("library_sha256", {})
    if (not binary_hashes or not all(name in recorded and
            str(recorded[name]).lower() == digest.lower()
            for name, digest in binary_hashes.items())):
        raise RuntimeError("acados build manifest does not match the loaded native libraries")
    return {"status": "binary-hashes-verified", "manifest_path": str(manifest_path),
            "manifest": manifest}


def _runtime_metadata(prefix, library_path, library):
    binaries = {}
    if os.name == "nt":
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [ct.c_wchar_p], ct.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ct.c_void_p, ct.c_wchar_p, ct.c_uint]
        kernel.GetModuleFileNameW.restype = ct.c_uint
        for filename in ("acados.dll", "frs_imitation_blasfeo.dll", "frs_imitation_hpipm.dll"):
            handle = library._handle if filename == "acados.dll" else kernel.GetModuleHandleW(filename)
            if not handle:
                raise RuntimeError(f"acados dependency is not loaded: {filename}")
            buffer = ct.create_unicode_buffer(32768)
            length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
            if not length or length >= len(buffer):
                raise RuntimeError(f"cannot identify loaded acados dependency: {filename}")
            path = Path(buffer.value).resolve()
            binaries[filename] = {"path": str(path), "path_source": "loaded Windows module",
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        # Explicitly label these as installation files.  Do not infer an
        # external override's compiler/target from its directory name.
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        for filename in ("libacados"+suffix, "libblasfeo"+suffix, "libhpipm"+suffix):
            path = library_path if filename == "libacados"+suffix else prefix/"lib"/filename
            if path.is_file():
                binaries[filename] = {"path": str(path.resolve()),
                                      "path_source": "installation file",
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "native_libraries": binaries,
        "native_build": _build_configuration(
            prefix/"build_manifest.json", {name: item["sha256"] for name, item in binaries.items()}),
        "strict_hpipm_kernel_timer": True,
        "timing_protocol": TIMING_PROTOCOL,
    }


def _make_ocp(ev, args, prefix, name, sensitivity, template):
    """Translate the canonical model directly; no state or cost transcription."""
    m, N = ev.model, ev.N
    ocp = template.AcadosOcp(acados_path=str(prefix))
    model = ocp.model
    model.name, model.x, model.u = name, m.x, m.u
    # All learned weights are global; dt is fixed at construction.
    theta = ca.SX.sym("theta", m.d)
    p = ca.vertcat(theta, ev.dt)
    replace = lambda expr: ca.substitute(expr, m.parameters, p)
    model.p_global = theta
    model.disc_dyn_expr = replace(m.dynamics)
    model.cost_expr_ext_cost = replace(m.running)
    model.cost_expr_ext_cost_e = replace(m.terminal)
    model.con_h_expr_0 = replace(m.margins)
    model.con_h_expr = replace(m.margins)
    model.con_h_expr_e = replace(m.terminal_margins)
    ocp.p_global_values = np.asarray(m.teacher_theta, dtype=np.float64)
    ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.constraints.x0 = np.asarray(m.initial_center, dtype=np.float64)
    # HPIPM represents one-sided bounds using the official infinity sentinel.
    from acados_template.utils import ACADOS_INFTY
    for suffix, count in (("_0", m.margins.numel()), ("", m.margins.numel()),
                          ("_e", m.terminal_margins.numel())):
        setattr(ocp.constraints, "lh"+suffix, np.zeros(int(count)))
        setattr(ocp.constraints, "uh"+suffix, np.full(int(count), ACADOS_INFTY))
    o = ocp.solver_options
    o.N_horizon, o.tf = N, N*ev.dt
    o.cost_scaling = np.ones(N+1)  # same sum; no additional dt scaling
    o.integrator_type, o.nlp_solver_type = "DISCRETE", "SQP"
    o.hessian_approx = "EXACT"
    o.regularize_method = "NO_REGULARIZE" if sensitivity else "MIRROR"
    o.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    o.qp_solver_cond_N = N
    o.qp_solver_ric_alg = o.qp_solver_cond_ric_alg = 0
    o.nlp_solver_max_iter = int(args.forward_max_iterations)
    o.nlp_solver_tol_stat = o.nlp_solver_tol_eq = float(args.forward_tolerance)
    o.nlp_solver_tol_ineq = o.nlp_solver_tol_comp = float(args.forward_tolerance)
    qp_tol = min(float(args.forward_tolerance)*0.01, 1e-10)
    o.qp_solver_tol_stat = o.qp_solver_tol_eq = qp_tol
    o.qp_solver_tol_ineq = o.qp_solver_tol_comp = qp_tol
    o.qp_solver_iter_max = 100
    o.globalization = "FIXED_STEP"
    o.with_solution_sens_wrt_params = True
    o.solution_sens_qp_t_lam_min = 1e-9
    return ocp


class Acados:
    """SafePDP-compatible native adjoint, with complete closed-loop chain rule."""

    timing_protocol = TIMING_PROTOCOL
    derivative_backend = "acados exact HPIPM adjoint and initial-state sensitivities"

    def __init__(self, ev, args):
        self.ev, self.args = ev, args
        prefix, library_path, library, template = _runtime(args)
        self.timer = NativeSolveTimer(library)
        m, N = ev.model, ev.N
        self.primal_size = N*(m.nx+m.nu)+m.nx
        self._ready = False
        self.control_parameter_jacobian = None
        self.forward_seconds = self.forward_native_seconds = self.backward_native_seconds = float("nan")
        cache = Path(getattr(args, "acados_cache_dir", None)
                     or Path.home()/".fatrop_factor_reuse"/"imitation_acados").resolve()
        identity = {
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": hashlib.sha256((ROOT/"imitation_models.py").read_bytes()).hexdigest(),
            "library": str(library_path),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "template": str(template.__file__), "casadi": ca.__version__,
            "problem": m.name, "N": N, "dt": ev.dt,
            "tolerance": args.forward_tolerance,
            "max_iterations": args.forward_max_iterations,
        }
        tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.solvers = []
        for sensitivity in (False, True):
            name = f"il_ac_{m.name}_{tag}_{'sens' if sensitivity else 'fwd'}"
            folder = cache/name
            folder.mkdir(parents=True, exist_ok=True)
            ocp = _make_ocp(ev, args, prefix, name, sensitivity, template)
            ocp.code_export_directory = str(folder/"code")
            json_file = folder/(name+".json")
            builder = None
            if os.name == "nt":
                builder = template.ocp_get_default_cmake_builder()
                builder.generator, builder.host = "Visual Studio 17 2022", "x64"
                builder.build_dir = str(folder/"build")
            from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
            generated = Path(ocp.code_export_directory)/(
                f"{get_shared_lib_prefix()}acados_ocp_solver_{name}{get_shared_lib_ext()}")
            build = not (generated.is_file() and json_file.is_file())
            solver = template.AcadosOcpSolver(
                ocp, json_file=str(json_file), generate=build, build=build,
                cmake_builder=builder, verbose=False)
            # Verify the hooks on the module actually used by this capsule.
            NativeSolveTimer(solver.acados_lib)
            if solver.acados_lib._handle != library._handle:
                raise RuntimeError("acados capsule and solve timer loaded different libraries")
            self.solvers.append(solver)
        self.forward_solver, self.sensitivity_solver = self.solvers
        self.runtime_details = dict(identity, **_runtime_metadata(prefix, library_path, library))

    def forward(self, theta, initial, guess):
        self._ready = False
        self.control_parameter_jacobian = None
        m, N = self.ev.model, self.ev.N
        theta, initial, guess = (np.ascontiguousarray(v, dtype=np.float64).ravel()
                                 for v in (theta, initial, guess))
        if (theta.size != m.d or initial.size != m.nx or guess.size != self.primal_size
                or not all(np.all(np.isfinite(v)) for v in (theta, initial, guess))):
            raise ValueError("acados forward requires finite, correctly sized inputs")
        f, s = self.forward_solver, self.sensitivity_solver
        # Upload parameters and initial bounds; no native solve has started.
        for solver in (f, s):
            solver.set_p_global_and_precompute_dependencies(theta)
            solver.constraints_set(0, "lbx", initial)
            solver.constraints_set(0, "ubx", initial)
        f.reset()
        for k in range(N+1):
            offset = k*(m.nx+m.nu)
            f.set(k, "x", guess[offset:offset+m.nx])
            if k < N:
                f.set(k, "u", guess[offset+m.nx:offset+m.nx+m.nu])
        # Every nonlinear iteration's derivatives/linearization and all
        # separable packing remain untimed. Only instrumented HPIPM numerical
        # work accumulates while the native solve is running.
        self.timer.reset()
        try:
            status = f.solve()
        except BaseException:
            try:
                self.timer.read(allow_no_qp=True)
            except RuntimeError:
                pass
            raise
        self.forward_native_seconds = float(f.get_stats("time_tot"))
        self.forward_iterations = int(f.get_stats("sqp_iter"))
        self.forward_seconds = self.timer.read(allow_no_qp=self.forward_iterations == 0)
        self.forward_kernel_counts = dict(self.timer.last_counts)
        self.forward_qp_calls = self.timer.last_qp_calls
        self.forward_qp_iterations = np.asarray(f.get_stats("qp_iter"), dtype=int).ravel().tolist()
        if status != 0:
            raise RuntimeError(f"acados SQP forward failed: status={status}, "
                               f"residuals={f.get_residuals().tolist()}")
        if (self.forward_iterations > 0 and self.forward_qp_calls == 0):
            raise RuntimeError("acados SQP iterations were reported without any instrumented QP calls")
        output = np.empty(self.primal_size)
        for k in range(N+1):
            offset = k*(m.nx+m.nu)
            output[offset:offset+m.nx] = f.get(k, "x")
            if k < N:
                output[offset+m.nx:offset+m.nx+m.nu] = f.get(k, "u")
        if not np.all(np.isfinite(output)):
            raise RuntimeError("acados forward returned a nonfinite trajectory")
        # Transfer the accepted iterate outside the reported solver times.  The
        # sensitivity factorization itself is deferred to the backward call.
        for k in range(N+1):
            for field in ("x", "u", "pi", "lam", "sl", "su"):
                if k == N and field in ("u", "pi"):
                    continue
                value = f.get(k, field)
                if value.size:
                    s.set(k, field, value)
        self._ready = True
        return output

    def gradient(self, rhs, state_jacobian=None):
        if not self._ready:
            raise RuntimeError("acados requires a matching successful forward solve")
        m, N = self.ev.model, self.ev.N
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        if (rhs.size not in (self.primal_size, self.primal_size+(N+1)*m.nx)
                or not np.all(np.isfinite(rhs)) or np.any(rhs[self.primal_size:] != 0)):
            raise ValueError("acados loss seed must be finite with zero equality blocks")
        if state_jacobian is not None:
            state_jacobian = np.asarray(state_jacobian, dtype=np.float64)
            if state_jacobian.shape != (m.nx, m.d) or not np.all(np.isfinite(state_jacobian)):
                raise ValueError("state_jacobian must be finite with shape (nx, d)")
            self.timer.require_forward_sensitivity_hook()
        s = self.sensitivity_solver
        count = 1 if state_jacobian is None else 1+m.nu
        seed_x, seed_u = [], []
        for k in range(N+1):
            offset = k*(m.nx+m.nu)
            sx = np.zeros((m.nx, count))
            sx[:, 0] = rhs[offset:offset+m.nx]
            seed_x.append((k, sx))
            if k < N:
                su = np.zeros((m.nu, count))
                su[:, 0] = rhs[offset+m.nx:offset+m.nx+m.nu]
                if k == 0 and state_jacobian is not None:
                    su[:, 1:] = np.eye(m.nu)
                seed_u.append((k, su))
        self.timer.reset()
        started = time.perf_counter()
        try:
            status = s.setup_qp_matrices_and_factorize()
            if status != 0:
                raise RuntimeError(
                    f"acados exact sensitivity factorization failed: status={status}")
            parameter_rows = s.eval_adjoint_solution_sensitivity(seed_x, seed_u)
            initial = None
            if state_jacobian is not None:
                initial = s.eval_solution_sensitivity(
                    list(range(N+1)), with_respect_to="initial_state")
        except BaseException:
            try:
                self.timer.read()
            except RuntimeError:
                pass
            raise
        self.backward_native_seconds = time.perf_counter()-started
        linear = self.timer.read()
        self.backward_kernel_counts = dict(self.timer.last_counts)
        parameter_rows = np.asarray(parameter_rows, dtype=np.float64)
        if parameter_rows.shape != (count, m.d) or not np.all(np.isfinite(parameter_rows)):
            raise RuntimeError("acados returned invalid native adjoint derivatives")
        gradient = parameter_rows[0].copy()
        self.native_parameter_gradients = parameter_rows.copy()
        if initial is not None:
            dx, du = np.asarray(initial["sens_x"]), np.asarray(initial["sens_u"])
            if (dx.shape != (N+1, m.nx, m.nx) or du.shape != (N, m.nu, m.nx)
                    or not np.all(np.isfinite(dx)) or not np.all(np.isfinite(du))):
                raise RuntimeError("acados returned invalid initial-state sensitivities")
            loss_x0 = sum(sx[:, 0]@dx[k] for k, sx in seed_x)
            loss_x0 += sum(su[:, 0]@du[k] for k, su in seed_u)
            gradient += loss_x0@state_jacobian
            self.control_parameter_jacobian = np.ascontiguousarray(
                parameter_rows[1:]+du[0]@state_jacobian)
        # Target construction and contraction happen partly inside acados;
        # their individual wall times are not available, but hooks exclude both.
        return np.ascontiguousarray(gradient), float("nan"), linear, float("nan")

    def close(self):
        self._ready = False
        self.control_parameter_jacobian = None
        self.solvers.clear()
        self.forward_solver = self.sensitivity_solver = None
