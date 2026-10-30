"""CasADi's IPOPT reverse derivative with a retained numerical timing node.

This adapter keeps the state and control inequalities, including the arm's
nonlinear obstacle, in the nominal NLP.  Its derivative is nlpsol.reverse(1);
there is no separately assembled NumPy KKT solve or perturbed NLP solve.

For timing, its native reverse graph's one linear-solve node is replaced by a
retained CasADi QR Linsol with the identical matrix, RHS, transpose flag and
solver options.  This is a timing adaptation of the native derivative graph,
not the original reverse Function left untouched.  Linsol's own numerical
factorization and backsolve statistics supply the backward solve-kernels-v4
measurement. Forward uses the same IPOPT algorithm in a private instrumented
plugin, excluding derivatives and KKT construction on every iteration.
Unknown graph structures or missing counters fail explicitly.
"""
from __future__ import annotations

import math
import os
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import casadi as ca
import numpy as np
from .imitation_ipopt_timing import NativeIpoptTimer

TIMING_PROTOCOL = "solve-kernels-v4"
SENSITIVITY_SOLVER = "qr"
SENSITIVITY_OPTIONS = {"record_time": True, "print_time": False}


class NativeReverseTimer:
    """Expose the actual solve statistics without timing graph preparation.

    The caller must construct native reverse with SENSITIVITY_SOLVER and
    SENSITIVITY_OPTIONS: MX.info exposes transpose, not the original plugin.
    CasADi 3.8's LinsolCall resets fstats on every evaluation; its nfact/solve
    timers surround exactly the numerical plugin calls required by v4.
    sfact, RHS copies, graph derivatives and contraction remain excluded.
    """

    def __init__(self, native_reverse):
        for function in native_reverse.find_functions(-1):
            if function.is_a("MXFunction") and any(
                    function.instruction_id(i) == ca.OP_SOLVE
                    for i in range(function.n_instructions())):
                raise RuntimeError("CasADi timing does not support nested linear solves")
        inputs = native_reverse.mx_in()
        outputs = native_reverse.call(inputs, True, False)
        stack, seen, solve_nodes = list(outputs), set(), []
        while stack:
            node = stack.pop()
            identity = hash(node)
            if identity in seen:
                continue
            seen.add(identity)
            if node.op() == ca.OP_SOLVE:
                solve_nodes.append(node)
            stack.extend(node.dep(i) for i in range(node.n_dep()))
        # Current nlpsol.reverse(1) generates exactly one solve. Reject a new
        # graph layout instead of risking overwritten or unobserved counters.
        if len(solve_nodes) != 1:
            raise RuntimeError(
                f"CasADi timing expects one native linear solve, found {len(solve_nodes)}")
        node = solve_nodes[0]
        info = node.info()
        if set(info) != {"tr"} or node.n_dep() != 2:
            raise RuntimeError("CasADi timing encountered an unknown solve-node layout")
        rhs, matrix = node.dep(0), node.dep(1)
        self.linsol = ca.Linsol("imitation_timed_qr", SENSITIVITY_SOLVER,
                                matrix.sparsity(), dict(SENSITIVITY_OPTIONS))
        replacement = self.linsol.solve(matrix, rhs, bool(info["tr"]))
        outputs = ca.graph_substitute(outputs, [node], [replacement])
        self.function = ca.Function("timed_"+native_reverse.name(), inputs, outputs,
                                    native_reverse.name_in(), native_reverse.name_out())
        count = sum(self.function.instruction_id(i) == ca.OP_SOLVE
                    for i in range(self.function.n_instructions()))
        if count != 1:
            raise RuntimeError("CasADi timed graph does not execute exactly one linear solve")
        self.transpose = bool(info["tr"])
        self.calls, self._armed, self.last_stats = 0, False, None

    def reset(self):
        if self._armed:
            raise RuntimeError("CasADi solve timer is already awaiting a result")
        self._armed = True

    def cancel(self):
        self._armed = False

    def read(self):
        if not self._armed:
            raise RuntimeError("CasADi solve timer was not armed for this call")
        self._armed = False
        self.last_stats = self.linsol.stats()
        record = self.last_stats
        if record.get("n_call_nfact") != 1 or record.get("n_call_solve") != 1:
            raise RuntimeError("CasADi factorization/backsolve counters are missing or repeated")
        parts = [float(record.get(name, float("nan")))
                 for name in ("t_wall_nfact", "t_wall_solve")]
        value = sum(parts)
        if any(not math.isfinite(part) or part < 0 for part in parts) or value <= 0:
            raise RuntimeError(
                "CasADi native reverse did not produce a positive finite v4 "
                "factorization/backsolve measurement; refusing a timing row")
        self.calls += 1
        return value


class CasADi:
    """Persistent constrained IPOPT forward and its own native adjoint.

    ``gradient(rhs, state_jacobian=None)`` returns the same four fields as
    SafePDP.gradient.  Closed-loop calls also populate
    ``control_parameter_jacobian`` with shape (nu, d).  CasADi combines target
    derivatives and parameter contraction inside its reverse graph, so their
    separate attribution is NaN.  They are nevertheless excluded from the
    measured numerical kernel time by the retained Linsol's own statistics.
    """

    timing_protocol = TIMING_PROTOCOL
    derivative_backend = "CasADi nlpsol(ipopt).reverse(1), retained QR timing node"

    def __init__(self, ev, args):
        self.ev, self.args = ev, args
        self.forward_timer = NativeIpoptTimer()
        self.runtime_details = self.forward_timer.runtime_details
        m, N = ev.model, ev.N
        self.primal_size = N*(m.nx+m.nu)+m.nx
        self.active_tolerance = float(getattr(args, "casadi_active_tolerance", 1e-6))
        if not math.isfinite(self.active_tolerance) or self.active_tolerance <= 0:
            raise ValueError("casadi_active_tolerance must be positive and finite")

        w = ca.MX.sym("w", self.primal_size)
        par = ca.MX.sym("par", m.d+1+m.nx)
        p, initial = par[:m.d+1], par[m.d+1:]
        objective, constraints = 0, [w[:m.nx]-initial]
        lower, upper = [0.0]*m.nx, [0.0]*m.nx
        for k in range(N):
            offset = k*(m.nx+m.nu)
            x, u = w[offset:offset+m.nx], w[offset+m.nx:offset+m.nx+m.nu]
            next_x = w[offset+m.nx+m.nu:offset+2*m.nx+m.nu]
            objective += ev.functions["cost"](x, u, p, ev.zero, 1)
            constraints.append(next_x-ev.functions["f"](x, u, p, ev.zero, 1))
            lower.extend([0.0]*m.nx)
            upper.extend([0.0]*m.nx)
            constraints.append(ev.functions["ineq"](x, u, p, ev.zero, 1))
            count = int(m.margins.numel())
            lower.extend([0.0]*count)
            upper.extend([np.inf]*count)
        objective += ev.functions["costT"](w[-m.nx:], [], p, ev.zero, 1)
        constraints.append(ev.functions["ineqT"](w[-m.nx:], [], p, ev.zero, 1))
        terminal_count = int(m.terminal_margins.numel())
        lower.extend([0.0]*terminal_count)
        upper.extend([np.inf]*terminal_count)

        self.solver = ca.nlpsol("imitation_casadi_"+m.name, self.forward_timer.plugin, {
            "x": w, "p": par, "f": objective, "g": ca.vertcat(*constraints),
        }, {
            "print_time": False, "record_time": True, "error_on_fail": False,
            "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.tol": args.forward_tolerance,
            "ipopt.acceptable_tol": args.forward_tolerance,
            "ipopt.constr_viol_tol": args.forward_tolerance,
            "ipopt.max_iter": args.forward_max_iterations,
            "ipopt.nlp_scaling_method": "none", "ipopt.bound_relax_factor": 0.0,
            "sens_linsol": SENSITIVITY_SOLVER,
            "sens_linsol_options": dict(SENSITIVITY_OPTIONS),
            # Native reverse treats every nonzero multiplier as active.  IPOPT
            # leaves barrier-sized multipliers on inactive inequalities; clear
            # them using CasADi's documented slack-based activity test.
            "clip_inactive_lam": True, "inactive_lam_strategy": "abstol",
            "inactive_lam_value": self.active_tolerance,
        })
        self.native_reverse = self.solver.reverse(1)
        self.timer = NativeReverseTimer(self.native_reverse)
        self.reverse = self.timer.function
        defaults = {"lbg": ca.DM(lower), "ubg": ca.DM(upper)}
        self.inputs = [defaults.get(name, ca.DM.ones(self.solver.sparsity_in(i))
                                   * self.solver.default_in(i))
                       for i, name in enumerate(self.solver.name_in())]
        self.input_indices = {name: i for i, name in enumerate(self.solver.name_in())}
        self.output_indices = {name: i for i, name in enumerate(self.solver.name_out())}
        self.reverse_seed_index = self.reverse.index_in("adj_x")
        self.reverse_parameter_index = self.reverse.index_out("adj_p")
        self._reverse_base = None
        self.control_parameter_jacobian = None
        self.forward_seconds = self.backward_native_seconds = float("nan")

    def forward(self, theta, initial, guess):
        self._reverse_base = None
        self.control_parameter_jacobian = None
        self.inputs[self.input_indices["x0"]] = ca.DM(np.asarray(guess, dtype=np.float64))
        self.inputs[self.input_indices["p"]] = ca.DM(np.r_[self.ev.parameters(theta), initial])
        # The native hooks time only numerical solve work inside IPOPT; model
        # derivatives, KKT assembly and cache setup are excluded each iteration.
        self.forward_timer.reset()
        try:
            result = self.solver.call(self.inputs)
        except BaseException:
            self.forward_timer.cancel()
            raise
        stats = self.solver.stats()
        if not stats.get("success", False):
            self.forward_timer.cancel()
            raise RuntimeError("CasADi-IPOPT forward: "+str(stats.get("return_status")))
        self.forward_seconds = self.forward_timer.read()
        self.forward_native_seconds = float(stats["t_wall_total"])
        self.forward_iterations = int(stats.get("iter_count", 0))

        self._reverse_base = []
        for i, name in enumerate(self.reverse.name_in()):
            if name.startswith("out_"):
                value = result[self.output_indices[name[4:]]]
            elif name in self.input_indices:
                value = self.inputs[self.input_indices[name]]
            else:
                value = ca.DM.zeros(self.reverse.sparsity_in(i))
            self._reverse_base.append(value)
        output = np.asarray(result[self.output_indices["x"]], dtype=np.float64).ravel()
        if not np.all(np.isfinite(output)):
            self._reverse_base = None
            raise RuntimeError("CasADi forward returned a nonfinite trajectory")
        return np.ascontiguousarray(output)

    def gradient(self, rhs, state_jacobian=None):
        if self._reverse_base is None:
            raise RuntimeError("CasADi requires a matching successful forward solve")
        m = self.ev.model
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        if rhs.size not in (self.primal_size, self.primal_size+(self.ev.N+1)*m.nx):
            raise ValueError("CasADi loss seed has an unexpected size")
        if np.any(rhs[self.primal_size:] != 0) or not np.all(np.isfinite(rhs)):
            raise ValueError("CasADi loss seed must be finite with zero equality blocks")
        if state_jacobian is not None:
            state_jacobian = np.asarray(state_jacobian, dtype=np.float64)
            if state_jacobian.shape != (m.nx, m.d) or not np.all(np.isfinite(state_jacobian)):
                raise ValueError("state_jacobian must be finite with shape (nx, d)")

        seeds = [rhs[:self.primal_size]]
        if state_jacobian is not None:
            for j in range(m.nu):
                seed = np.zeros(self.primal_size)
                seed[m.nx+j] = 1.0
                seeds.append(seed)
        parameter_rows, linear_seconds, native_seconds = [], 0.0, 0.0
        for seed in seeds:
            inputs = list(self._reverse_base)
            inputs[self.reverse_seed_index] = ca.DM(seed)
            self.timer.reset()
            started = time.perf_counter()
            try:
                answer = self.reverse.call(inputs)
            except BaseException:
                self.timer.cancel()
                raise
            native_seconds += time.perf_counter()-started
            linear_seconds += self.timer.read()
            parameter_rows.append(np.asarray(
                answer[self.reverse_parameter_index], dtype=np.float64).ravel())
        self.backward_native_seconds = native_seconds
        parameter_rows = np.asarray(parameter_rows)
        expected = (len(seeds), m.d+1+m.nx)
        if parameter_rows.shape != expected or not np.all(np.isfinite(parameter_rows)):
            raise RuntimeError("CasADi native reverse returned invalid parameter derivatives")
        self.native_parameter_gradients = parameter_rows
        gradients = parameter_rows[:, :m.d].copy()
        if state_jacobian is not None:
            # par is [theta, dt, initial_state].  The fixed dt column is omitted.
            gradients += parameter_rows[:, m.d+1:] @ state_jacobian
            self.control_parameter_jacobian = np.ascontiguousarray(gradients[1:])
        return (np.ascontiguousarray(gradients[0]), float("nan"),
                linear_seconds, float("nan"))

    def close(self):
        self._reverse_base = None
        self.control_parameter_jacobian = None
