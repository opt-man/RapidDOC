"""Nonlinear constrained OCPs for open- and closed-loop imitation learning.

The nominal objective and physical inequalities are kept separate so every
backend can use its native constrained solver. All learned parameters are cost
weights; obstacle geometry is fixed across methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import casadi as ca
import numpy as np

VERSION = "nonlinear-imitation-v5-moderate-initials"
NAMES = ("cartpole", "quadrotor", "robotarm")


@dataclass
class Model:
    name: str
    nx: int
    nu: int
    d: int
    x: object
    u: object
    parameters: object
    dynamics: object
    running: object
    terminal: object
    margins: object
    terminal_margins: object
    teacher_theta: np.ndarray
    initial_center: np.ndarray
    initial_radius: np.ndarray
    goal: np.ndarray
    resting_control: np.ndarray
    state_scale: np.ndarray
    control_scale: np.ndarray

    def initial_conditions(self, count: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.initial_center + rng.uniform(-1, 1, (count, self.nx)) * self.initial_radius

    def functions(self) -> dict:
        """Dense outputs, canonical stage order [x,u], native Hamiltonian sign +lambda*f."""
        lam = ca.SX.sym("costate", self.nx)
        scale = ca.SX.sym("objective_scale")
        inequality_multiplier = ca.SX.sym(
            "inequality_multiplier", int(self.margins.numel()))
        terminal_inequality_multiplier = ca.SX.sym(
            "terminal_inequality_multiplier", int(self.terminal_margins.numel()))
        z = ca.vertcat(self.x, self.u)
        args = [self.x, self.u, self.parameters, lam, scale]
        terminal_args = [self.x, ca.SX.sym("empty_control", 0), self.parameters, lam, scale]
        outputs = {
            "f": self.dynamics,
            "jac": ca.jacobian(self.dynamics, z),
            "hess": ca.hessian(scale*self.running + ca.dot(lam, self.dynamics), z)[0],
            "grad": scale*ca.gradient(self.running, z),
            "cost": scale*self.running,
            "margin": ca.mmin(self.margins),
            "ineq": self.margins,
            "ineqJac": ca.jacobian(self.margins, z),
            "mixed": ca.jacobian(ca.gradient(self.running, z), self.parameters)[:, :self.d],
        }
        result = {key: ca.Function(f"il_{self.name}_{key}", args, [ca.densify(value)])
                  for key, value in outputs.items()}
        for key, value in {
            "hessT": scale*ca.hessian(self.terminal, self.x)[0],
            "gradT": scale*ca.gradient(self.terminal, self.x),
            "costT": scale*self.terminal,
            "marginT": ca.mmin(self.terminal_margins),
            "ineqT": self.terminal_margins,
            "ineqJacT": ca.jacobian(self.terminal_margins, self.x),
            "mixedT": ca.jacobian(ca.gradient(self.terminal, self.x), self.parameters)[:, :self.d],
        }.items():
            result[key] = ca.Function(f"il_{self.name}_{key}", terminal_args, [ca.densify(value)])
        result["ineqHess"] = ca.Function(
            f"il_{self.name}_ineqHess",
            [self.x, self.u, self.parameters, inequality_multiplier, scale],
            [ca.densify(ca.hessian(
                ca.dot(inequality_multiplier, self.margins), z)[0])],
        )
        result["ineqHessT"] = ca.Function(
            f"il_{self.name}_ineqHessT",
            [self.x, ca.SX.sym("empty_control_hess", 0), self.parameters,
             terminal_inequality_multiplier, scale],
            [ca.densify(ca.hessian(
                ca.dot(terminal_inequality_multiplier, self.terminal_margins),
                self.x)[0])],
        )
        return result


def build(name: str) -> Model:
    if name not in NAMES:
        raise ValueError(name)
    nx, nu, d = {"cartpole": (4, 1, 5), "quadrotor": (6, 2, 6),
                 "robotarm": (2, 2, 5)}[name]
    x, u = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
    p = ca.SX.sym("p", d+1)
    weights, dt = ca.exp(p[:d]), p[d]
    if name == "cartpole":
        # x = [cart position, pole angle from upright, cart speed, pole speed].
        def flow(s):
            angle, omega = s[1], s[3]
            si, co = ca.sin(angle), ca.cos(angle)
            temp = (u[0] + 0.1*0.5*omega**2*si)/1.1
            acc_angle = (9.81*si-co*temp)/(0.5*(4/3-0.1*co**2/1.1))
            return ca.vertcat(s[2], omega, temp-0.1*0.5*acc_angle*co/1.1, acc_angle)
        k1 = flow(x)
        k2 = flow(x+dt*k1/2)
        k3 = flow(x+dt*k2/2)
        k4 = flow(x+dt*k3)
        dynamics = x+dt*(k1+2*k2+2*k3+k4)/6
        goal = np.zeros(nx)
        tracking = ca.dot(weights[:4]*ca.DM([1, 5, 0.2, 0.2]), x*x)/2
        effort, terminal_multiplier = 0.05*weights[4]*u[0]**2, 10
        state_margins = ca.vertcat(2-x[0], 2+x[0])
        margins = ca.vertcat(state_margins, 12-u[0], 12+u[0])
        teacher = np.log([2, 1.5, 0.6, 1.8, 0.35])
        # Moderate stabilization states avoid introducing a separate swing-up
        # basin while retaining nonlinear, constrained trajectories.
        center = np.array([0.0, 0.45, 0.0, 0.0])
        radius = np.array([0.2, 0.12, 0.12, 0.2])
        rest, sx, su = np.zeros(nu), np.ones(nx), np.ones(nu)
    elif name == "quadrotor":
        # Planar nonlinear quadrotor, not the 12-state efficiency benchmark.
        vx = x[2]-dt*u[0]*ca.sin(x[4])
        vz = x[3]+dt*(u[0]*ca.cos(x[4])-9.81)
        omega = x[5]+4*dt*u[1]
        angle = x[4]+dt*omega
        dynamics = ca.vertcat(x[0]+dt*vx, x[1]+dt*vz, vx, vz, angle, omega)
        goal = np.array([0.5, 1.0, 0, 0, 0, 0])
        error = x-ca.DM(goal)
        tracking = (4*weights[0]*ca.sumsqr(error[:2]) + weights[1]*ca.sumsqr(x[2:4])
                    + 3*weights[2]*x[4]**2 + 0.3*weights[3]*x[5]**2)/2
        effort = (0.08*weights[4]*(u[0]-9.81)**2
                  + 0.15*weights[5]*u[1]**2)/2
        terminal_multiplier = 15
        state_margins = ca.vertcat(x[1]-0.2, 1.8-x[1], 0.6-x[4], 0.6+x[4])
        margins = ca.vertcat(state_margins, u[0], 2.2*9.81-u[0], 3-u[1], 3+u[1])
        teacher = np.log([1.8, 0.7, 1.5, 0.6, 0.4, 2.0])
        center = np.array([0.05, 0.75, 0, 0, 0.03, 0])
        radius = np.array([0.08, 0.06, 0.06, 0.06, 0.04, 0.06])
        rest, sx, su = np.array([9.81, 0]), np.ones(nx), np.ones(nu)
    else:
        # Kinematic arm: linear joint dynamics, nonlinear task cost/obstacle.
        dynamics = x+dt*u
        goal = np.array([0.75, -0.65])
        def ee(q):
            return ca.vertcat(ca.cos(q[0])+0.8*ca.cos(q[0]+q[1]),
                              ca.sin(q[0])+0.8*ca.sin(q[0]+q[1]))
        tracking = (2*weights[0]*ca.sumsqr(x-ca.DM(goal))
                    + weights[1]*ca.sumsqr(ee(x)-ee(ca.DM(goal))))/2
        effort = (0.1*weights[2]*u[0]**2+0.1*weights[3]*u[1]**2)/2
        terminal_multiplier = 20
        obstacle_radius = 0.39
        obstacle_margin = ca.sumsqr(ee(x)-ca.DM([1.3, 0.3]))-obstacle_radius**2
        state_margins = ca.vertcat(x+1.6, 1.6-x, obstacle_margin)
        margins = ca.vertcat(state_margins, 3-u, 3+u)
        teacher = np.log([0.7, 2.2, 0.5, 1.5, 2.0])
        center, radius = np.array([0.0, 0.05]), np.array([0.08, 0.08])
        rest, sx, su = np.zeros(nu), np.ones(nx), np.ones(nu)
        terminal_multiplier = terminal_multiplier+weights[4]
    return Model(name, nx, nu, d, x, u, p, dynamics,
                 tracking+effort, terminal_multiplier*tracking,
                 margins, state_margins, teacher, center, radius, goal, rest, sx, su)


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(directory)
        generator = ca.CodeGenerator("imitation_models.c", {"with_header": True})
        for name in NAMES:
            for function in build(name).functions().values():
                generator.add(function)
        generator.generate()
    finally:
        os.chdir(previous)
