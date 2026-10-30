"""Frozen KKT data used by the factor-reuse ablation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FrozenKKT:
    problem: str
    seed: int
    horizon: int
    nx: int
    nu: int
    target_hessian: Array
    target_terminal_hessian: Array
    target_dynamics: Array
    forward_hessian: Array
    forward_terminal_hessian: Array
    forward_dynamics: Array
    value_hessians: Array
    control_cholesky: Array
    cross_factors: Array
    rhs: Array
    optimal_primal: Array
    metadata: dict

    @property
    def stage_dimension(self) -> int:
        return self.nx + self.nu

    @property
    def primal_dimension(self) -> int:
        return self.horizon * self.stage_dimension + self.nx

    @property
    def total_dimension(self) -> int:
        return self.primal_dimension + (self.horizon + 1) * self.nx


def load(path: Path) -> FrozenKKT:
    with np.load(path, allow_pickle=False) as payload:
        return FrozenKKT(
            str(payload["problem"]), int(payload["seed"]),
            int(payload["horizon"]), int(payload["nx"]), int(payload["nu"]),
            *(np.asarray(payload[name], dtype=float) for name in (
                "target_hessian", "target_terminal_hessian", "target_dynamics",
                "forward_hessian", "forward_terminal_hessian", "forward_dynamics",
            )),
            *(np.asarray(payload[name], dtype=float) if name in payload else np.empty(0)
              for name in ("value_hessians", "control_cholesky", "cross_factors")),
            *(np.asarray(payload[name], dtype=float) for name in ("rhs", "optimal_primal")),
            json.loads(str(payload["metadata"])),
        )
