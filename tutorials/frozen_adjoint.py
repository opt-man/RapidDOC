"""Solve one saved target adjoint using the original forward-factor cache."""
from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/exp1_factor_reuse_ablation'))
from frozen_cases import load
from linear_system import NativeFactor, _residual

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['cartpole','quadrotor','robotarm'], default='cartpole')
    parser.add_argument('--seed', type=int, choices=range(32), default=0)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--max-iterations', type=int)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('--tolerance must be finite and positive')
    case = load(ROOT/'experiments/exp1_factor_reuse_ablation/data'/args.problem/f'seed_{args.seed:03d}.npz')
    limit = case.total_dimension if args.max_iterations is None else args.max_iterations
    if limit < 1:
        parser.error('--max-iterations must be positive')
    target = (case.target_hessian, case.target_terminal_hessian, case.target_dynamics)
    with NativeFactor(case, forward=True) as factors:
        if not factors.from_fatrop_cache:
            raise RuntimeError('This tutorial requires the saved FATROP factor cache.')
        solution, stats, _ = factors.solve(target, case.rhs, args.tolerance, limit)
    residual = _residual(case, solution)
    print(f'Task: {args.problem}; case: {args.seed}; KKT dimension: {case.total_dimension}')
    print(f'Independent relative target residual: {residual:.6e}')
    print(f'GMRES correction iterations: {stats.mean_iterations:g}')
    print(f'Cached factor applications: {stats.mean_cached_backsolves:g}')
    if not np.isfinite(residual) or residual > args.tolerance:
        raise RuntimeError(f'Target tolerance not reached: {residual} > {args.tolerance}')
    print('PASS: target tolerance reached using the retained factor cache.')

if __name__ == '__main__':
    main()
