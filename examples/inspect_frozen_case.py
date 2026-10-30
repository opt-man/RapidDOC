"""Inspect one recorded real-OCP KKT instance, without native dependencies."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'experiments/exp1_factor_reuse_ablation'))
from frozen_cases import load
case=load(root/'experiments/exp1_factor_reuse_ablation/data/cartpole/seed_000.npz')
print(f'{case.problem}: horizon={case.horizon}, nx={case.nx}, nu={case.nu}, KKT dimension={case.total_dimension}')
print('Target stage Hessians:',case.target_hessian.shape)
print('Retained control factors:',case.control_cholesky.shape)
print('This inspects a frozen input; it does not run an optimization.')
