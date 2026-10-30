"""Check all96 frozen KKT inputs and immutable release files; no native loading."""
from pathlib import Path
import sys,json,hashlib,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/exp1_factor_reuse_ablation'))
from frozen_cases import load
def check():
    count=0
    for problem in ['cartpole','quadrotor','robotarm']:
        paths=sorted((ROOT/'experiments/exp1_factor_reuse_ablation/data'/problem).glob('*.npz'))
        assert len(paths)==32,(problem,len(paths))
        for path in paths:
            c=load(path);T,n,m=c.horizon,c.nx,c.nu
            assert c.target_hessian.shape==(T,n+m,n+m)
            assert c.forward_hessian.shape==(T,n+m,n+m)
            assert c.target_terminal_hessian.shape==(n,n)
            assert c.target_dynamics.shape==(T,n,n+m)
            assert c.rhs.size==c.total_dimension
            assert c.value_hessians.shape==(T+1,n,n)
            assert c.control_cholesky.shape==(T,m,m)
            for value in vars(c).values():
                if isinstance(value,np.ndarray):assert np.isfinite(value).all(),path
            assert np.allclose(c.target_hessian,c.target_hessian.swapaxes(-1,-2))
            count+=1
    for path in ROOT.rglob('*.py'):
        if not any(part in {'build','.git','.venv'} for part in path.relative_to(ROOT).parts):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    manifest=json.loads((ROOT/'artifact_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(f'PASS: {count} frozen KKT inputs; {len(manifest["sha256"])} immutable files; Python syntax.')
    return count
if __name__=='__main__':check()
