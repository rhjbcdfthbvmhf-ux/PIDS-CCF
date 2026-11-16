"""Complete the six S2Mamba runs on CUDA and audit all eighteen native runs."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
lock = (root / 'native_gpu_queue.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
logs = root / 'native_gpu_logs'
logs.mkdir(exist_ok=True)
for fold in range(3):
    for variant in ('clean', 'pids'):
        run = root / 'native_runs/s2mamba' / f'fold_{fold}/PaviaU' / variant / f'seed_{23+fold}'
        if (run / 'metrics.json').exists():
            continue
        if run.exists():
            raise RuntimeError(f'Incomplete run requires inspection: {run}')
        with (logs / f's2mamba_{fold}_{variant}.log').open('x') as stream:
            subprocess.run([sys.executable, '-u', str(root / 'run_native_gpu.py'),
                            '--model', 's2mamba', '--fold', str(fold), '--variant', variant],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
assert len(list((root / 'native_runs').rglob('metrics.json'))) == 18
(root / 'NATIVE_RUNS_COMPLETE.json').write_text(json.dumps({
    'runs': 18, 'completed_unix': time.time(), 'prediction_audit': 'pending',
    's2mamba_execution': 'CUDA FP32; CPU perturbations'}))
subprocess.run([sys.executable, str(root / 'audit_revision_runs.py'), 'native',
                '--original-data-root', str(root.parent/'data/raw')], check=True)
