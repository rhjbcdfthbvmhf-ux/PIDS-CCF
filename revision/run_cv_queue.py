"""Run the approved bounded CPU experiment sequentially; never overwrite runs."""
from pathlib import Path
import subprocess, sys, json, time

root = Path(__file__).resolve().parent
logdir = root / 'logs'
logdir.mkdir(exist_ok=True)
jobs = [(fold, variant) for fold in range(5) for variant in ('clean', 'pids')]
for fold, variant in jobs:
    metrics = root / 'runs' / f'fold_{fold}' / 'PaviaU' / variant / f'seed_{23+fold}' / 'metrics.json'
    if metrics.exists():
        continue
    with (logdir / f'fold_{fold}_{variant}.log').open('x') as log:
        result = subprocess.run([sys.executable, str(root/'run_small_cv.py'),
            '--fold', str(fold), '--variant', variant, '--epochs', '40'], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'fold {fold}, {variant}: exit {result.returncode}; inspect log, do not overwrite')
    if not metrics.exists():
        raise RuntimeError('Successful process did not produce metrics')
(root/'CV_RUNS_COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),
    'runs':len(jobs), 'device':'cpu', 'prediction_audit':'pending'}, indent=2))
