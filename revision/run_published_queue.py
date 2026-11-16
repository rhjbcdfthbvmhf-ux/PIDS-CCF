"""Run the fixed 28 CCF jobs sequentially; refuse incomplete existing output."""
from pathlib import Path
import argparse
import fcntl
import json
import subprocess
import sys
import time
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
lock=(root/'published_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
preflight=json.loads((root/'PUBLISHED_INPUT_PREFLIGHT.json').read_text())
assert preflight['status']=='PASS_INPUTS_NO_TRAINING' and len(preflight['folds'])==14
logs=root/'published_logs';logs.mkdir(exist_ok=True)
for ds,n in [('IndianPines',4),('PaviaU',5),('Salinas',5)]:
    for fold in range(n):
        for variant in ['clean','pids']:
            run=root/'published_runs'/ds/('fold_'+str(fold))/ds/variant/('seed_'+str(23+fold))
            if (run/'metrics.json').exists():continue
            if run.exists():raise RuntimeError('Incomplete run must be inspected: '+str(run))
            print(ds,fold,variant,'START',time.time(),flush=True)
            with (logs/('%s_%s_%s.log'%(ds,fold,variant))).open('x') as f:
                subprocess.run([sys.executable,'-u',str(root/'run_published_fold.py'),
                    '--data-root',str(a.data_root),'--dataset',ds,'--fold',str(fold),'--variant',variant],
                    stdout=f,stderr=subprocess.STDOUT,check=True)
assert len(list((root/'published_runs').rglob('metrics.json')))==28
(root/'PUBLISHED_RUNS_COMPLETE.json').write_text(json.dumps({'runs':28,'completed_unix':time.time(),'prediction_audit':'required'}))
audit=root/'audit_published_results.py'
if audit.exists():subprocess.run([sys.executable,str(audit),'--data-root',str(a.data_root)],check=True)
