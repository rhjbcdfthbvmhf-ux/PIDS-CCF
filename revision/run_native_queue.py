from pathlib import Path
import subprocess,sys,json,time,argparse
r=Path(__file__).resolve().parent;logs=r/'native_logs';logs.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',choices=['ssftt','dbda','s2mamba'],default=['ssftt','dbda','s2mamba'])
args=parser.parse_args()
for model in args.models:
    for fold in range(3):
        for variant in ('clean','pids'):
            target=r/'native_runs'/model/f'fold_{fold}'/'PaviaU'/variant/f'seed_{23+fold}'/'metrics.json'
            if target.exists():continue
            with (logs/f'{model}_{fold}_{variant}.log').open('x') as f:
                subprocess.run([sys.executable,str(r/'run_native_baseline.py'),'--model',model,
                    '--fold',str(fold),'--variant',variant],stdout=f,stderr=subprocess.STDOUT,check=True)
completed=len(list((r/'native_runs').rglob('metrics.json')))
if completed==18:
    (r/'NATIVE_RUNS_COMPLETE.json').write_text(json.dumps({'runs':18,'completed_unix':time.time(),'prediction_audit':'pending'}))
print(json.dumps({'completed_runs':completed,'models_requested':args.models}))
