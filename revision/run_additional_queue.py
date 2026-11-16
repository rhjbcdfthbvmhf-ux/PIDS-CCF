from pathlib import Path
import sys,subprocess,json,time
r=Path(__file__).resolve().parent
logs=r/'additional_logs';logs.mkdir(exist_ok=True)
for dataset in ('Botswana','KSC'):
    for seed in (23,37,51):
        for variant in ('clean','pids'):
            result=r/'additional_runs'/dataset/variant/f'seed_{seed}'/'metrics.json'
            if result.exists():continue
            with (logs/f'{dataset}_{seed}_{variant}.log').open('x') as f:
                subprocess.run([sys.executable,str(r/'run_additional_scene.py'),'--dataset',dataset,
                    '--seed',str(seed),'--variant',variant],stdout=f,stderr=subprocess.STDOUT,check=True)
(r/'ADDITIONAL_RUNS_COMPLETE.json').write_text(json.dumps({'runs':12,'completed_unix':time.time(),'prediction_audit':'pending'}))
