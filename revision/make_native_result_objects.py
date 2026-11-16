"""Create a table only after all added native-model predictions pass audit."""
from pathlib import Path
import json,csv,statistics,hashlib

R=Path(__file__).resolve().parents[1]
E=R/'revision'
O=R/'revision/result_objects'
native_path=E/'NATIVE_AUDIT_AND_RESULTS.json'
if not native_path.exists():
    raise SystemExit('Native-model audit is not available. No result objects were generated.')
n=json.loads(native_path.read_text());c=json.loads((E/'CV_AUDIT_AND_RESULTS.json').read_text())
assert n['status']==c['status']=='PASS'
assert n['runs']==18 and n['prediction_files']==288
assert c['runs']==10 and c['prediction_files']==160
assert n['matched_pairs'] and n['training_only_scaling_recomputed']
assert c['within_fold_patch_support_disjoint'] and c['matched_augmentation_parameters']
records=[]
for r in c['records']:
    if r['fold']<=3:
        records.append(dict(model='ccf',seed=22+r['fold'],variant=r['variant'],
                            counts=r['counts'],results=r['results']))
records.extend(n['records'])
models=['ssftt','dbda','s2mamba','ccf'];names=['SSFTT','DBDA','S2Mamba','CCF']
assert len(records)==24
summary={}
for model in models:
    summary[model]={}
    for variant in ['clean','pids']:
        rows=sorted([r for r in records if r['model']==model and r['variant']==variant],key=lambda r:r['seed'])
        assert [r['seed'] for r in rows]==[23,24,25]
        for row in rows:
            ref=next(r for r in records if r['model']=='ccf' and r['variant']==variant and r['seed']==row['seed'])
            assert row['counts']==ref['counts']
        summary[model][variant]={condition:{metric:{
            'mean':statistics.mean(r['results'][condition][metric] for r in rows),
            'sd':statistics.stdev(r['results'][condition][metric] for r in rows)}
            for metric in rows[0]['results'][condition]} for condition in rows[0]['results']}

def cell(model,variant,condition,metric):
    v=summary[model][variant][condition][metric]
    return '$'+f"{100*v['mean']:.2f}\\pm{100*v['sd']:.2f}"+'$'

columns=[('clean','Clean','oa'),('clean','L5','oa'),('pids','Clean','oa'),
         ('pids','L5','oa'),('pids','L5','macro_f1')]
lines=[r'\begin{table}[!htbp]',r'\caption{\label{tab:native-models}Classification performance of additional comparison models.}',
       r'\centering',r'\JARStableformat',r'\begin{tabularx}{\textwidth}{lYYYYY}',
       r'\toprule',r'\ifdefined\JARSClean\else\rowcolor{yellow}\fi & \multicolumn{2}{c}{Clean training} & \multicolumn{3}{c}{PIDS training} \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}',
       r'\ifdefined\JARSClean\else\rowcolor{yellow}\fi Model & Clean OA (\%) & L5 OA (\%) & Clean OA (\%) & L5 OA (\%) & L5 macro-F1 (\%) \\',r'\midrule']
for model,name in zip(models,names):
    lines.append(r'\ifdefined\JARSClean\else\rowcolor{yellow}\fi '+name+' & '+' & '.join(cell(model,*args) for args in columns)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
O.mkdir(exist_ok=True)
(O/'table10_native_models.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (O/'native_models_all_conditions.csv').open('w',newline='',encoding='utf8') as stream:
    writer=csv.writer(stream);writer.writerow(['model','training','condition','metric','mean','sample_sd','folds'])
    for model in models:
        for variant in ['clean','pids']:
            for condition,metrics in summary[model][variant].items():
                for metric,v in metrics.items():writer.writerow([model,variant,condition,metric,v['mean'],v['sd'],3])
payload={'status':'AUDITED_TABLE_PREPARED_NOT_INSERTED','fold_seeds':[23,24,25],
         'runs':24,'summary':summary,
         'l5_oa_training_gain_pp':{m:100*(summary[m]['pids']['L5']['oa']['mean']-summary[m]['clean']['L5']['oa']['mean']) for m in models},
         'audit_source_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest()}
(O/'native_model_table_source.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({k:v for k,v in payload.items() if k!='summary'},indent=2))
