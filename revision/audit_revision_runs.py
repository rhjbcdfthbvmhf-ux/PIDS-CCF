"""Recompute all reported metrics and inspect paired inputs in added experiments."""
from pathlib import Path
import argparse,json
import sklearn
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from sklearn.metrics import confusion_matrix,f1_score,cohen_kappa_score
from sklearn.model_selection import train_test_split

R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('kind',choices=['additional','native'])
p.add_argument('--completed-only',action='store_true',help='Audit finished native runs without marking the whole experiment complete.')
p.add_argument('--original-data-root',type=Path,default=R.parent/'data/raw');a=p.parse_args()
if a.completed_only:assert a.kind=='native','Partial inspection is only implemented for native runs.'
if a.kind=='additional':assert sklearn.__version__=='1.7.2','Exact split reconstruction requires the recorded scikit-learn 1.7.2 environment.'
records=[];files=0;maxerr=0.;paired={}
paths=sorted((R/(a.kind+'_runs')).rglob('metrics.json'))
expected_runs=12 if a.kind=='additional' else 18
if a.completed_only:assert 0<len(paths)<=expected_runs,len(paths)
else:assert len(paths)==expected_runs,len(paths)
cache={}
for path in paths:
    run=path.parent;cfg=json.loads((run/'config.json').read_text());raw=json.loads(path.read_text())
    dataset=cfg['dataset'];seed=cfg['seed'];variant=cfg['variant']
    if dataset not in cache:
        if dataset=='PaviaU':
            datadir=a.original_data_root
            cube=loadmat(datadir/'PaviaU.mat')['paviaU'].astype('float32')
            gt=loadmat(datadir/'PaviaU_gt.mat')['paviaU_gt']
        else:
            cube=loadmat(R.parent/'data/additional'/f'{dataset}.mat')[dataset].astype('float32')
            gt=loadmat(R.parent/'data/additional'/f'{dataset}_gt.mat')[dataset+'_gt']
        cache[dataset]=(cube,gt)
    cube,gt=cache[dataset];classes=int(gt.max())
    split=json.loads((run/'split.json').read_text());c={k:np.asarray(split[k]) for k in ['train','val','test']}
    assert cfg['epochs']==40 and cfg['patch_size']==11
    if cfg['device']=='cuda':
        protocol=cfg['revision_protocol']
        assert a.kind=='native' and cfg['backbone']=='s2mamba'
        assert protocol['precision']=='float32' and protocol['tf32'] is False
        assert protocol['perturbation_device']=='cpu' and protocol['gpu_authorized']=='2026-11-13'
    else:
        assert cfg['device']=='cpu'
    for k,v in c.items():
        assert len(np.unique(v,axis=0))==len(v)
        assert set(gt[tuple(v.T)])==set(range(1,classes+1))
    for left,right in [('train','val'),('train','test'),('val','test')]:
        assert not set(map(tuple,c[left]))&set(map(tuple,c[right]))
        if a.kind=='native':
            mask=np.zeros(gt.shape,dtype='uint8');mask[tuple(c[left].T)]=1
            assert not maximum_filter(mask,size=21,mode='constant')[tuple(c[right].T)].any()
    if a.kind=='additional':
        coord=np.argwhere(gt>0);label=gt[tuple(coord.T)]-1
        dev,test=train_test_split(np.arange(len(coord)),train_size=.15,stratify=label,random_state=seed)
        tr,va=train_test_split(dev,test_size=.10,stratify=label[dev],random_state=seed+10000)
        for role,index in [('train',tr),('val',va),('test',test)]:assert np.array_equal(c[role],coord[index])
        pairkey=(dataset,seed)
    else:
        fold=seed-23;assert fold in range(3)
        reference=json.loads((R/'runs'/f'fold_{fold}'/'PaviaU/clean'/f'seed_{seed}'/'split.json').read_text())
        assert reference==split
        assert raw['evaluation_parameters']==json.loads((R/'runs'/f'fold_{fold}'/'PaviaU/clean'/f'seed_{seed}'/'metrics.json').read_text())['evaluation_parameters']
        pairkey=(cfg['backbone'],seed)
    paired.setdefault(pairkey,[]).append((split,raw['evaluation_parameters']))
    scale=json.loads((run/'scaling.json').read_text());spec=cube[tuple(c['train'].T)]
    assert scale['fit_scope']=='training center spectra only'
    assert np.array_equal(spec.min(0),scale['minimum']) and np.array_equal(spec.max(0),scale['maximum'])
    summaries={}
    for level in range(6):
        cond='Clean' if level==0 else f'L{level}';draws=[]
        for draw in range(1 if level==0 else 3):
            pred=np.load(run/'predictions'/f'{cond}_draw_{draw}.npz')
            truth=pred['truth'];guess=pred['prediction'];confidence=pred['confidence']
            assert np.array_equal(pred['dataset_index'],np.arange(len(c['test'])))
            assert np.array_equal(truth,gt[tuple(c['test'].T)]-1)
            assert np.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all()
            assert guess.min()>=0 and guess.max()<classes
            cm=confusion_matrix(truth,guess,labels=range(classes));correct=truth==guess;ece=0.
            for left,right in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
                m=(confidence>left)&(confidence<=right)
                if m.any():ece+=m.mean()*abs(correct[m].mean()-confidence[m].mean())
            metrics={'oa':correct.mean(),'aa':np.mean(np.diag(cm)/cm.sum(1)),
                'macro_f1':f1_score(truth,guess,labels=range(classes),average='macro'),
                'kappa':cohen_kappa_score(truth,guess),'ece':ece,
                'nll':float(-np.log(np.maximum(pred['true_probability'],1e-12)).mean())}
            for k,v in metrics.items():
                err=abs(v-raw['results'][cond]['draws'][draw][k]);maxerr=max(maxerr,err);assert err<1e-7
            assert np.array_equal(cm,raw['results'][cond]['draws'][draw]['confusion_matrix'])
            draws.append(metrics);files+=1
        summaries[cond]={k:float(np.mean([v[k] for v in draws])) for k in draws[0]}
        for k,v in summaries[cond].items():assert abs(v-raw['results'][cond][k])<1e-7
    records.append({'dataset':dataset,'seed':seed,'model':cfg['backbone'],'variant':variant,
        'counts':{k:len(v) for k,v in c.items()},'best_epoch':raw['best_epoch'],'results':summaries})
pending_pairs=[]
for key,rows in paired.items():
    if len(rows)==1 and a.completed_only:pending_pairs.append(key)
    else:assert len(rows)==2 and rows[0]==rows[1],key
aggregate={}
for rec in records:
    key=rec['dataset'] if a.kind=='additional' else rec['model']
    aggregate.setdefault(key,{})
for key in aggregate:
    for variant in ('clean','pids'):
        rows=[v for v in records if (v['dataset'] if a.kind=='additional' else v['model'])==key and v['variant']==variant]
        if a.completed_only and len(rows)!=3:continue
        assert len(rows)==3
        aggregate[key][variant]={cond:{metric:{'mean':float(np.mean([v['results'][cond][metric] for v in rows])),
            'sd':float(np.std([v['results'][cond][metric] for v in rows],ddof=1))}
            for metric in rows[0]['results'][cond]} for cond in rows[0]['results']}
report={'status':'PASS_COMPLETED_RUNS_ONLY' if a.completed_only else 'PASS',
    'runs':len(records),'expected_runs':expected_runs,'prediction_files':files,'max_metric_error':maxerr,
    'sklearn_version':sklearn.__version__,
    'training_only_scaling_recomputed':True,'matched_pairs':not pending_pairs,
    'paired_comparisons_checked':sum(len(rows)==2 for rows in paired.values()),
    'incomplete_pairs':pending_pairs,'records':records,'aggregate':aggregate}
filename=a.kind.upper()+('_PARTIAL_AUDIT.json' if a.completed_only else '_AUDIT_AND_RESULTS.json')
(R/filename).write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'runs':len(records),'prediction_files':files,'max_metric_error':maxerr,
    'summary':{k:{v:{c:{m:aggregate[k][v][c][m] for m in ['oa','macro_f1']} for c in ['Clean','L5']} for v in aggregate[k]} for k in aggregate}},indent=2))
