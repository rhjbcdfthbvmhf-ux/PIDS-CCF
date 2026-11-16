"""Independent reconstruction of fold geometry, labels and metrics from NPZ."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from scipy.stats import t
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

root=Path(__file__).resolve().parent
gt=loadmat(root.parent/'data/raw/PaviaU_gt.mat')['paviaU_gt']
rr,cc=np.indices(gt.shape);groups=(rr//24+2*(cc//24))%5
records=[];tests=[];max_error=0.;files=0
for fold in range(5):
    splits=[]
    for variant in ('clean','pids'):
        run=root/'runs'/f'fold_{fold}'/'PaviaU'/variant/f'seed_{23+fold}'
        payload=json.loads((run/'metrics.json').read_text())
        conf=json.loads((run/'config.json').read_text())
        assert conf['revision_protocol']['fold']==fold and conf['epochs']==40 and conf['device']=='cpu'
        sp=json.loads((run/'split.json').read_text());splits.append(sp)
        coords={role:np.asarray(sp[role]) for role in ('train','val','test')}
        for role,c in coords.items():
            assert len(np.unique(c,axis=0))==len(c)
            assert set(gt[tuple(c.T)])==set(range(1,10))
        assert np.all(groups[tuple(coords['test'].T)]==fold)
        for a,b in [('train','val'),('train','test'),('val','test')]:
            mask=np.zeros(gt.shape,dtype='uint8');mask[tuple(coords[a].T)]=1
            assert not maximum_filter(mask,size=21,mode='constant')[tuple(coords[b].T)].any()
        if variant=='clean':tests.append(set(map(tuple,coords['test'])))
        scal=json.loads((run/'scaling.json').read_text())
        assert scal['fit_scope']=='training center spectra only'
        summaries={}
        for level in range(6):
            condition='Clean' if level==0 else f'L{level}'
            draws=[]
            for draw in range(1 if level==0 else 3):
                p=run/'predictions'/f'{condition}_draw_{draw}.npz';a=np.load(p)
                truth=a['truth'];pred=a['prediction'];confidence=a['confidence'];index=a['dataset_index']
                assert np.array_equal(index,np.arange(len(coords['test'])))
                assert np.array_equal(truth,gt[tuple(coords['test'].T)]-1)
                assert np.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all()
                cm=confusion_matrix(truth,pred,labels=range(9));correct=truth==pred
                ece=0.
                for left,right in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
                    m=(confidence>left)&(confidence<=right)
                    if m.any():ece+=m.mean()*abs(correct[m].mean()-confidence[m].mean())
                d={'oa':correct.mean(),'aa':np.mean(np.diag(cm)/cm.sum(1)),
                   'macro_f1':f1_score(truth,pred,labels=range(9),average='macro'),
                   'kappa':cohen_kappa_score(truth,pred), 'ece':ece,
                   'nll':float(-np.log(np.maximum(a['true_probability'],1e-12)).mean())}
                for k,v in d.items():
                    err=abs(v-payload['results'][condition]['draws'][draw][k]);max_error=max(max_error,err)
                    assert err<1e-7,(fold,variant,condition,k,err)
                assert np.array_equal(cm,payload['results'][condition]['draws'][draw]['confusion_matrix'])
                draws.append(d);files+=1
            summaries[condition]={k:float(np.mean([d[k] for d in draws])) for k in draws[0]}
            for k,v in summaries[condition].items():assert abs(v-payload['results'][condition][k])<1e-7
        records.append({'fold':fold+1,'variant':variant,'counts':{k:len(v) for k,v in coords.items()},
                        'best_epoch':payload['best_epoch'],'results':summaries})
    assert splits[0]==splits[1],f'unmatched split {fold}'
    a=json.loads((root/'runs'/f'fold_{fold}'/'PaviaU/clean'/f'seed_{23+fold}'/'metrics.json').read_text())
    b=json.loads((root/'runs'/f'fold_{fold}'/'PaviaU/pids'/f'seed_{23+fold}'/'metrics.json').read_text())
    assert a['evaluation_parameters']==b['evaluation_parameters']
for i in range(5):
    for j in range(i):assert not tests[i]&tests[j]
aggregate={}
for variant in ('clean','pids'):
    aggregate[variant]={}
    rows=[r for r in records if r['variant']==variant]
    for cond in ('Clean','L1','L2','L3','L4','L5'):
        aggregate[variant][cond]={k:{'mean':float(np.mean([r['results'][cond][k] for r in rows])),
            'sd':float(np.std([r['results'][cond][k] for r in rows],ddof=1))} for k in rows[0]['results'][cond]}
paired={cond:[records[2*i+1]['results'][cond]['oa']-records[2*i]['results'][cond]['oa'] for i in range(5)] for cond in ('Clean','L5')}
report={'status':'PASS','runs':len(records),'prediction_files':files,'max_metric_error':max_error,
        'complete_class_coverage':True,'within_fold_patch_support_disjoint':True,'test_centers_across_folds_disjoint':True,
        'matched_augmentation_parameters':True,'records':records,'aggregate':aggregate,'paired_oa':paired,
        'inference':'Descriptive fold mean and sample SD; no independent-fold t-test is claimed.'}
(root/'CV_AUDIT_AND_RESULTS.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ('status','runs','prediction_files','max_metric_error','aggregate','paired_oa')},indent=2))
