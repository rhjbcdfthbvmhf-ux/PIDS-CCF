"""Independent full prediction, protocol, preprocessing and selection audit."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix,cohen_kappa_score

R=Path(__file__).resolve().parent
SPECS={'IndianPines':('Indian_pines_corrected.mat','indian_pines_corrected','Indian_pines_gt.mat','indian_pines_gt',4),
       'PaviaU':('PaviaU.mat','paviaU','PaviaU_gt.mat','paviaU_gt',5),
       'Salinas':('Salinas_corrected.mat','salinas_corrected','Salinas_gt.mat','salinas_gt',5)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--completed-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    manifest=json.loads((R/'published_split_bank/manifest.json').read_text())
    source=json.loads((R/'PUBLISHED_INPUT_PREFLIGHT.json').read_text())
    assert source['status']=='PASS_INPUTS_NO_TRAINING'
    paths=sorted((R/('published_smoke' if a.smoke else 'published_runs')).rglob('metrics.json'))
    if a.completed_only or a.smoke:assert 0<len(paths)<=28
    else:assert len(paths)==28,len(paths)
    cache={};rows=[];pairs={};files=0;maxerr=0.
    for path in paths:
        run=path.parent;cfg=json.loads((run/'config.json').read_text());raw=json.loads(path.read_text())
        ds=cfg['dataset'];fold=cfg['revision_protocol']['fold'];seed=23+fold;variant=cfg['variant']
        assert cfg['seed']==seed and cfg['epochs']==(1 if a.smoke else 40)
        assert cfg['patch_size']==7 and cfg['backbone']=='ccf' and cfg['device']=='cuda'
        assert cfg['batch_size']==32 and cfg['eval_batch_size']==64 and cfg['evaluation_draws']==3
        assert cfg['learning_rate']==3e-4 and cfg['validation_interval']==5
        protocol=cfg['revision_protocol'];assert protocol['smoke_only']==a.smoke
        assert protocol['precision']=='float32' and not protocol['amp'] and not protocol['tf32']
        assert protocol['perturbation_device']=='cpu' and protocol['all_published_test_centers']
        if not a.smoke:
            for name,sha in protocol['execution_source_sha256'].items():
                assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,(ds,fold,name)
        if ds not in cache:
            cf,ck,gf,gk,_=SPECS[ds]
            cache[ds]=(loadmat(a.data_root/cf)[ck].astype('float32'),loadmat(a.data_root/gf)[gk])
        cube,gt=cache[ds];classes=int(gt.max())
        entry=next(v for v in manifest['folds'] if v['dataset']==ds and v['fold']==fold)
        frozen_path=R/'published_split_bank'/entry['file']
        assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==entry['sha256']==protocol['frozen_split_sha256']
        frozen=json.loads(frozen_path.read_text());split=json.loads((run/'split.json').read_text())
        for role in ['train','val','test']:assert split[role]==frozen[role]
        tr=np.array(split['train']);te=np.array(split['test']);truth_expected=gt[tuple(te.T)]-1
        scale=json.loads((run/'scaling.json').read_text());spectra=cube[tuple(tr.T)]
        assert scale['fit_scope']=='training center spectra only'
        assert np.array_equal(scale['minimum'],spectra.min(0)) and np.array_equal(scale['maximum'],spectra.max(0))
        history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in history]==list(range(1,cfg['epochs']+1))
        best=-np.inf;epoch=None
        for item in history:
            assert np.isfinite(item['total'])
            if 'validation_score' in item:
                vs=item['validation'];objective=cfg['resolved_selection_objective']
                if objective=='clean':v=[vs['Clean'][k] for k in ['oa','aa','macro_f1']]
                else:v=[vs['Clean']['oa'],.5*(vs['L3']['oa']+vs['L5']['oa']),vs['L5']['aa']]
                score=3/sum(1/max(x,1e-6) for x in v)
                assert abs(score-item['validation_score'])<1e-10
                if score>best+cfg['minimum_improvement']:best=score;epoch=item['epoch']
        assert raw['best_epoch']==epoch and abs(raw['best_validation_score']-best)<1e-10
        summaries={}
        for level in range(6):
            cond='Clean' if level==0 else 'L'+str(level);results=[]
            for draw in range(1 if level==0 else 3):
                with np.load(run/'predictions'/('%s_draw_%s.npz'%(cond,draw)),allow_pickle=False) as z:
                    y=z['truth'];pred=z['prediction'];conf=z['confidence'];tp=z['true_probability']
                    assert np.array_equal(z['dataset_index'],np.arange(len(te))) and np.array_equal(y,truth_expected)
                    assert pred.min()>=0 and pred.max()<classes
                    assert np.isfinite(conf).all() and np.isfinite(tp).all()
                    assert ((conf>=0)&(conf<=1)).all() and ((tp>=0)&(tp<=1)).all()
                    cm=confusion_matrix(y,pred,labels=np.arange(classes));diag=np.diag(cm).astype(float)
                    recall=np.divide(diag,cm.sum(1),out=np.zeros(classes),where=cm.sum(1)>0)
                    precision=np.divide(diag,cm.sum(0),out=np.zeros(classes),where=cm.sum(0)>0)
                    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(classes),where=(recall+precision)>0)
                    correct=y==pred;ece=0.
                    edges=np.linspace(0,1,16)
                    for left,right in zip(edges[:-1],edges[1:]):
                        mask=(conf>left)&(conf<=right)
                        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
                    metrics={'oa':float(correct.mean()),'aa':float(recall.mean()),'macro_f1':float(f1.mean()),
                             'worst_class_f1':float(f1.min()),'kappa':float(cohen_kappa_score(y,pred,labels=np.arange(classes))),
                             'ece':float(ece),'nll':float(-np.log(np.maximum(tp,1e-12)).mean())}
                    claimed=raw['results'][cond]['draws'][draw]
                    assert np.array_equal(cm,claimed['confusion_matrix'])
                    for key,v in metrics.items():
                        error=abs(v-claimed[key]);maxerr=max(maxerr,error);assert error<1e-7,(ds,fold,cond,key,error)
                    for key,v in [('per_class_recall',recall),('per_class_precision',precision),('per_class_f1',f1)]:
                        assert np.allclose(v,claimed[key],atol=1e-12,rtol=0)
                    results.append(metrics);files+=1
            summaries[cond]={}
            for key in results[0]:
                values=np.array([v[key] for v in results]);mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else 0.
                assert abs(raw['results'][cond][key]-mean)<1e-7 and abs(raw['results'][cond][key+'_std']-sd)<1e-7
                summaries[cond][key]=mean
        pair=(ds,fold);pairs.setdefault(pair,[]).append((split,scale,raw['evaluation_parameters']))
        rows.append({'dataset':ds,'fold':fold,'seed':seed,'variant':variant,'model':'ccf',
                     'counts':{k:len(split[k]) for k in ['train','val','test']},'best_epoch':epoch,'results':summaries})
    pending=[]
    for key,values in pairs.items():
        if len(values)==1 and (a.completed_only or a.smoke):pending.append(key)
        else:assert len(values)==2 and values[0]==values[1],key
    aggregate={}
    for ds,spec in SPECS.items():
        aggregate[ds]={}
        for variant in ['clean','pids']:
            values=[x for x in rows if x['dataset']==ds and x['variant']==variant]
            if len(values)!=spec[-1]:continue
            aggregate[ds][variant]={cond:{metric:{'mean':float(np.mean([v['results'][cond][metric] for v in values])),
                                                    'sd':float(np.std([v['results'][cond][metric] for v in values],ddof=1))}
                                        for metric in values[0]['results'][cond]} for cond in values[0]['results']}
    status='PASS_SMOKE_ONLY' if a.smoke else 'PASS_COMPLETED_RUNS_ONLY' if a.completed_only else 'PASS'
    report={'status':status,'runs':len(rows),'expected_runs':28,'prediction_files':files,'max_metric_error':maxerr,
            'matched_pairs':sum(len(x)==2 for x in pairs.values()),'incomplete_pairs':pending,'records':rows,'aggregate':aggregate,
            'source_archive_sha256':manifest['archive_sha256'],'validation_selection_recomputed':True,
            'all_published_test_centers':True,'training_only_scaling_recomputed':True}
    name='PUBLISHED_SMOKE_AUDIT.json' if a.smoke else 'PUBLISHED_PARTIAL_AUDIT.json' if a.completed_only else 'PUBLISHED_AUDIT_AND_RESULTS.json'
    (R/name).write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['status','runs','prediction_files','max_metric_error','matched_pairs']}))

if __name__=='__main__':main()
