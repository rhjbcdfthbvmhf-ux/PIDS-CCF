"""Run one immutable release-derived CCF comparison; never edit original runs."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import argparse
import hashlib
import json
import sys

def main():
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'src'))
    import torch
    import data
    import train_published_gpu as train
    from published_fold_data import frozen_split,region_map,dataset_class
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=['IndianPines','PaviaU','Salinas'],required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--variant',choices=['clean','pids'],required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    row,bundle,split_digest=frozen_split(root/'published_split_bank',a.dataset,a.fold,a.data_root)
    cube,gt=data.load_cube(a.data_root,a.dataset)
    train.stratified_split=lambda gt,seed:bundle
    train.HSIPatchDataset=dataset_class(region_map(gt,row))
    out=root/('published_smoke' if a.smoke else 'published_runs')/a.dataset/('fold_'+str(a.fold))
    destination=out/a.dataset/a.variant/('seed_'+str(row['seed']))
    if destination.exists():raise FileExistsError(destination)
    args=train.build_parser().parse_args([
        '--data-root',str(a.data_root),'--output-root',str(out),'--dataset',a.dataset,
        '--variant',a.variant,'--seed',str(row['seed']),'--epochs','1' if a.smoke else '40',
        '--workers','0','--batch-size','32','--eval-batch-size','64','--patch-size','7',
        '--backbone','ccf','--learning-rate','3e-4','--validation-interval','5',
        '--patience','40','--evaluation-draws','3',
        '--selection-objective','clean' if a.variant=='clean' else 'robust']+
        (['--training-compound'] if a.variant=='pids' else []))
    args.revision_protocol={
        'type':'original published patch-based partitions','source_doi':'10.1109/LGRS.2019.2895697',
        'source_archive_sha256':json.loads((root/'published_split_bank/manifest.json').read_text())['archive_sha256'],
        'fold':a.fold,'frozen_split_sha256':split_digest,'epochs':40,'stages':1,'patch':7,
        'input_boundary':'each development region independent; zero beyond region; test masks all development regions',
        'validation':'10 percent of published development labels, classwise; distinct centers, shared development context allowed',
        'all_published_test_centers':True,'train_test_support_disjoint':True,
        'execution_device':'cuda','precision':'float32','tf32':False,'amp':False,
        'perturbation_device':'cpu','smoke_only':a.smoke,'original_results_unchanged':True}
    args.revision_protocol['execution_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ['run_published_fold.py','published_fold_data.py','train_published_gpu.py','src/data.py','src/model.py','src/pids.py']}
    print(json.dumps({'dataset':a.dataset,'fold':a.fold,'variant':a.variant,
                      'counts':[len(bundle.train),len(bundle.val),len(bundle.test)],'smoke':a.smoke}),flush=True)
    train.train(args)

if __name__=='__main__':main()
