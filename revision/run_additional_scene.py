"""Bounded three-seed evaluation on two further public benchmark scenes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys, argparse
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
import torch,train,data
torch.set_num_threads(2);torch.set_num_interop_threads(1)
data.DATASETS.update({name:(name+'.mat',name+'_gt.mat',name,name+'_gt') for name in ('Botswana','KSC')})
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['Botswana','KSC'],required=True)
p.add_argument('--variant',choices=['clean','pids'],required=True);p.add_argument('--seed',type=int,required=True)
a=p.parse_args()
out=root/'additional_runs'
if (out/a.dataset/a.variant/f'seed_{a.seed}').exists():raise FileExistsError('Run exists')
args=train.build_parser().parse_args(['--data-root',str(root.parent/'data/additional'),'--output-root',str(out),
    '--dataset','PaviaU','--variant',a.variant,'--seed',str(a.seed),'--epochs','40','--workers','0',
    '--batch-size','32','--eval-batch-size','64','--patch-size','11','--backbone','ccf',
    '--validation-interval','5','--patience','40','--evaluation-draws','3',
    '--selection-objective','clean' if a.variant=='clean' else 'robust']+
    (['--training-compound'] if a.variant=='pids' else []))
args.dataset=a.dataset
args.revision_protocol={'type':'additional scenes, original stratified 13.5/1.5/85 center partition',
    'seeds':[23,37,51],'stages':1,'epochs':40,'evaluation_draws':3,'device':'cpu',
    'original_results_unchanged':True}
train.train(args)
