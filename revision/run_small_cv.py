"""Small-budget five-fold spatial-block validation, independent of locked runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch,train
from data import SplitBundle
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

def split(gt,fold):
    rr,cc=np.indices(gt.shape); groups=(rr//24+2*(cc//24))%5
    test=(gt>0)&(groups==fold)
    forbidden=maximum_filter(test.astype('uint8'),size=21,mode='constant')>0
    val=(gt>0)&(groups==(fold+1)%5)&~forbidden
    tr=(gt>0)&(groups!=fold)&(groups!=(fold+1)%5)&~forbidden
    tr &= maximum_filter(val.astype('uint8'),size=21,mode='constant')==0
    rng=np.random.default_rng(23+fold)
    arrays=[]
    for mask,cap in [(tr,128),(val,64),(test,256)]:
        chunks=[]
        for cls in range(1,int(gt.max())+1):
            coords=np.argwhere(mask&(gt==cls))
            if not len(coords):raise ValueError(f'fold {fold}, missing class {cls}')
            chunks.append(coords[rng.permutation(len(coords))[:cap]])
        arrays.append(np.concatenate(chunks).astype('int32'))
    for left,right in [(arrays[0],arrays[1]),(arrays[0],arrays[2]),(arrays[1],arrays[2])]:
        a=np.zeros_like(gt,dtype='uint8'); a[tuple(left.T)]=1
        assert not np.any(maximum_filter(a,size=21,mode='constant')[tuple(right.T)])
    return SplitBundle(*arrays,seed=23+fold)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5))
    p.add_argument('--variant',required=True,choices=['clean','pids']);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    data=Path(__file__).resolve().parents[1]/'data/raw'
    bundle=split(loadmat(data/'PaviaU_gt.mat')['paviaU_gt'],a.fold)
    train.stratified_split=lambda gt,seed:bundle
    out=ROOT/('smoke' if a.smoke else 'runs')/f'fold_{a.fold}'
    destination=out/'PaviaU'/a.variant/f'seed_{23+a.fold}'
    if destination.exists():raise FileExistsError(destination)
    args=train.build_parser().parse_args(['--data-root',str(data),'--output-root',str(out),
        '--dataset','PaviaU','--variant',a.variant,'--seed',str(23+a.fold),
        '--epochs',str(a.epochs),'--workers','0','--batch-size','32','--eval-batch-size','64',
        '--patch-size','11','--backbone','ccf','--learning-rate','3e-4',
        '--validation-interval','5','--patience','40','--evaluation-draws','3',
        '--selection-objective','clean' if a.variant=='clean' else 'robust']+
        (['--training-compound'] if a.variant=='pids' else []))
    args.revision_protocol={'type':'five-fold spatial-block capped evaluation','standard_published_folds':False,
        'block_size':24,'fold':a.fold,'training_cap_per_class':128,'validation_cap_per_class':64,
        'test_cap_per_class':256,'patch_size':11,'center_exclusion_radius':10,'device':'CPU',
        'stages':1,'prespecified_epochs':a.epochs,'statistical_unit':'spatial fold',
        'selection_of_block_size':'first candidate with complete class support in geometry-only preflight',
        'original_results_unchanged':True}
    print(json.dumps({'device':'cpu','fold':a.fold,'variant':a.variant,'counts':[len(x) for x in [bundle.train,bundle.val,bundle.test]]}),flush=True)
    train.train(args)
if __name__=='__main__':main()
