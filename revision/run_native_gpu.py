import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,argparse,json,time
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'deps'));sys.path.insert(0,str(r/'src'))
import numpy as np,torch,data
import train_native_gpu as train
from sklearn.decomposition import PCA
from native_baselines import build_native
from run_small_cv import split
os.environ['CUDA_VISIBLE_DEVICES']='0'
p=argparse.ArgumentParser();p.add_argument('--model',choices=['ssftt','dbda','s2mamba'],required=True)
p.add_argument('--fold',type=int,choices=range(3),default=0)
p.add_argument('--variant',choices=['clean','pids'],default='clean');p.add_argument('--check-only',action='store_true')
a=p.parse_args();data_root=Path(__file__).resolve().parents[1]/'data/raw'
assert a.model=='s2mamba' and torch.cuda.is_available()
cube,gt=data.load_cube(data_root,'PaviaU');bundle=split(gt,a.fold)
scaled,_=data.train_only_scale(cube,bundle.train)
pca=None
if a.model=='ssftt':
    pca=PCA(n_components=30,whiten=True,svd_solver='full').fit(scaled[tuple(bundle.train.T)])
def builder(**kwargs):return build_native(a.model,kwargs['bands'],kwargs['classes'],pca)
if a.check_only:
    data.seed_everything(23);net=builder(bands=cube.shape[-1],classes=int(gt.max()))
    x=torch.rand(2,cube.shape[-1],11,11);start=time.time();out=net(x)['logits']
    assert out.shape==(2,9) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
    if a.model=='ssftt':
        sample=x.permute(0,2,3,1).reshape(-1,cube.shape[-1]).numpy()
        expected=pca.transform(sample)
        actual=torch.nn.functional.conv2d(x,net.content.weights,net.content.bias).permute(0,2,3,1).reshape(-1,30).numpy()
        assert np.allclose(actual,expected,atol=.001,rtol=.0001)
    print(json.dumps({'model':a.model,'forward_backward':'PASS','seconds':time.time()-start,'parameters':sum(v.numel() for v in net.parameters())}))
    raise SystemExit
out=r/'native_runs'/a.model/f'fold_{a.fold}'
if (out/'PaviaU'/a.variant/f'seed_{23+a.fold}').exists():raise FileExistsError('Run exists')
train.stratified_split=lambda gt,seed:bundle;train.build_model=builder
args=train.build_parser().parse_args(['--data-root',str(data_root),'--output-root',str(out),
    '--dataset','PaviaU','--variant',a.variant,'--seed',str(23+a.fold),'--epochs','40','--workers','0',
    '--batch-size','32','--eval-batch-size','64','--patch-size','11','--backbone','ccf',
    '--validation-interval','5','--patience','40','--evaluation-draws','3',
    '--selection-objective','clean' if a.variant=='clean' else 'robust']+
    (['--training-compound'] if a.variant=='pids' else []))
args.backbone=a.model
args.revision_protocol={'type':'matched additional author-model comparison','folds':[0,1,2],
    'epochs':40,'patch':11,'cpu':False,'classifier':'native author head',
    'execution_device':'cuda','precision':'float32','tf32':False,
    'perturbation_device':'cpu','gpu_authorized':'2026-11-13',
    'restart_policy':'incomplete CPU run archived; restart all 40 epochs from original seed',
    'ssftt_pca':'30 whitened components fitted on training centers; physical perturbation precedes PCA',
    's2mamba_scan':'official mamba v1.2.0 pure-PyTorch selective_scan_ref',
    'original_results_unchanged':True}
train.train(args)
