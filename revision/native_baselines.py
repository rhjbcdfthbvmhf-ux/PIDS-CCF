"""Adapters around author models; retained classifier heads and architecture bodies."""
from pathlib import Path
import ast,sys,types,math,copy,time
from functools import partial
from typing import Optional,Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange,repeat
R=Path(__file__).resolve().parent/'third_party'

def definitions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)

def build_native(name,bands,classes,pca=None):
    if name=='ssftt':
        path=R/'ssftt/cls_SSFTT_IP/SSFTTnet.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        # Drop unused visualization imports; preserve every network definition.
        tree.body=[n for n in tree.body if not isinstance(n,ast.Import) or all(a.name not in ['PIL','torchvision'] for a in n.names)]
        namespace={'__name__':'official_ssftt'}
        exec(compile(tree,str(path),'exec'),namespace)
        native=namespace['SSFTTnet'](num_classes=classes)
        return Wrapped(PCASpectralInput(native,pca),classes)
    if name=='dbda':
        sys.path.insert(0,str(R/'dbda/DBDA'));sys.path.insert(0,str(R/'dbda/global_module'))
        import network
        return Wrapped(DBDAInput(network.DBDA_network_MISH(bands,classes)),classes)
    if name=='s2mamba':
        namespace=dict(torch=torch,nn=nn,F=F,rearrange=rearrange,repeat=repeat,math=math,copy=copy,
                       time=time,partial=partial,Optional=Optional,Callable=Callable,checkpoint=checkpoint,np=np)
        definitions(R/'s2mamba/timm_drop_source.py',['drop_path','DropPath'],namespace)
        definitions(R/'s2mamba/selective_scan_reference_source.py',['selective_scan_ref'],namespace)
        namespace['selective_scan_fn_v1']=namespace['selective_scan_ref']
        namespace['trunc_normal_']=nn.init.trunc_normal_
        path=R/'s2mamba/s2mamba.py';tree=ast.parse(path.read_text(encoding='utf-8'))
        # The paper model body is unmodified; the published pure-PyTorch scan
        # replaces its CUDA execution backend for the bounded CPU experiment.
        tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
        exec(compile(tree,str(path),'exec'),namespace)
        native=namespace['S2Mamba'](in_chans=bands,patch=11,num_classes=classes,
                    depths=[1],dims=[64],drop_path_rate=.1,attn_drop_rate=.1)
        return Wrapped(native,classes)
    raise ValueError(name)

class PCASpectralInput(nn.Module):
    def __init__(self,network,pca):
        super().__init__();self.network=network
        weights=pca.components_/np.sqrt(pca.explained_variance_[:,None])
        self.register_buffer('weights',torch.tensor(weights[:,:,None,None],dtype=torch.float32))
        self.register_buffer('bias',torch.tensor(-weights@pca.mean_,dtype=torch.float32))
    def forward(self,x):return self.network(F.conv2d(x,self.weights,self.bias).unsqueeze(1))

class DBDAInput(nn.Module):
    def __init__(self,network):super().__init__();self.network=network
    def forward(self,x):return self.network(x.permute(0,2,3,1).unsqueeze(1).contiguous())

class Wrapped(nn.Module):
    auxiliary=False
    def __init__(self,network,classes):
        super().__init__();self.content=network;self.classifier=nn.Identity()
    def forward(self,x,adversarial_scale=0.):
        logits=self.content(x);return {'content':logits,'logits':logits}
