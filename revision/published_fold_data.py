"""Load frozen published regions with isolated zero-padded model inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import SplitBundle

def frozen_split(bank, dataset, fold, data_root):
    bank=Path(bank);manifest=json.loads((bank/'manifest.json').read_text())
    entry=next(x for x in manifest['folds'] if x['dataset']==dataset and x['fold']==fold)
    path=bank/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    row=json.loads(path.read_text());assert row['input_patch']==7 and row['seed']==23+fold
    from data import DATASETS
    cube,gt,_,_=DATASETS[dataset]
    for name,key in [(cube,'source_cube_sha256'),(gt,'source_gt_sha256')]:
        h=hashlib.sha256()
        with (Path(data_root)/name).open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):h.update(block)
        assert h.hexdigest()==row[key],(dataset,name)
    bundle=SplitBundle(*(np.asarray(row[k],dtype=np.int32) for k in ['train','val','test']),seed=row['seed'])
    return row,bundle,entry['sha256']

def dataset_class(regions):
    class PublishedPatchDataset(torch.utils.data.Dataset):
        def __init__(self,cube,gt,coords,patch_size):
            assert patch_size==7
            self.coords=np.asarray(coords,dtype=np.int32)
            self.labels=gt[tuple(self.coords.T)].astype(np.int64)-1
            self.margin=patch_size//2;self.patch_size=patch_size
            padded=np.pad(cube,((self.margin,self.margin),(self.margin,self.margin),(0,0)),mode='constant')
            self.padded=torch.from_numpy(np.ascontiguousarray(padded.transpose(2,0,1)))
            self.region=np.pad(regions,((self.margin,self.margin),(self.margin,self.margin)),constant_values=-2)
            self.center_regions=regions[tuple(self.coords.T)]
        def __len__(self):return len(self.coords)
        def __getitem__(self,index):
            r,c=self.coords[index];n=self.patch_size
            # Every training region is an independent image. The test view has
            # all development regions removed, as in the published archive.
            mask=torch.from_numpy(self.region[r:r+n,c:c+n]==self.center_regions[index])
            patch=self.padded[:,r:r+n,c:c+n]*mask.unsqueeze(0)
            return patch,torch.tensor(self.labels[index]),torch.tensor(index)
    return PublishedPatchDataset

def region_map(gt,row):
    regions=np.full(gt.shape,-1,dtype=np.int32)
    for i,p in enumerate(row['patches']):
        r,c=p['origin'];h,w,_=p['shape']
        assert not (regions[r:r+h,c:c+w]>=0).any()
        regions[r:r+h,c:c+w]=i
    return regions
