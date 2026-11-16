"""Independently check frozen centers, masks, scaling, and actual dataset tensors."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

def main():
    root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
    import data
    from published_fold_data import frozen_split,region_map,dataset_class
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
    bank=root/'published_split_bank';manifest=json.loads((bank/'manifest.json').read_text());reports=[]
    for entry in manifest['folds']:
        ds,fold=entry['dataset'],entry['fold'];row,split,_=frozen_split(bank,ds,fold,a.data_root)
        cube,gt=data.load_cube(a.data_root,ds);regions=region_map(gt,row)
        coords=[split.train,split.val,split.test]
        sets=[{tuple(x) for x in q} for q in coords]
        assert all(not (sets[i]&sets[j]) for i,j in [(0,1),(0,2),(1,2)])
        dev=set(map(tuple,np.argwhere((gt>0)&(regions>=0))))
        test=set(map(tuple,np.argwhere((gt>0)&(regions<0))))
        assert sets[0]|sets[1]==dev and sets[2]==test
        scaled,record=data.train_only_scale(cube,split.train)
        tr=cube[tuple(split.train.T)];lo=tr.min(0);hi=tr.max(0)
        assert np.array_equal(lo,record['minimum']) and np.array_equal(hi,record['maximum'])
        expected=np.clip((cube-lo)/np.maximum(hi-lo,1e-6),0,1)
        assert np.array_equal(scaled,expected)
        Dataset=dataset_class(regions);checks=0
        for q in coords:
            dataset=Dataset(scaled,gt,q,7)
            # Fixed raster-spanning cases include both interior and edge centers.
            samples=np.unique(np.linspace(0,len(q)-1,min(64,len(q)),dtype=int))
            for i in samples:
                r,c=q[i];rid=regions[r,c];wanted=np.zeros((cube.shape[2],7,7),dtype=np.float32)
                for dy in range(-3,4):
                    for dx in range(-3,4):
                        y,x=r+dy,c+dx
                        if 0<=y<gt.shape[0] and 0<=x<gt.shape[1] and regions[y,x]==rid:
                            wanted[:,dy+3,dx+3]=scaled[y,x]
                actual,label,index=dataset[int(i)]
                assert np.array_equal(actual.numpy(),wanted)
                assert int(label)==int(gt[r,c])-1 and int(index)==i
                checks+=1
        reports.append({'dataset':ds,'fold':fold,'samples_checked':checks,'counts':[len(x) for x in coords],
                        'fixed_coordinates':True,'train_only_scaling':True,'independent_region_boundary':True,
                        'all_released_centers_retained':True})
        print(ds,fold,'PASS',flush=True)
    report={'status':'PASS_INPUTS_NO_TRAINING','folds':reports,'checks':sum(x['samples_checked'] for x in reports)}
    (root/'PUBLISHED_INPUT_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(report['status'])

if __name__=='__main__':main()
