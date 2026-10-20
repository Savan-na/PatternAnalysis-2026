"""Fixed Validation-only four-checkpoint pictures, separate v2 experiment area."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hipmri_rewrite.improved_v2 import ImprovedUNet3DV2
from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
from hipmri_rewrite.inference import predict_volume
from hipmri_rewrite.metrics import confusion
from hipmri_rewrite.splits import discover
from tools.review_g5_validation_failures import load_selected

NAMES=['Background','Body','Bones','Bladder','Rectum','Prostate']
CASES=['S022_Week2','K018_Week4','M030_Week0','J026_Week4']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use an allocated GPU job')
    out=ROOT/'results/experiments/improved_3d_v2/case_studies/validation/review_45_50'
    if out.exists(): raise ValueError('Output exists; inspect before rerunning')
    torch.set_num_threads(2)
    specs=[('Standard 3D','g5_standard_20261018','standard_3d',UNet3D),
           ('Improved v1','g5_improved_20261018','improved_3d',ImprovedUNet3D)]
    models={}; records={}; provenance={}
    for label,folder,identity,ctor in specs:
        model,record,receipt=load_selected(ROOT/'logs/training'/folder,identity,ctor)
        models[label]=model; records[label]=record; provenance[label]=receipt
    run=ROOT/'logs/training/improved_3d_v2_ds_20261020'
    manifest=json.loads((run/'manifest.json').read_text())
    checks={'v2_source_sha256':'hipmri_rewrite/improved_v2.py',
            'model_source_sha256':'hipmri_rewrite/models/improved_3d.py',
            'data_sha256':'hipmri_rewrite/data.py',
            'inference_sha256':'hipmri_rewrite/inference.py',
            'splits_sha256':'hipmri_rewrite/splits.py'}
    assert all(sha(ROOT/'code'/p)==manifest[k] for k,p in checks.items())
    assert sha(run/'config_frozen.json')==manifest['config_sha256']
    for e in [45,50]:
        label=f'v2 epoch{e}'
        path=run/f'epoch_{e:03d}.pt'; checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==e and checkpoint['model']=='improved_3d_v2'
        assert checkpoint['config_sha256']==manifest['config_sha256']
        record=json.loads((run/f'epoch_{e:03d}.json').read_text())
        assert record['epoch']==e
        if e==45: assert record['best_epoch']==45
        model=ImprovedUNet3DV2().cuda().eval(); model.load_state_dict(checkpoint['model_state'])
        models[label]=model; records[label]=record
        provenance[label]={'epoch':e,'checkpoint_sha256':sha(path),
            'record_sha256':sha(run/f'epoch_{e:03d}.json'),'manifest_sha256':sha(run/'manifest.json')}
    cases={c.identifier:c for c in discover(args.dataset_root)['validation']}
    out.mkdir(parents=True)
    receipts=[]
    for case_id in CASES:
        case=cases[case_id]; volume=zscore_nonzero(downsample(load_case(case)))
        predictions={}
        for name,model in models.items():
            pred=predict_volume(model,volume,mode='3d',device='cuda')
            saved=next(r for r in records[name]['validation']['cases'] if r['case']==case_id)
            assert confusion(volume.mask,pred).tolist()==saved['confusion'], f'{case_id}/{name}: replay mismatch'
            predictions[name]=pred
        nz=volume.image[volume.image!=0]; low,high=np.percentile(nz,[1,99])
        fig,axes=plt.subplots(4,6,figsize=(23,14),layout='constrained')
        for j,label in enumerate(NAMES):
            z=int(np.argmax((volume.mask==j).sum(axis=(0,1))))
            for i,(name,pred) in enumerate(predictions.items()):
                ax=axes[i,j]; truth=(volume.mask[:,:,z]==j).T; found=(pred[:,:,z]==j).T
                ax.imshow(volume.image[:,:,z].T,cmap='gray',origin='lower',vmin=low,vmax=high)
                for mask,color in [(truth,'cyan'),(found,'orange')]:
                    if mask.any() and not mask.all(): ax.contour(mask.astype(float),[.5],colors=[color],linewidths=.8)
                rgba=np.zeros((*truth.shape,4)); rgba[found & ~truth]=[1,0,0,.38]; rgba[truth & ~found]=[0,.3,1,.38]
                ax.imshow(rgba,origin='lower'); ax.set_title(f'{name}\n{label}, axial z={z}',fontsize=10); ax.axis('off')
        fig.suptitle(f'Validation {case_id} | all six classes | common GT maximum-area slice per class',fontsize=15)
        fig.legend([Line2D([0],[0],color=c) for c in ['cyan','orange','red','blue']],
                   ['GT contour','Prediction contour','False positive','False negative'],loc='outside lower center',ncol=4)
        fig.savefig(out/f'{case_id}_six_classes.png',dpi=140); plt.close(fig)
        distance=distance_transform_edt(volume.mask!=1,sampling=volume.spacing_mm)
        false=(volume.mask==0)&(predictions['v2 epoch45']==1)
        center=(np.unravel_index(np.argmax(np.where(false,distance,-1)),false.shape)
                if false.any() else tuple(n//2 for n in false.shape))
        x,y,z=center
        selectors=[(slice(None),slice(None),z),(slice(None),y,slice(None)),(x,slice(None),slice(None))]
        aspects=[volume.spacing_mm[1]/volume.spacing_mm[0],volume.spacing_mm[2]/volume.spacing_mm[0],volume.spacing_mm[2]/volume.spacing_mm[1]]
        fig,axes=plt.subplots(4,3,figsize=(13,14),layout='constrained')
        for i,(name,pred) in enumerate(predictions.items()):
            for j,(selection,orientation,aspect) in enumerate(zip(selectors,['Axial','Coronal','Sagittal'],aspects)):
                ax=axes[i,j]; truth=(volume.mask[selection]==1).T; found=(pred[selection]==1).T
                ax.imshow(volume.image[selection].T,cmap='gray',origin='lower',vmin=low,vmax=high,aspect=aspect)
                for mask,color in [(truth,'cyan'),(found,'orange')]:
                    if mask.any() and not mask.all(): ax.contour(mask.astype(float),[.5],colors=[color],linewidths=.8)
                rgba=np.zeros((*truth.shape,4)); rgba[found & ~truth]=[1,0,0,.4]; rgba[truth & ~found]=[0,.3,1,.4]
                ax.imshow(rgba,origin='lower',aspect=aspect); ax.set_title(name+' | '+orientation,fontsize=10); ax.axis('off')
        fig.suptitle(f'Validation {case_id} | Body | common planes through v2 epoch45 most distant Background-to-Body FP\nGT cyan / prediction orange / FP red / FN blue')
        fig.savefig(out/f'{case_id}_body_three_planes.png',dpi=150); plt.close(fig)
        receipts.append({'case':case_id,'mri_sha256':sha(case.mri),'mask_sha256':sha(case.mask),
                         'all_four_confusion_matrices_reproduced':True})
        print(case_id+': reproduced all four checkpoints; saved six-class panel',flush=True)
    (out/'provenance.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],
        'script_sha256':sha(Path(__file__)),'checkpoints':provenance,'cases':receipts},indent=2)+'\n')
    (out/'README.md').write_text('# v2 Validation对照\n\n四个预先指定病例，分别覆盖S022后期退步、K018远处误报、M030患者边界改善和J026残余困难。每图四行：Standard、v1、v2第45及50轮；六列覆盖全部类别。每类固定在GT面积最大的轴向切片，四模型使用相同切片。青色GT、橙色预测、红色FP、蓝色FN。像素索引用于同网格对照，不是毫米距离，也不是完整三维边界证据。图仅用于描述病例；整体结论使用全部38卷、六患者结果。四模型混淆矩阵复现后才绘图，不读取Test。\n',encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__': main()
