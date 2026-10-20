"""Train-only first-epoch paired exposure audit and real-image previews; no training."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'code'))
from hipmri_rewrite.data import load_case, downsample, zscore_nonzero
from hipmri_rewrite.splits import discover
from hipmri_rewrite.sampling import epoch_cases, crop_origin, PATCH_XYZ
from hipmri_rewrite.augmentation_v4 import paired_patch


def main():
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,required=True); args=p.parse_args()
    out=PROJECT/'results/experiments/improved_3d_v4/augmentation_audit_20261020'
    if out.exists(): raise ValueError('Audit output exists; inspect before rerun')
    out.mkdir(parents=True)
    seed=371017; rows=[]; before=np.zeros(6,dtype=np.int64); after=before.copy(); previews=0
    for case in epoch_cases(discover(args.dataset_root)['train'],epoch=0,seed=seed):
        v=zscore_nonzero(downsample(load_case(case)))
        for draw in range(10):
            origin=crop_origin(v.mask,PATCH_XYZ,'uniform',seed=seed,epoch=0,case_id=case.identifier,draw=draw)
            x,y,_=paired_patch(v,origin,seed=seed,epoch=0,draw=draw,enabled=False)
            a,b,r=paired_patch(v,origin,seed=seed,epoch=0,draw=draw,enabled=True)
            before+=r['before_counts']; after+=r['after_counts']; rows.append(r)
            if r['applied'] and previews<6:
                z=int(np.argmax((y>2).sum(axis=(1,2)))) if np.any(y>2) else y.shape[0]//2
                fig,ax=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
                for panel,im,ma,title in zip(ax,[x,a],[y,b],['A0 original','A1 paired affine']):
                    panel.imshow(im[0,z],cmap='gray'); panel.imshow(np.ma.masked_equal(ma[z],0),cmap='tab10',vmin=0,vmax=9,alpha=.4)
                    panel.set_title(title); panel.axis('off')
                fig.suptitle(f'{case.identifier} draw {draw}; slice D={z}; labels 1 Body,2 Bones,3 Bladder,4 Rectum,5 Prostate')
                fig.savefig(out/f'preview_{previews+1:02d}_{case.identifier}.png',dpi=150); plt.close(fig); previews+=1
        print(f'{case.identifier}: ten paired crops',flush=True)
    shares0=before/before.sum(); shares1=after/after.sum()
    ratio=np.divide(shares1,shares0,out=np.ones(6),where=shares0>0)
    gates={'all_26_patients':len(rows)==260,'six_real_previews':previews==6,
           'small_organ_exposure_ratio_ge_085':bool(np.all(ratio[3:]>=.85)),
           'mean_outside_fraction_le_010':float(np.mean([r['outside_fraction'] for r in rows]))<=.10}
    d={'scope':'Train epoch0; mechanical/exposure check, not efficacy proof','gates':gates,'eligible':all(gates.values()),
       'voxel_share_ratio_A1_A0':ratio.tolist(),'patches':rows}
    (out/'audit.json').write_text(json.dumps(d,indent=2)+'\n')
    (out/'README.md').write_text('# V4 Train配对增强预览\n\n同一Train来源、中心和切片的A0/A1叠加图。机械门槛只检查第一轮曝光及填充；不能证明增强有效。人工核查后才提交GPU训练。\n',encoding='utf-8')
    checks={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
    (out/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('gates',gates,'eligible',d['eligible']); print(out)
    if not d['eligible']: raise ValueError('Exposure gates failed; do not train')


if __name__=='__main__': main()
