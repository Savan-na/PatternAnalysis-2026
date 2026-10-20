"""Seven locked models: descriptive Validation/Test audit. No training or model selection."""
from pathlib import Path
import argparse, gc, hashlib, json, os, sys
from time import perf_counter
import numpy as np
import torch
from scipy.ndimage import label, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hipmri_rewrite.models import UNet2D, UNet3D, ImprovedUNet3D
from hipmri_rewrite.improved_v2 import ImprovedUNet3DV2
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3
from hipmri_rewrite.data import load_case, downsample, zscore_nonzero
from hipmri_rewrite.inference import predict_volume
from hipmri_rewrite.boundaries import prostate_axial_regions, slice_dice, symmetric_surface_distances
from hipmri_rewrite.metrics import confusion
from hipmri_rewrite.splits import discover, DEFAULT_SPLIT

NAMES=['Background','Body','Bones','Bladder','Rectum','Prostate']
SPECS=[('2D','g5_2d_20261019','baseline_2d',UNet2D,'2d'),
       ('Standard','g5_standard_20261018','standard_3d',UNet3D,'3d'),
       ('v1','g5_improved_20261018','improved_3d',ImprovedUNet3D,'3d'),
       ('v2','improved_3d_v2_ds_20261020','improved_3d_v2',ImprovedUNet3DV2,'3d'),
       ('v3','improved_3d_v3_gn1_20261020','improved_3d_v3',ImprovedUNet3DV3,'3d'),
       ('A0','improved_3d_v4_a0_20261020','improved_3d_v4_a0',ImprovedUNet3DV3,'3d'),
       ('A1','improved_3d_v4_a1_20261020','improved_3d_v4_a1',ImprovedUNet3DV3,'3d')]

PANELS={'validation':{'B040_Week1':list(range(6)),'J026_Week4':[4],'K008_Week0':[3,5],
                     'K018_Week4':[0,1],'M030_Week4':[1,2],'S022_Week2':[3,5]},
        'test':{'M015_Week7':[3],'M020_Week1':[5],'W029_Week6':[1,2,5]}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def geometry(mask,spacing):
    components,n=label(mask)
    counts=np.bincount(components.ravel())[1:]
    occupied=np.flatnonzero(mask.any(axis=(0,1)))
    return {'voxels':int(mask.sum()),'volume_mm3':float(mask.sum()*np.prod(spacing)),
            'components_6':int(n),'largest_component_fraction':float(counts.max()/counts.sum()) if n else None,
            'occupied_z_slices':int(len(occupied)),
            'internal_empty_z_slices':int(occupied[-1]-occupied[0]+1-len(occupied)) if len(occupied) else 0}


def diagnostics(volume,prediction):
    truth=volume.mask; spacing=volume.spacing_mm
    far=((truth==0)&(prediction==1)&(distance_transform_edt(truth!=1,sampling=spacing)>=20))
    result={'far_background_to_body_voxels':int(far.sum()),'background_voxels':int((truth==0).sum()),
            'surfaces':{NAMES[k]:symmetric_surface_distances(truth,prediction,spacing,label=k) for k in range(1,6)},
            'geometry':{NAMES[k]:{'truth':geometry(truth==k,spacing),'prediction':geometry(prediction==k,spacing)} for k in range(1,6)}}
    try:
        regions=prostate_axial_regions(truth,volume.affine)
        per_slice=slice_dice(truth,prediction)
        result['prostate_regions']={n:float(np.mean([per_slice[z] for z in zz])) for n,zz in regions.items()}
        result['prostate_region_status']='physical_superior_axis_partition'
    except ValueError as error:
        result['prostate_regions']=None; result['prostate_region_status']=str(error)
    for k in (3,4,5):
        gt=(truth==k); pr=(prediction==k)
        gt_area=gt.sum(axis=(0,1)); pr_area=pr.sum(axis=(0,1))
        both=(gt_area[:-1]>0)&(gt_area[1:]>0)
        def variation(area):
            den=np.maximum((area[:-1]+area[1:])/2,1)
            return float(np.mean((np.abs(np.diff(area.astype(float)))/den)[both])) if both.any() else None
        result['geometry'][NAMES[k]]['area_transition']={'truth':variation(gt_area),'prediction':variation(pr_area),
            'definition':'Mean normalized adjacent area change on GT-positive adjacent pairs; descriptive, not anatomy validity'}
    result['prostate_in_true_oar']={n:int(((prediction==5)&(truth==k)).sum())
                                    for n,k in [('Bladder',3),('Rectum',4)]}
    gt_slices=np.flatnonzero((truth==5).any(axis=(0,1)))
    pred_slices=np.flatnonzero((prediction==5).any(axis=(0,1)))
    result['prostate_extent']={
        'GT_positive_slices_without_any_predicted_prostate':int(len(set(gt_slices)-set(pred_slices))),
        'predicted_slices_outside_GT_extent':int(((pred_slices<gt_slices[0])|(pred_slices>gt_slices[-1])).sum()) if len(gt_slices) else None,
        'note':'Slice presence is not correct localization; read alongside base/apex Dice and voxel errors'}
    return result


def load_model(spec,lock):
    arm,folder,identity,ctor,mode=spec; epoch=lock['selected'][arm]['epoch']; run=ROOT/'logs/training'/folder
    manifest=json.loads((run/'manifest.json').read_text()); record_path=run/f'epoch_{epoch:03d}.json'
    record=json.loads(record_path.read_text()); expected=lock['selected'][arm]
    assert sha(run/'manifest.json')==expected['manifest_sha256'] and sha(record_path)==expected['record_sha256']
    assert manifest['split_sha256']==sha(DEFAULT_SPLIT)
    candidates=[json.loads((run/f'epoch_{e:03d}.json').read_text()) for e in range(1,51)]
    selected=max((r for r in candidates if r['validation'] is not None),
                 key=lambda r:(r['validation']['selection_score'],-r['epoch']))
    assert selected['epoch']==epoch, 'Locked epoch must be best recorded Validation checkpoint, not necessarily final'

    source_map={'data_sha256':'hipmri_rewrite/data.py','inference_sha256':'hipmri_rewrite/inference.py',
        'model_source_sha256':('hipmri_rewrite/models/baseline_2d.py' if arm=='2D' else 'hipmri_rewrite/models/baseline_3d.py' if arm=='Standard' else 'hipmri_rewrite/models/improved_3d.py'),
        'model_common_sha256':'hipmri_rewrite/models/common.py','boundaries_sha256':'hipmri_rewrite/boundaries.py'}
    if arm in ('A0','A1'):
        source_map.update({'v3_source_sha256':'hipmri_rewrite/improved_v3.py', 'augmentation_sha256':'hipmri_rewrite/augmentation_v4.py', 'trainer_sha256':'tools/train_v4_cohort.py'})
    if arm in ('v2','v3'):source_map[arm+'_source_sha256']='hipmri_rewrite/improved_'+arm+'.py'
    for key,relative in source_map.items():
        if key in manifest:assert sha(ROOT/'code'/relative)==manifest[key],relative
    checkpoint_path=run/f'epoch_{epoch:03d}.pt'
    assert sha(checkpoint_path)==sha(run/'best.pt')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['model']==identity and checkpoint['epoch']==epoch==record['best_epoch']
    assert checkpoint['config_sha256']==manifest['config_sha256']
    assert abs(checkpoint['best_score']-record['validation']['selection_score'])<1e-12
    model=ctor();model.load_state_dict(checkpoint['model_state'],strict=True)
    del checkpoint
    return model.cuda().eval(),record,{'checkpoint_sha256':sha(checkpoint_path),**expected}


def panel(volume,predictions,target,case_id,path,split,region_panel=False):
    gt=volume.mask==target
    peak=int(np.argmax(gt.sum(axis=(0,1))))
    error=(predictions['A1']==target)!=gt
    worst=int(np.argmax(error.sum(axis=(0,1))))
    zlist=[peak,worst]
    row_labels=['GT peak area','A1 largest FP+FN area']
    if region_panel:
        regions=prostate_axial_regions(volume.mask,volume.affine)
        zlist=[int(regions[n][len(regions[n])//2]) for n in ('apex','central','base')]
        row_labels=['apex: inferior 20%','central region','base: superior 20%']
    nz=volume.image[volume.image!=0];low,high=np.percentile(nz,[1,99])
    fig,axes=plt.subplots(len(zlist),len(predictions)+1,figsize=(30,4*len(zlist)),layout='constrained')
    for row,z in enumerate(zlist):
        truth=gt[:,:,z].T
        for col,(name,pred) in enumerate([('MRI + GT',None),*predictions.items()]):
            ax=axes[row,col];ax.imshow(volume.image[:,:,z].T,cmap='gray',origin='lower',vmin=low,vmax=high)
            if truth.any() and not truth.all():ax.contour(truth.astype(float),[.5],colors=['cyan'],linewidths=1)
            if pred is not None:
                found=(pred[:,:,z]==target).T
                if found.any() and not found.all():ax.contour(found.astype(float),[.5],colors=['orange'],linewidths=1)
                rgba=np.zeros((*truth.shape,4));rgba[found&~truth]=[1,0,0,.4];rgba[truth&~found]=[0,.3,1,.4]
                ax.imshow(rgba,origin='lower')
            ax.axis('off');ax.set_title(f'{name} | z={z}\n'+row_labels[row])
    fig.suptitle(f'{split.upper()} {case_id} | {NAMES[target]} | GT cyan / prediction orange / FP red / FN blue\nPurposeful diagnostic slices, not whole-volume performance')
    fig.savefig(path,dpi=150);plt.close(fig)


def body_three_planes(volume, predictions, case_id, path):
    """Shared A1 worst-error grid planes, displayed with physical pixel aspect."""
    truth = volume.mask == 1
    error = (predictions['A1'] == 1) != truth
    fig, axes = plt.subplots(3, len(predictions)+1, figsize=(30, 12), layout='constrained')
    nz = volume.image[volume.image != 0]
    low, high = np.percentile(nz, [1, 99])
    for row, axis in enumerate((2, 1, 0)):
        reduction = tuple(i for i in range(3) if i != axis)
        index = int(np.argmax(error.sum(axis=reduction)))
        remaining = [i for i in range(3) if i != axis]
        aspect = volume.spacing_mm[remaining[1]] / volume.spacing_mm[remaining[0]]
        image = np.take(volume.image, index, axis=axis).T
        gt = np.take(truth, index, axis=axis).T
        for col, (name, pred) in enumerate([('MRI + GT', None), *predictions.items()]):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray', origin='lower', vmin=low, vmax=high, aspect=aspect)
            if gt.any() and not gt.all():
                ax.contour(gt.astype(float), [.5], colors=['cyan'], linewidths=.8)
            if pred is not None:
                found = np.take(pred == 1, index, axis=axis).T
                if found.any() and not found.all():
                    ax.contour(found.astype(float), [.5], colors=['orange'], linewidths=.8)
                rgba = np.zeros((*gt.shape, 4))
                rgba[found & ~gt] = [1, 0, 0, .4]
                rgba[gt & ~found] = [0, .3, 1, .4]
                ax.imshow(rgba, origin='lower', aspect=aspect)
            ax.set_title(f'{name} | grid axis {axis}, index {index}')
            ax.axis('off')
    fig.suptitle(f'Validation {case_id}: Body, A1 maximum-error plane per grid axis\nPhysical pixel aspect; GT cyan, prediction orange, FP red, FN blue; selected planes are not aggregate performance')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Use allocated GPU')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock_path=ROOT/'code/config/all_model_assessment_audit.json'; lock=json.loads(lock_path.read_text())
    assert sha(ROOT/lock['v4_pretest_plan_path'])==lock['v4_pretest_plan_sha256']
    out=ROOT/'results/experiments/improved_3d_v4/comprehensive_audit_20261021'
    out.mkdir(parents=True,exist_ok=False)
    all_cases=discover(args.dataset_root)
    assert len(all_cases['validation'])==38 and len(all_cases['test'])==30
    split=json.loads(DEFAULT_SPLIT.read_text())
    assert not (set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test']))
    g1=ROOT/'logs/audits/train_full_20261018_job671208.json'
    assert sha(g1)=='6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1'
    audit=json.loads(g1.read_text()); fingerprints={}
    for row in audit['cases']:
        fingerprints.setdefault(row['mri_sha256'],[]).append(('train',row['case']))
    cases={};hashes={};volumes={};cross_duplicates=[]
    for split_name in ('validation','test'):
        for case in all_cases[split_name]:
            key=case.identifier;cases[key]=case
            hashes[key]={'mri_sha256':sha(case.mri),'mask_sha256':sha(case.mask)}
            for previous_split,previous_case in fingerprints.get(hashes[key]['mri_sha256'],[]):
                if previous_split!=split_name:cross_duplicates.append([previous_split,previous_case,split_name,key])
            fingerprints.setdefault(hashes[key]['mri_sha256'],[]).append((split_name,key))
            volumes[key]=zscore_nonzero(downsample(load_case(case)))
    write(out/'input_provenance.json',{'hashes':hashes,'exact_cross_split_mri_duplicates':cross_duplicates,
        'note':'Exact file SHA only; different encodings/near duplicates are not excluded'})
    if cross_duplicates:raise ValueError('Cross-split exact MRI duplicate; stop evaluation and investigate')
    # Existing Test provides input/GT provenance, never checkpoint/branch selection.
    historical_path=ROOT/'results/tables/comparisons/g6_test_3d.json'
    assert sha(historical_path)==lock['historical_test_sha256']
    historical=json.loads(historical_path.read_text())
    for row in historical['arms']['standard_3d']['cases']:
        assert all(hashes[row['case']][k]==row[k] for k in ('mri_sha256','mask_sha256'))
    result={'kind':'fixed_checkpoint_descriptive_generalization_audit',
        'test_limit':'Previously inspected cohort; no tuning or fresh-blind claim; All seven checkpoints and evaluation rules fixed before this evaluation; no future tuning from Test',
        'job':os.environ['SLURM_JOB_ID'],'script_sha256':sha(Path(__file__)),'lock_sha256':sha(lock_path),
        'v4_pretest_plan_sha256':lock['v4_pretest_plan_sha256'],'gpu':torch.cuda.get_device_name(0),
        'torch':str(torch.__version__),'numpy':np.__version__,'arms':{}}
    cache={key:{} for sp in PANELS.values() for key in sp}
    for spec in SPECS:
        arm=spec[0];mode=spec[4];gc.collect();torch.cuda.empty_cache()
        model,record,receipt=load_model(spec,lock)
        reference=volumes['K008_Week0']
        for _ in range(3):predict_volume(model,reference,mode=mode,device='cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(5):
            torch.cuda.synchronize();t=perf_counter();predict_volume(model,reference,mode=mode,device='cuda');torch.cuda.synchronize();times.append(perf_counter()-t)
        result['arms'][arm]={'selected':receipt,'benchmark':{'case':'K008_Week0','warmup':3,'repeats':5,
            'seconds':times,'median_seconds':float(np.median(times)),
            'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'parameters':sum(p.numel() for p in model.parameters()),
            'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
            'definition':'One resident model; prepared whole-volume predict_volume including transfer/argmax, excludes preprocessing/metrics/IO'},'splits':{}}
        for sp in ('validation','test'):
            rows=[]
            for i,case in enumerate(all_cases[sp],1):
                volume=volumes[case.identifier]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started=perf_counter()
                prediction=predict_volume(model,volume,mode=mode,device='cuda')
                torch.cuda.synchronize()
                per_case_seconds=perf_counter()-started
                per_case_peak=int(torch.cuda.max_memory_allocated())
                matrix=confusion(volume.mask,prediction).tolist()
                if sp=='validation':
                    saved=next(r for r in record['validation']['cases'] if r['case']==case.identifier)
                    assert matrix==saved['confusion'],f'{arm}/{case.identifier}: Validation replay mismatch'
                elif arm in ('Standard','v1'):
                    identity='standard_3d' if arm=='Standard' else 'improved_3d'
                    saved=next(r for r in historical['arms'][identity]['cases'] if r['case']==case.identifier)
                    assert matrix==saved['confusion'],f'{arm}/{case.identifier}: historical Test replay mismatch'
                rows.append({'case':case.identifier,'patient':case.patient,'confusion':matrix,**hashes[case.identifier],
                    'inference_seconds_single_pass':per_case_seconds,'inference_peak_allocated_bytes':per_case_peak,
                    'spacing_mm':list(volume.spacing_mm),'shape_xyz':list(volume.mask.shape),**diagnostics(volume,prediction)})
                if case.identifier in cache:cache[case.identifier][arm]=prediction
                print(f'{arm} {sp} {i}/{len(all_cases[sp])}: {case.identifier}',flush=True)
            result['arms'][arm]['splits'][sp]=rows
            write(out/'metrics_partial.json',result)
        del model;gc.collect();torch.cuda.empty_cache()
    for sp,pairs in PANELS.items():
        dest=out/'images'/sp;dest.mkdir(parents=True)
        for key,targets in pairs.items():
            if sp == 'validation':
                body_three_planes(volumes[key], cache[key], key, dest/f'{key}_Body_three_planes.png')
            for target in targets:
                panel(volumes[key],cache[key],target,key,dest/f'{key}_{NAMES[target]}.png',sp)
                if target==5:
                    try:
                        panel(volumes[key],cache[key],target,key,dest/f'{key}_Prostate_base_apex.png',sp,region_panel=True)
                    except ValueError as error:
                        write(dest/f'{key}_region_unavailable.json',{'case':key,'reason':str(error)})
        (dest/'README.md').write_text('# 固定best权重诊断图\n\n第一行按GT最大面积选切片，第二行按固定A1的FP+FN最大面积选切片，七模型共用位置。青色GT、橙色预测、红色FP、蓝色FN。这样同时检查器官中心和中心之外的错误；是目的性失败分析，不能代表整体指标。Test图片只用于描述，不用于选模或调参。\n',encoding='utf-8')
    result['complete']=True;write(out/'metrics.json',result)
    (out/'README.md').write_text('# 最佳checkpoint泛化审计\n\n七个模型各自按已记录Validation选择best；38卷Validation、30卷已被查看的Test。所有Validation矩阵复现，Standard/v1 Test矩阵复现。记录六类矩阵、五类表面距离、基底/尖端、连通分量及面积过渡、远处误报和隔离推理成本。Test不是新盲测，不能用于后续调参。one-empty表面状态需单列，不能静默排除。指标不代表临床安全。\n',encoding='utf-8')
    write(out/'checksums.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print('Completed descriptive audit: '+str(out),flush=True)


if __name__=='__main__':main()
