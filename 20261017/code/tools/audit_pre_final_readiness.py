"""Read-only scientific/source audit; writes receipts only, no training or restructuring."""
from collections import defaultdict, Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from tools.build_model_comparison_gallery import collect_evidence, DIRS
from hipmri_rewrite.metrics import overlap
from hipmri_rewrite.training import patient_small_organ_score

OUT=ROOT/'results/reports/model_performance_comparison'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data=collect_evidence()
    rows=[]; validated_scores=0; patient_details={}
    raw=json.loads((ROOT/'results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics.json').read_text())
    v5=json.loads((ROOT/'results/experiments/improved_3d_v5/comparisons/test_assessment_20261021/metrics.json').read_text())
    reference_truth={split:{r['case']:np.asarray(r['confusion']).sum(1) for r in raw['arms']['Standard']['splits'][split]} for split in ('validation','test')}
    for model,directory in DIRS.items():
        records=[json.loads((directory/f'epoch_{i:03d}.json').read_text()) for i in range(1,51)]
        best_so_far=-1.; best_epoch=None
        for i,record in enumerate(records,1):
            assert record['epoch']==i and len(record['train']['batch_history'])==130
            assert np.isfinite(record['train']['mean_loss'])
            v=record.get('validation')
            if v:
                matrices=defaultdict(lambda:np.zeros((6,6),dtype=np.int64))
                assert len(v['cases'])==38
                assert {c['case'] for c in v['cases']}==set(reference_truth['validation'])
                for case in v['cases']:
                    matrix=np.asarray(case['confusion'])
                    assert matrix.shape==(6,6) and (matrix>=0).all()
                    assert np.array_equal(matrix.sum(1),reference_truth['validation'][case['case']])
                    matrices[case['patient']]+=matrix
                score,_=patient_small_organ_score(matrices)
                assert len(matrices)==6 and abs(score-v['selection_score'])<1e-12
                if score>best_so_far:best_so_far,best_epoch=score,i
                validated_scores+=1
            assert record['best_epoch']==best_epoch
            assert abs(record['best_score']-best_so_far)<1e-12
        epoch=best_epoch
        best=records[epoch-1]
        test=(raw['arms'][model]['splits']['test'] if model in raw['arms'] else v5['arms'][model[-1]]['cases'])
        assert len(test)==30 and {r['case'] for r in test}==set(reference_truth['test'])
        row=dict(model=model,epochs=50,updates=6500,best_epoch=epoch,
                 original_validation_selection=best_so_far)
        for split,cases in [('validation',best['validation']['cases']),('test',test)]:
            matrices=defaultdict(lambda:np.zeros((6,6),dtype=np.int64))
            for case in cases:
                matrix=np.asarray(case['confusion'])
                assert np.array_equal(matrix.sum(1),reference_truth[split][case['case']])
                matrices[case['patient']]+=matrix
            score,_=patient_small_organ_score(matrices)
            scores={p:overlap(m) for p,m in matrices.items()}
            row[split+'_small_organ_dice']=score
            row[split+'_foreground_dice']=float(np.mean([r['dice'][1:] for r in scores.values()]))
            row[split+'_prostate_dice']=float(np.mean([r['dice'][5] for r in scores.values()]))
            row[split+'_patients_prostate_below_07']=sum(r['dice'][5]<.7 for r in scores.values())
            patient_details[(model,split)]=scores
        rows.append(row)
    # Check all recorded case hashes for cross-split exact MRI duplication.
    train_path=ROOT/'logs/audits/train_full_20261018_job671208.json'
    train=json.loads(train_path.read_text())['cases']
    hashes={'train':{r['mri_sha256'] for r in train}}
    for split in ('validation','test'):
        hashes[split]={r['mri_sha256'] for r in raw['arms']['Standard']['splits'][split]}
    assert [len(hashes[s]) for s in ('train','validation','test')]==[143,38,30]
    assert not any(hashes[a]&hashes[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    split_config=json.loads((ROOT/'code/config/patient_split.json').read_text())
    assert not any(set(split_config[a])&set(split_config[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    manifest_path=DIRS['v3']/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    source_map={'v3_source_sha256':'hipmri_rewrite/improved_v3.py','model_source_sha256':'hipmri_rewrite/models/improved_3d.py',
                'model_common_sha256':'hipmri_rewrite/models/common.py','data_sha256':'hipmri_rewrite/data.py',
                'inference_sha256':'hipmri_rewrite/inference.py','loss_sha256':'hipmri_rewrite/losses.py',
                'sampling_sha256':'hipmri_rewrite/sampling.py','metrics_sha256':'hipmri_rewrite/metrics.py',
                'selection_sha256':'hipmri_rewrite/training.py','splits_sha256':'hipmri_rewrite/splits.py',
                'trainer_sha256':'tools/train_v3_cohort.py'}
    assert all(manifest[k]==sha(ROOT/'code'/p) for k,p in source_map.items())
    paired={}
    for split in ('validation','test'):
        a,b=patient_details[('v3',split)],patient_details[('Standard',split)]
        diffs={p:float(np.mean(a[p]['dice'][3:])-np.mean(b[p]['dice'][3:])) for p in a}
        paired[split]=dict(per_patient_small_organ_difference=diffs,patients_v3_above_standard=sum(v>0 for v in diffs.values()))
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    forbidden=[p for p in tracked if p.endswith(('.pt','.pth','.nii','.nii.gz','.zip'))]
    assert not forbidden
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    local_pt=Counter(str(p.parent.relative_to(ROOT)) for p in files if p.suffix=='.pt')
    spec=ROOT.parent.parent/'COMP3710_Report_2026_draft_v2.2.pdf'
    receipt=dict(scope='Local records/source/PDF evidence; no remote live inventory, no new raw MRI decode or GPU checkpoint replay',
                 specification_sha256=sha(spec),model_count=len(rows),epoch_records_checked=500,
                 formal_training_updates=65000,validation_selection_scores_recomputed=validated_scores,
                 selected_checkpoint_case_entries=680,distinct_patient_counts=[26,6,6],
                 exact_mri_hash_cross_split_duplicates=0,paired_v3_standard=paired,
                 current_v3_source_hashes_match_recorded_manifest=True,
                 current_v3_manifest_sha256=sha(manifest_path),tracked_forbidden_payloads=forbidden,
                 local_file_count=len(files),local_suffix_counts=dict(Counter(p.suffix for p in files)),
                 local_checkpoint_counts_by_directory=dict(local_pt),
                 final_recognition_directory_exists=(ROOT.parent/'recognition').is_dir(),
                 cpu_checks_this_review={'test_improved_v3.py':2,'test_data.py':6,'test_metrics.py':1},
                 limitations=['Recorded hashes do not prove anatomical ground truth or absence of near-duplicates',
                              'One seed per recipe; v3 and C repeat the same seed, not an independent-seed replication',
                              'Exact final submission rules and check-off/PR/Turnitin status require confirmation'])
    (OUT/'pre_final_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (OUT/'tables/pre_final_model_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=rows,paired=paired,manifest=receipt['current_v3_manifest_sha256'],
                         local_checkpoint_counts=local_pt),indent=2))


if __name__=='__main__':main()
