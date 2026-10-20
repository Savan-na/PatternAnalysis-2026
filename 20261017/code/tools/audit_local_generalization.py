"""Patient-level descriptive audit; historical Test cannot guide new design."""
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from tools.summarize_v2_results import metrics, patient_matrices


def main():
    out=ROOT/'results/model_evaluation/generalization_diagnostics'
    for name in ('tables','figures','provenance'):
        (out/name).mkdir(parents=True,exist_ok=True)
    hashes={}
    def read(p):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf-8'))
    split=read(ROOT/'code/config/patient_split.json')
    groups=[set(split[k]) for k in ('train','validation','test')]
    assert list(map(len,groups))==[26,6,6]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    specs={'Standard':(ROOT/'logs/training/baseline_3d',50),
           'v1':(ROOT/'logs/training/improved_3d',50),
           'v2':(ROOT/'results/experiments/improved_3d_v2/training_records',45),
           'v3':(ROOT/'results/experiments/improved_3d_v3/training_records',50)}
    records={k:read(p/f'epoch_{e:03d}.json') for k,(p,e) in specs.items()}
    training_audit={}
    training_specs={**specs,'2D':(ROOT/'logs/training/baseline_2d',50)}
    for name,(folder,_) in training_specs.items():
        epochs=[read(folder/f'epoch_{e:03d}.json') for e in range(1,51)]
        for e,r in enumerate(epochs,1):
            assert r['epoch']==e and len(r['train']['batch_history'])==130
            assert np.isfinite([b['loss'] for b in r['train']['batch_history']]).all()
        best=max((r for r in epochs if r['validation']),key=lambda r:r['validation']['selection_score'])
        training_audit[name]={'epochs':50,'updates':6500,'best_epoch':best['epoch'],
            'validation_score':best['validation']['selection_score'],
            'not_isolated_comparison':name=='2D'}
    patients=sorted(groups[1])
    values={k:np.array([metrics(patient_matrices(d)[p])['dice'][3:].mean() for p in patients]) for k,d in records.items()}
    # Enumerate the empirical bootstrap exactly for n=6, not 38 dependent scans.
    indices=np.indices((6,)*6).reshape(6,-1).T
    delta=values['v3']-values['Standard']
    interval=np.quantile(delta[indices].mean(1),[.025,.975])
    loo={p:float(np.delete(delta,i).mean()) for i,p in enumerate(patients)}
    rows=[]
    for k,d in records.items():
        m=np.array(d['validation']['pooled_confusion'])
        for label in (1,3,4,5):
            rows.append(dict(model=k,target=['Background','Body','Bones','Bladder','Rectum','Prostate'][label],
                tp=int(m[label,label]),fp=int(m[:,label].sum()-m[label,label]),
                fn=int(m[label].sum()-m[label,label]),
                fp_from_background=int(m[0,label]) if label else 0,
                fp_from_body=int(m[1,label]) if label!=1 else 0))
    with (out/'tables/validation_error_sources.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    test=read(ROOT/'results/tables/comparisons/g6_test_3d.json')
    tvalues={k:{p:float(np.mean(list(d.values()))) for p,d in a['per_patient_small_organ_dice'].items()} for k,a in test['arms'].items()}
    differences={p:tvalues['improved_3d'][p]-tvalues['standard_3d'][p] for p in sorted(groups[2])}
    testrows=[dict(patient=p,standard=tvalues['standard_3d'][p],v1=tvalues['improved_3d'][p],v1_minus_standard=d) for p,d in differences.items()]
    with (out/'tables/historical_test_patient_influence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(testrows[0]));w.writeheader();w.writerows(testrows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].bar(patients,delta);axes[0].axhline(0,color='black',lw=.8)
    axes[0].set(title='Validation: v3 minus Standard',ylabel='Patient small-organ Dice difference')
    axes[1].bar(list(differences),list(differences.values()));axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(title='Historical inspected Test: v1 minus Standard',ylabel='Descriptive difference; NOT a v4 selection input')
    for ext in ('png','svg'):fig.savefig(out/'figures'/f'patient_influence.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    result={'patient_id_split_disjoint':True,
        'five_completed_runs':training_audit,
        'cross_split_image_duplicate_check':'not established locally; requires remote raw-input hashes',
        'validation_v3_minus_standard':{'mean':float(delta.mean()),'wins':int((delta>0).sum()),
             'empirical_patient_bootstrap_percentile_95':interval.tolist(),
             'bootstrap_caveat':'Six fixed patients; repeated adaptive Validation use and model selection are NOT accounted for. Not confirmatory generalization evidence.',
             'leave_one_patient_out_mean_delta':loo},
        'historical_test_v1_minus_standard':{'mean':float(np.mean(list(differences.values()))),
             'without_M015':float(np.mean([d for p,d in differences.items() if p!='M015'])),
             'purpose':'descriptive influence audit only; no exclusions or new model choices'},
        'v3_test_status':'not yet evaluated; cannot claim generalization improvement'}
    (out/'tables/audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'provenance/local_source_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
