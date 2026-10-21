"""Verify local Standard/v3 backups and standalone components; no deletion."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
sys.path.insert(0,str(ROOT/'code'))
from hipmri_rewrite.models.baseline_3d import UNet3D
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3


def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    component=WORKSPACE/'recognition/HipMRI_UNet_s4906923/modules.py'
    spec=importlib.util.spec_from_file_location('standalone_models',component)
    modules=importlib.util.module_from_spec(spec);spec.loader.exec_module(modules)
    audit=json.loads((ROOT/'results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics.json').read_text())
    result={}
    for model,folder,record_dir,original,new,expected_files in [
        ('Standard','g5_standard_20261018',ROOT/'logs/training/baseline_3d',UNet3D,modules.StandardUNet3D,103),
        ('v3','improved_3d_v3_gn1_20261020',ROOT/'results/experiments/improved_3d_v3/training_records',ImprovedUNet3DV3,modules.ImprovedUNet3DV3,105)]:
        backup=ROOT/'archives/remote_training_backups'/folder
        files=sorted(p for p in backup.iterdir() if p.is_file())
        assert len(files)==expected_files
        assert len(list(backup.glob('epoch_*.pt')))==50
        selected=audit['arms'][model]['selected']
        assert sha(backup/'best.pt')==selected['checkpoint_sha256']
        assert sha(backup/'manifest.json')==selected['manifest_sha256']
        assert all(sha(backup/f'epoch_{e:03d}.json')==sha(record_dir/f'epoch_{e:03d}.json') for e in range(1,51))
        checkpoint=torch.load(backup/'best.pt',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==selected['epoch']==50
        assert sha(backup/'best.pt')==sha(backup/'epoch_050.pt')
        old=original().eval();target=new().eval()
        old.load_state_dict(checkpoint['model_state'],strict=True)
        target.load_state_dict(checkpoint['model_state'],strict=True)
        torch.manual_seed(20261021)
        x=torch.randn(1,1,16,16,16)
        with torch.no_grad():
            expected=old(x);actual=target(x)
        assert torch.equal(expected,actual)
        latest=torch.load(backup/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['epoch']==50 and latest['next_batch']==0
        hashes={p.name:sha(p) for p in files}
        manifest=backup.with_suffix('.sha256')
        manifest.write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()),encoding='utf-8',newline='\n')
        result[model]=dict(folder=str(backup.relative_to(WORKSPACE)),files=len(files),epoch_weights=50,
                           best_epoch=50,checkpoint_sha256=sha(backup/'best.pt'),
                           manifest_sha256=sha(backup/'manifest.json'),
                           all_epoch_json_match_previous_verified_records=True,
                           standalone_strict_load_and_cpu_logits_exact=True,
                           real_image_gpu_replay_after_migration='pending',
                           complete_remote_file_manifest_comparison='pending',
                           checksum_list_sha256=sha(manifest))
    remote_path=component.parent/'remote_backup_verification.json'
    if remote_path.exists():
        remote=json.loads(remote_path.read_text(encoding='utf-8'))
        for model, entry in result.items():
            evidence=remote['models'].get(model,{})
            if (evidence.get('all_files_passed') is True
                    and evidence.get('files')==entry['files']
                    and evidence.get('checksum_list_sha256')==entry['checksum_list_sha256']):
                entry['complete_remote_file_manifest_comparison']=True
    status=('Local backups, complete remote file comparisons and CPU component parity verified'
            if all(v['complete_remote_file_manifest_comparison'] is True for v in result.values())
            else 'Local backup and component parity verified; remote full-file verification pending')
    replay_path=component.parent/'evidence/migration_receipt.json'
    replay_digest=None
    if replay_path.exists():
        replay_digest=sha(replay_path)
        assert replay_digest=='14936ed27162acd63209f8ab34dec9fc05192f30746b80111e520c94abaa38fc'
        replay=json.loads(replay_path.read_text(encoding='utf-8'))
        assert all(sha(component.parent/name)==digest for name,digest in replay['sources'].items())
        for model,entry in result.items():
            observed=replay['models'][model]
            assert observed['cases']==38 and observed['all_confusions_and_inputs_match'] is True
            assert observed['checkpoint_sha256']==entry['checkpoint_sha256']
            entry['real_image_gpu_replay_after_migration']=True
        status='Local/remote backups, CPU component parity and standalone GPU Validation replay verified'
    receipt=dict(status=status,
                 modules_sha256=sha(component),torch=str(torch.__version__),models=result,
                 limitation=('Migration equivalence verified on Validation; no fresh Test, retraining or clinical safety validation'
                             if replay_digest else 'CPU random-tensor parity is not real MRI/GPU confusion replay'))
    if replay_digest: receipt['gpu_migration_receipt_sha256']=replay_digest
    out=WORKSPACE/'recognition/HipMRI_UNet_s4906923/preparation_verification.json'
    out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
