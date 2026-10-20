from pathlib import Path
import ast,hashlib,json,zipfile
r=Path(__file__).resolve().parents[2]; code=r/'code'
package=code/'transfer/v4_paired_spatial_20261020.zip'
if package.exists():
    raise FileExistsError('Frozen v4 package already exists; do not rewrite training transfer identity')
files=['hipmri_rewrite/augmentation_v4.py','tools/train_v4_cohort.py','tools/audit_v4_augmentation.py','tests/test_v4_training.py','config/v4_a0_frozen.json','config/v4_a1_frozen.json','tools/submit_v4_paired.sh','docs/V4_Paired_Training_ZH.md']
for n in files:
    if n.endswith('.py'): ast.parse((code/n).read_text(encoding='utf-8-sig'))
a=json.loads((code/'config/v4_a0_frozen.json').read_text()); b=json.loads((code/'config/v4_a1_frozen.json').read_text())
diff=[k for k in a if a[k]!=b[k]]
assert diff==['models','sampling','augmentation'],diff
checks={n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in files}
(code/'config/v4_transfer_checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
package.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
    for n in files+['config/v4_transfer_checksums.json']: z.write(code/n,'code/'+n)
with zipfile.ZipFile(package) as z: assert z.testzip() is None and len(z.namelist())==9
print('AST and paired config checks passed; differing keys:',diff)
print('ZIP SHA256:',hashlib.sha256(package.read_bytes()).hexdigest()); print('Bytes:',package.stat().st_size)
