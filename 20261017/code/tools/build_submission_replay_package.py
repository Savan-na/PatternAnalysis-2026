"""Bundle only standalone code/config/count evidence for read-only HPC replay."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib,json
WORKSPACE=Path(__file__).resolve().parents[3]
ROOT=WORKSPACE/"recognition/HipMRI_UNet_s4906923"
OUT=WORKSPACE/"20261017/code/transfer/submission_replay_20261021.zip"
FILES=("modules.py","dataset.py","utils.py","train.py","predict.py","verify_replay.py",
       "verify_evidence.py","tests/test_submission.py","config/patient_split.json",
       "config/training.json","evidence/experiments.json")

def main():
    checks={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}
    with ZipFile(OUT,"w",ZIP_DEFLATED) as z:
        for name in FILES:z.write(ROOT/name,name)
        z.writestr("transfer_checksums.json",json.dumps(checks,indent=2)+"\n")
    print(OUT)
    print("SHA256:",hashlib.sha256(OUT.read_bytes()).hexdigest())
    print("Code/config/count records only; no MRI, weights or report images")

if __name__=="__main__":main()
