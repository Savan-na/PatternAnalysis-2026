"""Bundle only standalone code/config/count evidence for read-only HPC replay."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib,json
WORKSPACE=Path(__file__).resolve().parents[3]
ROOT=WORKSPACE/"recognition/HipMRI_UNet_s4906923"
OUT=WORKSPACE/"20261017/code/transfer/submission_replay_20261022.zip"
ARCHIVE=WORKSPACE/"20261017/archives/submission_verification"
FILES={name:ROOT/name for name in ("modules.py","dataset.py","utils.py","train.py",
       "predict.py","tests/test_submission.py","config/patient_split.json",
       "config/training.json","evidence/experiments.json")}
FILES["verify_replay.py"]=ARCHIVE/"verify_replay.py"

def main():
    checks={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in FILES.items()}
    with ZipFile(OUT,"w",ZIP_DEFLATED) as z:
        for name,path in FILES.items():z.write(path,name)
        z.writestr("transfer_checksums.json",json.dumps(checks,indent=2)+"\n")
    print(OUT)
    print("SHA256:",hashlib.sha256(OUT.read_bytes()).hexdigest())
    print("Code/config/count records only; no MRI, weights or report images")
    print("In the isolated extraction directory, set HIPMRI_SUBMISSION_ROOT to that directory before replay.")

if __name__=="__main__":main()
