"""Submit a bounded Validation-only v3 C/L/S screen, or resume one branch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT/'code'
PYTHON = Path.home()/'miniconda3/envs/torch/bin/python'
DATA = '/home/groups/comp3710/HipMRI_Study_open'
sys.path.insert(0, str(CODE))


def verify():
    checks = json.loads((CODE/'config/v5_transfer_checksums.json').read_text())
    for relative, expected in checks.items():
        path = ROOT/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source/reference checksum mismatch: {relative}')


def run_dir(arm):
    return ROOT/f'logs/training/improved_3d_v5_{arm.lower()}_20261021'


def wrap(arguments):
    return ('export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
            'NUMEXPR_NUM_THREADS=1; exec ' + shlex.join([str(PYTHON), '-B', '-u', *arguments]))


def submit(options, command):
    result = subprocess.check_output(['sbatch', '--parsable', '--nodes=1', '--ntasks=1',
                                      '--cpus-per-task=2', f'--chdir={CODE}', *options,
                                      '--wrap='+wrap(command)], text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError(f'Unexpected sbatch response: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['new', 'resume-C', 'resume-L', 'resume-S', 'task'])
    parser.add_argument('--arm', choices=list('CLS'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    verify()
    from hipmri_rewrite.v5_branches import expected_config, validate_config
    for arm in 'CLS':
        validate_config(json.loads((CODE/f'config/v5_{arm.lower()}_frozen.json').read_text()))
    if args.mode == 'task':
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Task requires an allocated Slurm job')
        arm = args.arm or 'CLS'[int(os.environ['SLURM_ARRAY_TASK_ID'])]
        command = [str(PYTHON), '-B', '-u', str(CODE/'tools/train_v5_branch.py'),
                   '--dataset-root', DATA, '--config', str(CODE/f'config/v5_{arm.lower()}_frozen.json'),
                   '--model', f'improved_3d_v5_{arm.lower()}', '--run-dir', str(run_dir(arm))]
        if args.resume:
            command.append('--resume')
        os.execv(str(PYTHON), command)
    if args.arm or args.resume:
        parser.error('--arm/--resume are internal task flags')
    (ROOT/'logs/profiles').mkdir(parents=True, exist_ok=True)
    with (ROOT/'logs/profiles/v5_submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%j'], text=True)
        if any(name.startswith('v5-') for name in active.splitlines()):
            raise ValueError('A v5 job is queued/running; inspect before another submission')
        arms = 'CLS' if args.mode == 'new' else args.mode[-1]
        for arm in arms:
            if args.mode == 'new' and run_dir(arm).exists():
                raise ValueError(f'Run already exists: {run_dir(arm)}')
            if args.mode != 'new' and not (run_dir(arm)/'manifest.json').is_file():
                raise ValueError('Resume manifest missing')
        required = 6*1024**3 if args.mode == 'new' else 2*1024**3
        free = shutil.disk_usage(ROOT).free
        if free < required:
            raise ValueError(f'Free space {free/1024**3:.2f} GiB < required {required/1024**3:.1f} GiB')
        check = submit(['--partition=cpu', '--time=00:10:00', '--job-name=v5-check',
                        f'--output={ROOT}/logs/profiles/v5_check_%j.out'],
                       ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_v5_branches.py', '-v'])
        options = ['--partition=a100', '--gres=gpu:a100:1', '--time=06:00:00',
                   f'--dependency=afterok:{check}', '--kill-on-invalid-dep=yes']
        task = ['tools/submit_v5_screen.py', 'task']
        if args.mode == 'new':
            options += ['--array=0-2%3', '--job-name=v5-screen',
                        f'--output={ROOT}/logs/profiles/v5_training_%A_%a.out']
        else:
            arm = args.mode[-1]
            options += [f'--job-name=v5-{arm}', f'--output={ROOT}/logs/profiles/v5_{arm}_resume_%j.out']
            task += ['--arm', arm, '--resume']
        job = submit(options, task)
        receipt = ROOT/f'logs/profiles/v5_submission_{job}.json'
        with receipt.open('x') as stream:
            json.dump(dict(mode=args.mode, cpu_job=check, gpu_job=job, arms=list(arms),
                           free_bytes_at_submission=free), stream, indent=2)
        print(f'CPU checks: {check}\nGPU job: {job}\nBranches: {arms}\nFree space: {free/1024**3:.2f} GiB')
        subprocess.run(['squeue', '-j', f'{check},{job}', '-h', '-o', '%i|%P|%T|%M|%l|%R'], check=True)


if __name__ == '__main__':
    main()
