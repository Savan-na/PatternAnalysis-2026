# Verified Command Reference and AI-Assisted Development Record

This is a cumulative record for the HipMRI project. A command is marked **verified** only when its execution and result have been reported or checked. Expected output describes a successful run; actual output records the evidence available for this milestone.

Earlier PENDING statements describe their historical batch boundaries, not the latest project status. The Section 5 integration entries below reconcile subsequent E10 training, full Validation and real prediction evidence without rewriting those earlier records. Current results and remaining submission gaps are in the task README and report/evidence register.

## A. AI Usage and Human Verification

AI assistance helped draft `dataset.py`, its synthetic tests, the project README updates, and the temporary read-only verifier. Later entries document AI-assisted NIfTI loading, resampling, synthetic tests and targeted corrections. The approved patient assignments, expected counts, and CSIRO labels came from the project evidence supplied by the student; AI did not infer or generate them. The student executed the interactive Rangpur and GitHub commands and reported their outputs. The real-data results below are from that Rangpur execution, not from the earlier local synthetic dry run. Local Git metadata was also checked while preparing this record.

Commands proposed for a future batch are **not** verified commands and must not be added to the reference as completed work. M1-B4 Batch 1 proves filename discovery, pairing, and patient-level split integrity; it did not prove image-content processing or model performance. Batch 2.1, Batch 2.2, Batch 2.3, Batch 2.4, and the Standard 3D U-Net GPU smoke-test evidence are recorded separately below.

## B. Verified Command Reference

### M1-B4 Batch 1 — HipMRI Dataset Discovery, Patient-Level Split Verification and GitHub Submission

**Status:** APPROVED — completed and pushed. **Date:** 2026-10-09. **Repository:** `Savan-na/PatternAnalysis-2026`. **Branch:** `topic-recognition`. **Commit:** `273cc68` (`Implement HipMRI discovery and frozen patient split`). The HTTPS push succeeded; the final working tree was clean and aligned with `origin/topic-recognition`.

#### Windows PowerShell — Path and file verification

The following setup was used before the SSH/SCP commands. Adapt `$repo` only if the local checkout moves.

```powershell
$repo = 'D:\Desktop\UQ26s2-COPMP3710-Pattern-Recognition-and-Analysis\Final project'
$source = Join-Path $repo 'recognition\hipmri_3d_improved_unet_hard'
$verifier = Join-Path $env:TEMP 'comp3710-hipmri-real-verify\verify_real_dataset.py'
$remote = 's4906923@rangpur.compute.eait.uq.edu.au'
```

| Command | Environment | Purpose | Expected Output | Why Expected | Actual Verified Result | Interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Four assignments in the block above | Windows PowerShell | Store the checkout, project-source, verifier and Rangpur login paths. | Variables contain strings and paths; assignment is normally silent. | PowerShell evaluates the literals and `Join-Path`. | The variables were used in the successful transfer sequence. | Valid path construction alone does **not** prove a file exists. | `$remote` is a login address, not a credential. |
| `Join-Path $repo 'recognition\hipmri_3d_improved_unet_hard'` | Windows PowerShell | Form the project-source path without manual separator handling. | A path beneath the local checkout. | `Join-Path` combines its two arguments. | The resulting `$source` was used to transfer the two Python source files. | Establishes a path value, not file existence or file validity. | Keep `$repo` pointed at the local Git checkout. |
| `Test-Path -LiteralPath $verifier -PathType Leaf` | Windows PowerShell | Check that the verifier is a file. | `True` after the file is restored. | `-PathType Leaf` requires an existing file, not merely a directory or string. | Initially `False`; after restoration `True`. | Confirms physical existence at that moment, not correct contents. | Printing `$verifier` only prints a path string and is **not** this check. |
| `Test-Path -LiteralPath (Join-Path $env:TEMP 'comp3710-hipmri-real-verify\source_review_material.txt') -PathType Leaf` | Windows PowerShell | Check that the source-review file exists. | `True` after restoration. | The requested review file was written to that TEMP path. | Initially `False`; after restoration `True`. | Confirms physical existence at that moment. | TEMP files may later be cleared by the system. |
| `Get-Content -LiteralPath (Join-Path $source 'README.md') -Raw` | Windows PowerShell | Read the complete project README as one string. | Full README text. | `-Raw` reads the file without returning one object per line. | The README content was displayed and reviewed. | Supports documentation review; it does not execute dataset code. | `-LiteralPath` avoids interpreting path characters as wildcards. |

#### Windows PowerShell — SSH and SCP operations

The successful transfer sequence used interactive UQ authentication. PowerShell backticks below continue the `scp` command onto the next line; each backtick must be the final character on its line.

```powershell
ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive $remote 'mkdir -p "$HOME/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard"'

scp -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive `
  (Join-Path $source 'dataset.py') `
  (Join-Path $source 'test_dataset.py') `
  $verifier `
  "${remote}:comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/"

ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive $remote
```

| Command | Environment | Purpose | Expected Output | Why Expected | Actual Verified Result | Interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| First `ssh ... 'mkdir -p ...'` command above | Windows PowerShell to Rangpur Bash | Create a separate verification directory under the user's Rangpur home. | Usually no output after authentication. | `mkdir -p` succeeds silently and tolerates an existing directory. | The first SSH connection returned `Connection reset`; a retry succeeded. | The retry established access and prepared the remote directory; the first error was not a dataset result. | SSH opens an encrypted remote command/session. The directory is outside Git so verification copies cannot alter the project checkout. |
| Multiline `scp ...` command above | Windows PowerShell to Rangpur | Copy only `dataset.py`, `test_dataset.py`, and `verify_real_dataset.py`. | Each file reaches `100%`. | SCP reports transfer completion for each copied file. | All three file transfers displayed `100%`. | Confirms transfer completion, not that Python executed correctly. | SCP copies files over SSH. Do not transfer MRI data or secrets. |
| Final `ssh ... $remote` command above | Windows PowerShell to Rangpur | Open the interactive Rangpur shell for verification. | A Rangpur shell prompt after UQ authentication. | The SSH server starts a login session. | Interactive login succeeded. | Allows the subsequent read-only commands to run on Rangpur. | Rangpur uses UQ credentials; GitHub/GCM credentials are separate. Never record the password. |

#### Rangpur Linux Bash — Environment and real-dataset verification

| Command | Environment | Purpose | Expected Output | Why Expected | Actual Verified Result | Interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cd "$HOME/comp3710_hipmri_verify"` | Rangpur Linux Bash | Enter the independent verification workspace. | Usually silent; subsequent Python module imports resolve. | The directory was created and the three files were transferred beneath it. | The test and verifier modules ran from this workspace. | Establishes the correct import root; it is not the Git repository. | Run the following commands from this directory. |
| `conda activate torch` | Rangpur Linux Bash | Select the existing course Python environment. | Shell prompt may change to show `torch`. | Conda places that environment's Python on `PATH`. | The subsequent Python commands ran in the activated environment. | Establishes the interpreter context, not GPU use. | No package installation occurred. |
| `python --version` | Rangpur Linux Bash | Check the selected Python interpreter. | A Python version string. | `python` reports its interpreter version. | `Python 3.11.15`. | Confirms the environment's Python version; it does not validate data. | Version is the Rangpur result, not the Windows or WSL version. |
| `CUDA_VISIBLE_DEVICES='' python -B -m unittest recognition.hipmri_3d_improved_unet_hard.test_dataset -v` | Rangpur Linux Bash | Run focused synthetic tests against the transferred production module. | Verbose test names followed by `Ran 11 tests` and `OK`. | The test module contains 11 cases using empty synthetic files. | 11 tests ran; all passed; final status `OK`. | Supports filename, pairing, error-path and split logic on Rangpur; synthetic files are not real-data evidence. | Empty `CUDA_VISIBLE_DEVICES` hides GPUs; `-B` avoids `.pyc` writes; `-m` runs the module; `-v` prints each test. |
| `CUDA_VISIBLE_DEVICES='' python -B -m recognition.hipmri_3d_improved_unet_hard.verify_real_dataset` | Rangpur Linux Bash | Call the actual `dataset.py` discovery and split APIs on the real dataset root. | Counts of 211 pairs, 38 patients, 26/143 train, 6/38 validation, 6/30 test; empty error/overlap lists; final `PASS`. | The verifier invokes production pairing and coverage checks against `/home/groups/comp3710/HipMRI_Study_open`. | 211 matched pairs; 211 distinct MRIs and masks; 38 patients; train 26/143, validation 6/38, test 6/30; no missing or unexpected patients, missing paired paths, unmatched files, overlap or Week leakage; `K019_Week1` in Train; seed provenance 3710; `PASS`. | Verifies real filename discovery, MRI/mask pairing and frozen patient-level split integrity. It does **not** verify NIfTI array loading, resampling, tensor creation, Dice/IoU or trained models. | CPU-only filename/path checks; no MRI arrays were loaded and no GPU job was run. `-B` and `-m` have the meanings above. |

#### Windows PowerShell — Git review and commit

Run Git commands from the local `PatternAnalysis-2026` checkout. Git status markers before staging were `M` for a tracked file modified in the working tree and `??` for an untracked file. Staging with `git add` prepares content for a commit; a commit records it locally; `git push` updates the remote branch.

| Command | Environment | Purpose | Expected Output | Why Expected | Actual Verified Result | Interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `git branch --show-current` | Windows PowerShell, local Git checkout | Check the active branch. | `topic-recognition`. | The assessment work was on that branch. | `topic-recognition`. | Identifies the current branch, not its remote sync state. | A Git command run in the Rangpur temporary directory returned `Not a git repository`; return to the local checkout. |
| `git diff -- recognition/hipmri_3d_improved_unet_hard/README.md` | Windows PowerShell, local Git checkout | Review unstaged README changes. | Patch text before staging; empty after those changes are committed. | `git diff` compares the working tree with the index. | The documentation correction was reviewed before submission. | Shows unstaged content only; staged changes require `git diff --cached`. | A clean diff alone does not prove the remote is current. |
| `git status --short` | Windows PowerShell, local Git checkout | Inspect concise tracked/untracked state. | `M` and `??` before staging; empty after completion. | Git reports changed and untracked paths. | The final working tree was clean. | Clean means no local uncommitted file changes; it does **not** prove a push succeeded. | Earlier `M` and `??` were expected during implementation. |
| `git diff --check` | Windows PowerShell, local Git checkout | Detect whitespace errors in unstaged changes. | No whitespace-error lines. | Git reports relevant errors rather than a success message. | No relevant whitespace errors were reported. | Checks diff formatting, not program behavior. | LF-to-CRLF conversion warnings on Windows are warnings, not test failures or whitespace errors. |
| `git add -- recognition/hipmri_3d_improved_unet_hard/dataset.py recognition/hipmri_3d_improved_unet_hard/test_dataset.py recognition/hipmri_3d_improved_unet_hard/README.md` | Windows PowerShell, local Git checkout | Stage only the three approved project files. | Usually silent; staged files appear in the index. | `git add --` copies those paths into Git's staging area. | The three paths were included in commit `273cc68`. | Staging does not itself commit or push. | `--` ends option parsing before paths. |
| `git diff --cached --stat` | Windows PowerShell, local Git checkout | Summarise staged files before commit. | Three project paths with change counts. | `--cached` compares the index with HEAD. | The staged three-file change was reviewed. | Confirms the staged scope, not that tests passed. | Run before committing; after commit the staged diff is empty. |
| `git diff --cached --check` | Windows PowerShell, local Git checkout | Check staged whitespace. | No output on success. | Git prints only relevant whitespace errors. | No output. | No relevant staged whitespace errors were reported; it is not a code-quality test. | Distinct from unstaged `git diff --check`. |
| `git log -1 --format="%h %s%nAuthor: %an <%ae>"` | Windows PowerShell, local Git checkout | Inspect the latest committed hash, subject and author. | One commit line and one author line. | The format tokens select those fields from HEAD. | `273cc68 Implement HipMRI discovery and frozen patient split` with an author line. | Confirms the commit exists locally. | The later repeated `git commit` returned `nothing to commit, working tree clean`; that repeated invocation did **not** create the commit. |
| `git diff-tree --no-commit-id --name-only -r HEAD` | Windows PowerShell, local Git checkout | List paths contained in the latest commit. | The approved README, `dataset.py`, and `test_dataset.py`. | `diff-tree` enumerates HEAD's changed paths. | Exactly those three project paths were listed. | Verifies committed scope, not remote publication. | This command was also rechecked while preparing this record. |
| `git status -sb` | Windows PowerShell, local Git checkout | Show branch/tracking state concisely. | `ahead 1` before pushing; no `ahead` after push. | The indicator compares local and tracking commits. | Final line: `## topic-recognition...origin/topic-recognition`. | No `ahead` plus the verified push indicates the local tracking state is aligned. | A clean working tree alone cannot establish this; inspect tracking or hashes. |

Before the successful commit, Git reported `Author identity unknown`; the local author identity was configured and the commit was then verified with `git log`. The exact identity-configuration command is not recorded here because it was not supplied as verified command evidence. Do not treat the later `nothing to commit` message as the creation of `273cc68`.

#### Windows PowerShell — GitHub HTTPS and Git Credential Manager

| Command | Environment | Purpose | Expected Output | Why Expected | Actual Verified Result | Interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `git remote -v` | Windows PowerShell, local Git checkout | Inspect fetch and push URLs. | `origin` and `upstream` URLs. | Git lists configured remotes. | `origin` was changed to `https://github.com/Savan-na/PatternAnalysis-2026.git`; `upstream` remained `https://github.com/shakes76/PatternAnalysis-2026.git`. | Distinguishes the student's writable remote from the course upstream. | Inspect again after `set-url` to confirm the change. |
| `git credential-manager --version` | Windows PowerShell, local Git checkout | Check the installed Git Credential Manager. | A version string. | GCM reports its installed build. | `2.6.1+786ab03440ddc82e807a97c0e540f5247e44cec6`. | Confirms GCM is available, not that authentication has completed. | No secret is printed by this command. |
| `git config --get credential.helper` | Windows PowerShell, local Git checkout | Inspect the active credential helper. | `manager` after configuration. | Git returns the resolved helper setting. | `manager`. | Confirms Git will invoke GCM for applicable HTTPS credentials. | This is GitHub authentication configuration, not Rangpur login. |
| `git remote set-url origin https://github.com/Savan-na/PatternAnalysis-2026.git` | Windows PowerShell, local Git checkout | Switch `origin` from SSH to HTTPS. | Usually silent; `git remote -v` shows the new URL. | The command updates local remote configuration. | `origin` showed the HTTPS URL afterward. | Changes transport for future push/fetch; does not itself publish a commit. | `upstream` was left unchanged. |
| `git config --local credential.helper manager` | Windows PowerShell, local Git checkout | Select GCM for this repository. | Usually silent; later helper query returns `manager`. | Local Git config overrides the helper for this checkout. | `git config --get credential.helper` returned `manager`. | Establishes helper selection, not proof of a successful push. | HTTPS/GCM avoids the GitHub SSH private-key passphrase prompt. |
| `git push origin topic-recognition` | Windows PowerShell, local Git checkout | Publish the local assessment commit. | Remote update of `topic-recognition`. | `origin` points to the student's GitHub repository. | Push succeeded over HTTPS, updating `topic-recognition` from `3d22172` to `273cc68`. | Confirms publication of that branch update; it does not validate MRI content. | GitHub/GCM authentication is separate from the UQ password used for Rangpur. |
| `git status -sb` | Windows PowerShell, local Git checkout | Confirm the tracking state after push. | `topic-recognition...origin/topic-recognition` without `ahead`. | The tracking ref advances after a successful push. | Final line had no `ahead`; local HEAD and `origin/topic-recognition` were also checked as the same commit. | Confirms the local tracking ref is synchronized at `273cc68`. | Do not infer remote sync from an empty `git status --short` alone. |

HTTPS with Git Credential Manager handles GitHub authentication without prompting for the GitHub SSH private-key passphrase. A GitHub password, an SSH-key passphrase and a UQ/Rangpur password serve different systems and are not interchangeable. No credential or key material belongs in this record.

#### Remaining limitations after M1-B4 Batch 1

The following were **not completed**: production NIfTI array loading; MRI/mask voxel-level alignment checks during preprocessing; 3D resampling or downsampling; MRI intensity normalisation; six-class mask tensor encoding; Standard 3D U-Net training; Improved 3D U-Net training; and Dice/IoU evaluation. The next development milestone is **M1-B4 Batch 2 — 3D Preprocessing & Spatial Alignment**. Its commands and outcomes must be recorded only after execution and verification.

### M1-B4 Batch 2.1 — NIfTI Content Loading

**Status:** Technically approved after student-run Rangpur verification and source review. The results in this section were supplied by the student; Codex did not connect to Rangpur or independently rerun the real-data checks. Exact original command text is preserved only where available. An identified procedure is not presented as a verbatim terminal command.

The following is the supplied exact combined unit-test command for the corrected revision. The initial 27-test run exercised the same two modules, but its verbatim command transcript is unavailable:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_volume_io \
  -v
```

`CUDA_VISIBLE_DEVICES=''` hides GPUs from the process; `-B` prevents Python bytecode writes; `-m unittest` runs the test modules through Python's test runner; `-v` prints individual test names and outcomes. These flags do not, by themselves, prove that image arrays were read correctly. The synthetic NIfTI tests use small generated NIfTI files; the real-pair procedure below used HipMRI files.

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Windows PowerShell | `Test-Path` checks for local `volume_io.py` and `test_volume_io.py`; exact original expressions unavailable. | Confirm both source files existed before transfer. | Existing-file checks return `True`. | Both files had been created locally for Batch 2.1. | The student reported verifying both paths; the original Boolean lines were not supplied. | Supports a local file-presence check. | File presence does not validate source contents or remote execution; no verbatim command is claimed. |
| Windows PowerShell to Rangpur | Interactive `scp` transfer of `volume_io.py` and `test_volume_io.py`; exact original command and transfer log unavailable. | Place the actual loader and tests in the user's remote verification workspace. | Both transfers complete without error. | Rangpur needs the current source and test modules to import them. | The student reported the transfer and subsequently ran those modules on Rangpur. | The reported remote execution is consistent with a completed transfer. | No `100%` transfer lines or checksums were supplied for this batch. |
| Windows PowerShell to Rangpur | Interactive SSH login; exact original command and login transcript unavailable. | Open a Rangpur shell for manual, CPU-only verification. | A remote shell after authentication. | The student had access to the COMP3710 Rangpur account. | The student reported a successful session and supplied Rangpur test results. | Identifies the human-run remote environment. | No credential, password or private key is recorded; Codex did not log in. |
| Rangpur Bash | `cd "$HOME/comp3710_hipmri_verify"` | Select the established temporary import root. | Usually silent; subsequent package imports resolve. | The Python package files were copied beneath this workspace. | The student reported running the verification there; no separate `pwd` output was supplied. | Places later commands in the intended workspace. | This is a temporary copy, not the local Git checkout. |
| Rangpur Bash | `conda activate torch` | Select the course Python environment. | Activated environment; version imports work. | The existing environment supplied the Python and imaging dependencies. | The student reported using the `torch` environment. | Establishes the reported interpreter context. | Prompt text and an independent `which python` output were not supplied for this batch. |
| Rangpur Bash | Dependency-version inspection; exact original Python command unavailable. | Check the libraries used by NIfTI loading. | NiBabel and NumPy version strings. | Their import is required by `volume_io.py`. | NiBabel `5.4.2`; NumPy `2.4.6`. | Documents the student-reported tested library versions. | The exact version-inspection command and complete terminal transcript are unavailable. |
| Rangpur Bash | Initial combined unit-test procedure on the dataset and NIfTI modules; exact original command unavailable. | Exercise 11 discovery tests and 16 synthetic NIfTI tests before real-data loading. | Results end in `OK`. | The test modules contained 11 and 16 cases at that revision. | **27/27 PASS**: 11 dataset and 16 NIfTI tests. | The initial implementation passed synthetic checks. | No initial runtime line or verbatim command was supplied; synthetic success did not establish compatibility with real HipMRI mask-unit metadata. |
| Rangpur Bash | Real `K019_Week1` one-pair load and assertion procedure; exact original Python here-document unavailable. | Decode one approved real MRI/mask pair. | Loaded arrays and validated metadata, or a clear error. | `discover_dataset()` identifies the pair and `load_volume_pair()` validates its content. | `ValueError: K019_Week1: spatial unit mismatch: MRI='mm', segmentation='unknown'.` | The first real-data attempt **FAILED** because the loader required equal declared units. | This failure was not a shape, affine, spacing or orientation mismatch; exact procedure text is unavailable. |
| Rangpur Bash | Read-only 211-pair NIfTI header-audit procedure; exact original Python here-document unavailable. | Diagnose whether the unit mismatch was isolated and assess numerical grid consistency. | Counts of unit combinations, orientations and geometric mismatches; a final audit result. | Headers can be inspected without loading every image array. | **211 pairs:** MRI=`mm`, mask=`unknown` in all; LPS/LPS in all; zero shape, affine, spacing or orientation grid mismatches; zero unexpected unit combinations; audit `PASS`. | Student-run evidence supports a narrow metadata compatibility rule across the indexed dataset. | It did not decode all 211 image arrays or establish annotation accuracy; Codex did not perform this audit. |
| Rangpur Bash | Combined unit-test command in the block above, corrected implementation. | Recheck old behavior and six added compatibility cases. | Verbose results ending in `OK`. | The modules now contain 11 dataset and 22 synthetic NIfTI tests. | `Ran 33 tests in 0.226s`, then `OK` (**33/33 PASS**). | Synthetic regressions passed after the targeted correction. | Generated NIfTI cases cannot replace real-image testing. |
| Rangpur Bash | Real `K019_Week1` one-pair load and assertion procedure; exact original Python here-document unavailable. | Confirm the corrected loader reads one real pair with conditional unit inference. | Matching 3D arrays, valid labels and geometry, explicit mask-header provenance, final PASS. | The 211-pair header audit justified testing the `mm`/`unknown` path against the actual files. | MRI and mask shapes `(256, 256, 144)`; dtypes `float32` and `uint8`; working unit `mm`; mask header unit `unknown`; inference `True`; axis codes `('L', 'P', 'S')`; labels `(0, 1, 2, 3, 4, 5)`; `PASS: K019 Week1 real NIfTI pair loaded and validated`. | **PASS for one real pair's content loading and implemented validations.** | Does not prove that every array loads, that annotations are clinically correct, or that preprocessing or model training works. |

The numerical tolerances in the corrected loader remain absolute `1e-4` for affine elements and `1e-5` for spacing, with zero relative tolerance. The compatibility rule preserves the mask header value `unknown` separately from the MRI-declared working unit `mm` and records that the mask unit was inferred. No NIfTI file was rewritten.

### M1-B4 Batch 2.2 — 3D Physical Resampling and Boundary Correction

**Status:** Technically approved after student-run Rangpur regression and two real-pair acceptance checks. The Rangpur numbers below are **student-reported terminal results**, not tests independently rerun by Codex. The exact installation, dependency-check and final regression commands below were preserved by the student and supplied after the earlier documentation draft; earlier "unavailable" wording reflected what Codex had then, not the absence of these commands from the student's history. Complete training-only preflight and real-case Python here-document commands have not been supplied for this record, so their procedures remain summaries rather than verbatim commands.

The student supplied these exact Rangpur `torch` environment commands for SciPy installation and dependency validation:

```bash
python -m pip install --only-binary=:all: --no-deps "scipy==1.17.1"
python -c "import numpy, nibabel, scipy; from scipy.ndimage import affine_transform; print('NumPy:', numpy.__version__); print('NiBabel:', nibabel.__version__); print('SciPy:', scipy.__version__); print('SciPy affine_transform: available')"
python -m pip check
```

The student also supplied the exact final Rangpur regression command and observed terminal ending:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_volume_io \
  recognition.hipmri_3d_improved_unet_hard.test_resampling \
  -v
```

```text
Ran 53 tests in 0.219s

OK
```

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rangpur `torch` environment | Training-only spacing and memory preflight; exact original audit commands have not been supplied for this record. | Choose an initial physical spacing using training patients only and estimate one-pair resampling memory. | 26 training patients, 143 volumes, a patient-weighted spacing summary and per-case planning estimates. | The frozen split identifies training patients; volume headers provide spacing and shapes. | Patient-level median spacing `(1.6796875, 1.6796875, 1.55999755859375)` mm; 143 volumes examined; largest 4× estimate 365.7 MiB, K019 Week1 365.4 MiB, zero above 512 MiB. | Supports the first training-derived spacing candidate and estimated 512 MiB guard. | Header/planning evidence, not measured peak RSS or proof of clinical optimality; no validation/test spacing was used to fit the target. |
| Rangpur `torch` environment | Exact `python -m pip install --only-binary=:all: --no-deps "scipy==1.17.1"`, version-import command and `python -m pip check` shown above. | Make SciPy interpolation available and check the environment. | SciPy imports, `scipy.ndimage.affine_transform` exists, and dependency check reports no broken requirements. | Resampling calls SciPy; the existing NumPy and NiBabel imports must remain usable. | `Successfully installed scipy-1.17.1`; NumPy `2.4.6`; NiBabel `5.4.2`; SciPy `1.17.1`; `SciPy affine_transform: available`; `No broken requirements found`. | Establishes the student-reported remote dependency state before implementation verification. | Commands and listed results were supplied by the student after the earlier draft; the full installation transcript is not reproduced. Codex did not install packages on Rangpur. |
| Windows PowerShell / WSL `torch` Python | `python -c` AST/JSON checks and a PowerShell here-string piped to WSL Python with an in-memory NiBabel import shim; full synthetic-run transcript is not preserved here. | Check syntax/configuration and execute genuine SciPy resampling tests despite missing local NiBabel. | Syntax/JSON parse and synthetic tests pass. | Synthetic tests directly construct `LoadedVolumePair` arrays; they do not decode NIfTI files. | Initial **16/16 resampling tests passed**; after the boundary fix **20/20 passed**. The existing **11/11 dataset tests** also passed locally. | Local synthetic evidence for grid, interpolation, labels and memory guard. | The shim bypassed NiBabel import only; no local real NIfTI decoding or Rangpur access occurred. These runs are separate from student-run Rangpur evidence. |
| Rangpur `torch` environment | Initial combined discovery/loading/resampling test procedure; exact original command has not been supplied for this record. | Recheck the prior 33 tests and first 16 resampling tests. | 49 tests ending in `OK`. | The suites contained 11 discovery, 22 NIfTI and 16 resampling tests. | **49/49 PASS**, reported by the student. | Initial regression success before the boundary correction. | It does not validate the subsequently modified boundary policy. |
| Rangpur `torch` environment | Initial real K019 Week1 and S035 Week0 resampling procedures; exact original Python commands/output have not been supplied for this record. | Exercise one identity-spacing case and one in-plane spacing conversion on actual HipMRI data. | Each pair loads and resamples without geometry, label or memory-guard failure. | Production discovery and loading supply validated one-pair inputs. | The student reported that both real cases passed before source review identified the boundary issue. | Demonstrates those initial one-pair runs completed. | Numerical before-fix output and peak RSS transcripts have not been supplied; success did not rule out erased edge labels. |
| WSL `torch` Python, CPU | Synthetic 3D SciPy boundary reproduction; original PowerShell here-string was executed during Codex source review, but is not copied here as a verbatim command. | Compare boundary modes with a face-touching mask and constant-intensity MRI. | `constant` loses valid edge samples; alternative modes reveal their behavior. | The first output center maps to source index about `-0.05208`, inside voxel-edge bounds but outside center bounds. | Output shape `(18, 18, 4)`; `constant` mask had zero labelled voxels on both tested X faces, while `grid-constant` retained 72 on each. For MRI intensity 100, a tested edge value was `0` with `constant`, about `94.79` with `grid-constant`, and `100` with `nearest`. | Direct synthetic reproduction of the boundary defect and basis for separate MRI/mask policies. | This is synthetic evidence; it is not a real-image assessment. |
| Rangpur `torch` environment | Exact final `CUDA_VISIBLE_DEVICES='' python -B -m unittest ... -v` command shown above. | Regress all three modules after the boundary fix. | 53 tests ending in `OK`. | The previous 33 tests plus 20 current resampling tests total 53. | `Ran 53 tests in 0.219s`, then `OK` (**53/53 PASS**), reported by the student. | Student-run regression evidence for the corrected revision. | Synthetic tests alone do not establish all 211 real arrays can be resampled. |
| Rangpur `torch` environment, CPU | Student-executed Bash loop for `K019:1` and `S035:0`: set `CASE_ID`, wrapped each Python here-document invocation with `/usr/bin/time -v`, and called `discover_dataset`, `load_volume_pair` and `resample_volume_pair`. | Verify two actual arrays with the corrected boundary policy. | Exit status 0, retained valid labels and recorded shape/count/memory values. | Both were previously discoverable and loadable; corrected resampling validates output geometry and labels. | K019 Week1: `(256,256,144)` → same shape, prostate voxels `17,245 → 17,245`, maximum resident set size `182,120 KiB`, exit 0. S035 Week0: `(256,256,128)` → `(286,286,128)`, prostate voxels `3,478 → 4,448`, maximum resident set size `185,740 KiB`, exit 0. | Two real-case acceptance examples passed after the fix; the RSS values are measurements, unlike the preflight estimates. | The `/usr/bin/time -v` invocation pattern and results are verified from the student's account; the complete loop and Python here-document were not supplied verbatim and are not reconstructed here. Only two real MRI arrays were resampled. Voxel-count change is not a Dice/IoU or clinical-quality result. |

The earlier header audit covered 211 pair **headers**, not 211 decoded and resampled arrays. Neither this command reference nor the source review claims completed full-dataset resampling, clinically validated contours, model training, or performance evaluation.

### M1-B4 Batch 2.3 — MRI Intensity Normalisation

**Status:** Technically approved after student-run Rangpur regression and two real-pair checks. The remote outputs below were supplied by the student; Codex did not access Rangpur. Exact commands are reproduced only where supplied. The original intensity-audit and SCP/login command transcripts, and the full real-case Python here-document, are not available in this record.

The exact student-executed final combined regression command and reported terminal ending were:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_volume_io \
  recognition.hipmri_3d_improved_unet_hard.test_resampling \
  recognition.hipmri_3d_improved_unet_hard.test_intensity_normalization \
  -v
```

```text
Ran 71 tests in 0.214s

OK
```

For real-case verification, the student used a Bash loop over `K019:1` and `S035:0`. Each Python here-document was run with the supplied invocation pattern `/usr/bin/time -v env CUDA_VISIBLE_DEVICES='' python -B -`. The Python code called `discover_dataset`, `load_volume_pair`, `resample_volume_pair`, and `normalize_volume_pair` and asserted finite `float32` output, preservation of original zero positions, unchanged mask and affine, and originally selected-voxel output mean/std within `1e-4` of `0`/`1`. The complete here-document is not reproduced because its exact original text was not supplied.

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rangpur, two training cases after resampling | Student-run initial intensity audit; exact command unavailable in this record. | Inspect the input intensity distributions before selecting a normalisation policy. | Case-specific nonzero and zero statistics. | MRI intensities can vary across scans even after spatial resampling. | K019 Week1: zero fraction `0.2372`, nonzero mean/std `62.978`/`71.8581`, nonzero 0.5/50/99.5 percentiles `1.0`/`31.0`/`298.0`. S035 Week0: zero fraction `0.3184`, nonzero mean/std `64.4399`/`73.9491`, corresponding percentiles `0.012912326492369175`/`33.10286331176758`/`292.36941619872994`. | Student-reported **input-distribution evidence** supporting an initial per-volume nonzero policy. | These are two training cases, not post-normalisation results or proof of clinical optimality. |
| Local WSL `torch` Python | Focused synthetic `test_intensity_normalization` run with an in-memory NiBabel import shim; full local command is recorded in the Codex execution history, not reproduced here. | Exercise clipping, nonzero Z-scoring, zero preservation, determinism, metadata, mask integrity and rejection cases. | 18 tests ending in `OK`. | The new test module contains 18 generated-array cases. | **18/18 PASS**, Codex-reported local result. | Synthetic evidence for the algorithm and failure checks. | NiBabel was unavailable locally; no real NIfTI was decoded in this run. |
| Windows PowerShell to Rangpur | Student manual SCP transfer of three Batch 2.3 files and interactive SSH login; exact original commands unavailable in this record. | Place the implementation, test and config in the student's verification workspace. | Files transfer and the remote session opens. | Rangpur needs those files to import the new module and run the tests. | The student reported successful transfer and remote verification. | Identifies the student-executed handoff, not a Codex remote action. | No transfer transcript or credential is reproduced. |
| Rangpur `torch` environment, GPU hidden | Exact combined `CUDA_VISIBLE_DEVICES='' python -B -m unittest ... -v` command shown above. | Regress the previous 53 tests together with 18 normalisation tests. | 71 tests ending in `OK`. | 53 previously verified tests plus 18 new tests total 71. | `Ran 71 tests in 0.214s`, then `OK` (**71/71 PASS**). | Student-run regression evidence for the transferred Batch 2.3 revision. | Synthetic tests do not replace real-image or full-dataset checks. |
| Rangpur CPU, K019 Week1 | Bash-loop real-case procedure using `CASE_ID`, the `/usr/bin/time -v` invocation pattern above and the four production APIs. | Validate one real resampled MRI and record output statistics and process memory. | Finite `float32`, original zeros retained, mask/affine unchanged, selected output mean/std near 0/1, exit 0. | The policy clips and Z-scores only original MRI nonzeros; the accepted resampling output supplies the paired geometry. | Shape `(256, 256, 144)`; selected voxels `7,199,017`; bounds `1.0`/`298.0`; clipped mean/std `62.71922972261352`/`70.6617517907729`; output selected mean/std `-2.0519089452645352e-09`/`0.9999999993619909`; mask and affine checks **PASS**; peak process RSS `341780 KiB`; wall time `5.66 s`; exit `0`. | One student-run real normalisation acceptance example passed. | Peak RSS covers the whole discovery/loading/resampling/normalisation verification process, not only the normaliser. |
| Rangpur CPU, S035 Week0 | Same Bash-loop real-case procedure with the second `CASE_ID`. | Validate a second real MRI after in-plane resampling. | The same output, geometry and integrity assertions pass. | The same fixed policy is applied independently per MRI. | Shape `(286, 286, 128)`; selected voxels `7,136,647`; bounds `0.012912326492369175`/`292.36941619872994`; clipped mean/std `64.3451434267759`/`73.6351732255404`; output selected mean/std `-1.364474822943357e-09`/`0.9999999989009823`; mask and affine checks **PASS**; peak process RSS `345368 KiB`; wall time `5.54 s`; exit `0`. | Second student-run real normalisation acceptance example passed. | The two cases do not establish full 211-volume success or clinical accuracy. |

The 71-test result and two real-case values were reported from the student's Rangpur execution, not rerun during this documentation update. Full 211-volume normalisation, model training and clinical performance evaluation remain unverified.

### M1-B4 Batch 2.4 — Minimal PyTorch 3D Patch Dataset

**Status:** Technically approved after source review, student-run Rangpur regression and two real DataLoader patch checks. The remote results below were supplied by the student; Codex did not access Rangpur. The exact combined regression command is available. The original SCP/login commands and complete real-data Python here-document have not been supplied for this record and are not reconstructed.

The exact student-executed five-module regression command and reported terminal ending were:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_volume_io \
  recognition.hipmri_3d_improved_unet_hard.test_resampling \
  recognition.hipmri_3d_improved_unet_hard.test_intensity_normalization \
  recognition.hipmri_3d_improved_unet_hard.test_patch_dataset \
  -v
```

```text
Ran 87 tests in 8.646s

OK
```

For the real check, the student ran a Python here-document with `/usr/bin/time -v env CUDA_VISIBLE_DEVICES='' python -B -`. It used `HipMRIPatchDataset`, `Subset`, and `DataLoader` with `batch_size=1`, `num_workers=0`. The complete original Python command is not available here; the verified procedure and outputs follow.

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local WSL `torch` Python | Focused `test_patch_dataset` synthetic suite with mocked preprocessing and an in-memory NiBabel import shim; exact local command is in the Codex execution history. | Check tensor axes and dtypes, aligned crops, sampling, split restrictions, DataLoader collation and non-mutation. | 16 tests ending in `OK`. | The new module has 16 synthetic test cases. | **16/16 PASS**, Codex-reported local execution. | Tests Patch Dataset logic with constructed arrays. | No real NIfTI was loaded; the production preprocessing calls were mocked. |
| Windows PowerShell to Rangpur | Student manual SCP transfer of `patch_dataset.py`, `test_patch_dataset.py`, and `patch_dataset_config.json`, followed by interactive login and `torch` activation; exact transcript unavailable here. | Place the new files in the verification workspace and select its Python environment. | Files transfer and the new module imports remotely. | Rangpur requires current files and the existing dependencies. | The student reported transfer/login and successfully ran the tests and real check. | Identifies student-executed remote setup. | No password, SCP progress log, or invented command is included. |
| Rangpur `torch` environment, GPU hidden | Exact five-module command shown above. | Regress earlier pipeline tests with Patch Dataset tests. | 87 tests ending in `OK`. | Previous 71 plus 16 new tests total 87. | `Ran 87 tests in 8.646s`, then `OK` (**87/87 PASS**). | Student-run combined regression evidence. | Most tests are synthetic; this command alone does not verify real patch output. |
| Rangpur CPU, real train patch | Timed Python procedure with `HipMRIPatchDataset`, `Subset`, `DataLoader(batch_size=1, num_workers=0)`. | Check a prostate-aware K019 Week1 patch using the approved on-demand pipeline. | `[1,1,64,64,64]` float32 MRI, `[1,64,64,64]` int64 mask, finite values, labels 0–5 and prostate present. | The train sampler selects a class-5 voxel when foreground sampling succeeds. | Patient `K019`, Week `1`; MRI `(1,1,64,64,64)` float32; mask `(1,64,64,64)` int64; labels `[0,1,2,3,5]`; crop origin DHW `[40,60,64]`; mode `prostate`; **PASS**. | One real training patch contained prostate label 5 with aligned tensor shapes. | This is one patch, not all longitudinal scans or full-volume inference. |
| Rangpur CPU, real validation patch | Same timed procedure using the frozen validation split. | Check deterministic center-crop output for B040 Week0. | The same batch dtypes/shapes, finite values and valid labels; `center` mode. | Validation selection has no training-time crop randomness. | Patient `B040`, Week `0`; MRI `(1,1,64,64,64)` float32; mask `(1,64,64,64)` int64; labels `[1,2,3,4,5]`; crop origin DHW `[32,87,87]`; mode `center`; **PASS**. | One real validation center patch passed. | A center crop may omit the prostate in other cases. |

The real procedure ended `PASS: REAL PATCH DATASET VERIFICATION`; `/usr/bin/time -v` reported wall time **11.72 s**, maximum resident set size **768352 KiB**, and exit status **0**. RSS measures the whole Python verification process, not GPU memory or an isolated Dataset allocation. Only two real patches were accepted. Full 211-volume processing, training, full-volume inference, clinical accuracy and Dice/IoU evaluation remain unverified.

### Standard 3D U-Net — Architecture and First Real GPU Optimisation Step

**Status and evidence:** Source review, local synthetic CPU verification, student-run Rangpur CPU regression, and one real GPU optimisation step passed. At the original milestone documentation stage, the last pushed commit supplied was `938449a`, branch `topic-recognition`, and the architecture and smoke-test files were pending submission. They were subsequently committed and pushed in `dc20457`: `Implement and verify Standard 3D U-Net baseline`. Remote outputs below were supplied by the student and were not independently rerun by Codex. The student subsequently supplied the original Slurm allocation transcript and cancellation command, recorded below. Exact dependency-inspection and six-module CPU regression command transcripts were not supplied in this documentation request; their verified procedures and results are recorded without reconstructed commands.

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local WSL PyTorch, CPU | Codex synthetic `test_standard_unet` execution. | Check shapes, channels, skip concatenation, finite logits, integer-label cross entropy, finite backward gradients, an SGD update and seeded initialisation. | 10 tests ending in `OK`. | The architecture suite adds 10 focused synthetic tests. | Codex reported **10/10 PASS**. | Synthetic model execution and optimiser update worked locally. | Reduced channel widths were used for practical CPU tests; synthetic optimisation does not establish segmentation quality. |
| Rangpur login node, CPU-only | Student transfer of `modules.py` and `test_standard_unet.py`, followed by the six-module combined regression suite. | Regress the approved preprocessing/Patch Dataset pipeline with architecture tests. | 97 tests ending in `OK`. | 87 earlier tests plus 10 new architecture tests total 97. | `Ran 97 tests in 10.212s`, then `OK`; a nonfatal CUDA driver warning appeared. | **97/97 PASS** on CPU despite the warning. | No GPU optimisation was performed on the login node; this result did not verify allocated-node CUDA compatibility. |
| Rangpur Slurm | Student executed the `srun` request below; cancelled queued job `647036` using Ctrl+C and later executed `scancel 647036`; repeated the request with `--partition=a100`. | Acquire an authorised GPU compute session. | An allocated GPU compute node. | GPU work requires a compute allocation. | Job **647036** queued and was cancelled; the second request allocated job **647039**, partition **a100**, node **a100-5**, GPU **NVIDIA A100-PCIE-40GB**. | Successful GPU allocation after the queued attempt. | Allocation evidence does not establish model correctness; separate CPU and GPU results follow. |
| Allocated `a100-5` session | Student inspected versions, CUDA availability and CUDA tensor creation. | Establish that the allocated environment can execute CUDA operations. | CUDA available and a CUDA tensor created successfully. | A working GPU/driver/PyTorch combination is required before the smoke test. | PyTorch **2.13.0+cu130**, CUDA build **13.0**, NVIDIA driver **590.48.01**; `torch.cuda.is_available() = True`; CUDA tensor creation passed. | Basic CUDA execution succeeded on the allocated node. | Exact inspection commands were not supplied; broader compatibility or training stability is not established. |
| Allocated `a100-5` session | Exact GPU command below. | Execute exactly one real K019 Week1 forward/loss/backward/SGD update. | Final `PASS: ONE REAL STANDARD 3D U-NET GPU OPTIMISATION STEP` after all checks. | The script raises an error and exits non-zero for incompatible batches, non-finite logits/loss/gradients or no parameter change. | Final PASS; detailed values below. | One real float32 training-pipeline optimisation step completed. | No full epoch, validation-performance measurement, Dice/IoU result or checkpoint was produced. |

The original first interactive Slurm request supplied by the student was:

```bash
srun \
  --partition=comp3710 \
  --gres=gpu:a100:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --time=00:20:00 \
  --pty /bin/bash
```

Job **647036** queued and was cancelled by the student using Ctrl+C. The student later executed:

```bash
scancel 647036
```

The second request used the same resource settings with `--partition=a100`; Slurm allocated job **647039** on node **a100-5**.

The exact student-executed GPU command was:

```bash
python -B -m recognition.hipmri_3d_improved_unet_hard.smoke_standard_unet
```

`-B` prevents Python bytecode-cache writes; `-m` runs the project module with package-relative imports. The script uses the default `StandardUNet3D(in_channels=1, out_channels=6, base_channels=16)`, **5,646,470 trainable scalar parameters**, seed **3710**, a configured 64³ patch, foreground probability **1.0**, and `DataLoader(batch_size=1, num_workers=0, shuffle=False)`. MRI loading/preprocessing runs on CPU; only the patch tensors move to GPU. The model performs one float32 forward pass producing finite raw logits `(1, 6, 64, 64, 64)`, cross entropy against integer labels, `zero_grad(set_to_none=True)`, one backward pass and one SGD update with learning rate **0.001**, without AMP.

| Observed check | Student-supplied result |
| --- | --- |
| Case | `K019_Week1` |
| Crop origin DHW | `[40, 60, 64]` |
| Labels | `[0, 1, 2, 3, 5]` |
| MRI batch | `(1, 1, 64, 64, 64)` float32 |
| Target batch | `(1, 64, 64, 64)` torch.long |
| Gradient checks | All **64** trainable parameter tensors had finite gradients |
| Initial cross-entropy loss | **1.92833924** |
| Parameter-update check | **59/64** trainable parameter tensors changed after exactly one update |
| GPU allocated | **55286272 bytes** |
| GPU reserved | **465567744 bytes** |
| Peak GPU allocated | **441744384 bytes** |
| Peak GPU reserved | **465567744 bytes** |

```text
PASS: ONE REAL STANDARD 3D U-NET GPU OPTIMISATION STEP
```

**Interpretation:** The result demonstrates training-pipeline executability on one real HipMRI training patch. The reported loss is an initial single-patch loss, not a validation or convergence metric. Finite gradients do not establish model convergence. The 59 changed parameter tensors satisfy the requirement that at least one trainable parameter changes; scalar parameter count and parameter-tensor count describe different quantities. GPU memory measurements belong to this exact model, patch, dtype and execution configuration and do not predict full-training memory. No full-epoch training, validation performance, Dice/IoU evaluation or checkpointing occurred.

### Batch 1 — Preprocessing Throughput and Full Train/Validation Coverage

**Evidence and state:** Student-executed Rangpur audits passed; remote outputs were supplied by the student, not independently rerun by Codex. The baseline repository state is `dc20457` on `topic-recognition`. The profiler is implemented but uncommitted; this documentation batch performs no commit or push. This throughput Batch 1 is distinct from the earlier M1-B4 Batch 1 discovery milestone.

| Environment | Command or procedure | Purpose | Expected Output | Why Expected | Actual Result | Interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local Windows Python | Codex syntax, CLI and synthetic checks of `profile_pipeline.py`. | Validate small default selection, argument rejection, test-patient exclusion, reporting and failure handling. | Syntax valid and synthetic assertions pass. | The checks use constructed inputs and mocked loading/resampling, with actual production normalisation. | Syntax and help checks passed; **11 synthetic/CLI assertions passed**. Production invocation exited 1 with `No module named 'nibabel'`; separate inspection also found SciPy missing. | Local control-flow evidence; a dependency failure was correctly reported without a coverage PASS. | No local real-volume throughput or coverage result; no dependency installation. Full original local commands are not reproduced here. |
| Rangpur verification workspace, CPU | Student transferred `profile_pipeline.py` and executed the targeted command below. | Profile three selected Train/Validation cases before full coverage. | Three successful case records, zero failures, final PASS scoped to selected cases; full-coverage flag false. | Default/targeted mode processes only selected cases through the approved production pipeline. | K019 Week1 Train PASS **4.960915867239237 s**; S035 Week0 Train PASS **5.4963636212050915 s**; B040 Week0 Validation PASS **4.075449131429195 s**. Successful **3**, failed **0**; session wall **15.359112702310085 s**; full-coverage flag **false**. | Three real cases passed, without claiming 181-volume coverage. | Transfer command and complete per-stage targeted transcript were not supplied in this request; no invented transcript. |
| Rangpur Slurm CPU submission | Student attempted a CPU batch submission with `--mem=4G`, then submitted the same job without that option. | Run full coverage on a CPU compute node. | A CPU job created and completed with exit status zero. | A compute allocation permits the sequential full-volume audit without GPU work. | First memory request rejected **before job creation**. Retry created job **647115**, partition **cpu**, node **vcpu-5**, **2 requested CPUs**, **1-hour limit**; final state **COMPLETED**, `ExitCode=0:0`, elapsed **00:14:20**, empty stderr. | Submission correction succeeded; Slurm completion corroborates the module output. | The original submission commands, rejection errors and final `sacct` command were subsequently supplied by the student and are preserved below; Codex did not execute them. No first-job ID is invented. Slurm elapsed is distinct from Python session wall time. |
| Rangpur verification workspace on `vcpu-5`, CPU; explicit existing torch Python | Student executed the full-coverage module command below inside job 647115. | Decode, resample, normalise and validate every Train/Validation pair. | 143 Train volumes/26 patients and 38 Validation volumes/6 patients; 181 successes, zero failures, coverage true and final PASS. | Production discovery enforces frozen coverage, and full mode explicitly selects only Train/Validation arrays. | Train **143/143**, **26 patients**; Validation **38/38**, **6 patients**; total **181/181**, failed **0**; full-coverage flag **true**, final **PASS**; **no test-volume arrays processed**. Timings and RAM below. | Complete preprocessing executability for the 181 Train/Validation volumes. | No model inference, training, Dice/IoU, clinical accuracy or Test-split array-processing validation. |

**Subsequently supplied original CPU Slurm submission evidence:** These commands and terminal results were supplied by the student from the conversation; Codex did not independently execute them. They were unavailable to Codex during the earlier documentation drafting, rather than absent from the historical evidence.

First attempted command:

```bash
sbatch \
  --partition=cpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=4G \
  --time=01:00:00 \
  --job-name=hipmri-coverage \
  --output="$HOME/comp3710_hipmri_verify/profile_181_%j.jsonl" \
  --error="$HOME/comp3710_hipmri_verify/profile_181_%j.err" \
  --wrap='cd "$HOME/comp3710_hipmri_verify" && CUDA_VISIBLE_DEVICES="" "$HOME/miniconda3/envs/torch/bin/python" -B -m recognition.hipmri_3d_improved_unet_hard.profile_pipeline --full-coverage'
```

Actual error:

```text
sbatch: error: Memory specification can not be satisfied
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

The student then executed the same command without `--mem=4G`:

```bash
sbatch \
  --partition=cpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --time=01:00:00 \
  --job-name=hipmri-coverage \
  --output="$HOME/comp3710_hipmri_verify/profile_181_%j.jsonl" \
  --error="$HOME/comp3710_hipmri_verify/profile_181_%j.err" \
  --wrap='cd "$HOME/comp3710_hipmri_verify" && CUDA_VISIBLE_DEVICES="" "$HOME/miniconda3/envs/torch/bin/python" -B -m recognition.hipmri_3d_improved_unet_hard.profile_pipeline --full-coverage'
```

Actual result:

```text
Submitted batch job 647115
```

Final student-executed verification command:

```bash
sacct -j 647115 --format=JobID,State,ExitCode,Elapsed
```

Actual main-job result:

```text
647115 COMPLETED 0:0 00:14:20
```

The stderr file was empty. This evidence confirms submission recovery and final Slurm completion; preprocessing results and their limitations remain recorded separately below. No additional monitoring commands are inferred.

Exact student-executed targeted command:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m \
  recognition.hipmri_3d_improved_unet_hard.profile_pipeline \
  --cases K019_Week1 S035_Week0 B040_Week0
```

Exact student-executed full-coverage Python command, from the Rangpur verification workspace:

```bash
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m \
  recognition.hipmri_3d_improved_unet_hard.profile_pipeline \
  --full-coverage
```

`CUDA_VISIBLE_DEVICES=''` hides GPUs; the profiler itself uses CPU preprocessing and does not import a model or perform CUDA work. `-B` prevents bytecode-cache writes, and `-m` preserves package-relative imports. The explicit interpreter selects the existing torch environment without relying on batch-shell activation. `--cases` limits scope; `--full-coverage` explicitly requests all 181 Train/Validation volumes. Production discovery checks all approved filenames, but no Test arrays are decoded. The script writes reporting to stdout; the remote JSONL/error logs are not added to Git by this documentation update.

**Measured full-coverage output supplied by the student:**

| Quantity | Total seconds | Mean seconds/case |
| --- | ---: | ---: |
| Loading | 265.0306965364143 | 1.4642579919138914 |
| Resampling | 545.37738248799 | 3.0131347098783974 |
| Normalisation | 26.101277890615165 | 0.14420595519676888 |
| Total per-case processing | 856.6517753805965 | 4.732882736909373 |

Session wall: **859.4736277926713 seconds**. Peak RSS: **345210880 bytes**, cumulative whole-process RAM peak, **not GPU VRAM or a per-stage allocation**. Resampling accounts for approximately **63.7%** of total per-case processing time. Total per-case timing includes integrity checks beyond the three production-call timings; session wall additionally includes setup, reporting and cleanup. These are actual audit measurements, not projected full-training runtime. No runtime extrapolation is represented as measured evidence.

**Interpretation and limitations:** This audit extends the earlier one/two-case preprocessing evidence to all 181 Train/Validation volumes. It does not establish clinical annotation accuracy, convergence, segmentation Dice/IoU, full-volume inference correctness, Test-split array-processing success or full-training runtime. Only two real DataLoader patches have been accepted separately. On-demand Patch Dataset access repeats full-volume preprocessing per requested patch. A preprocessing cache is worth evaluating given the measured resampling cost, but no cache is approved or implemented. The frozen split and preprocessing policies remain unchanged.

### Section 5 integration — local evidence verification and subsequent student records

This is a documentation/evidence batch, not a new inference experiment. Original remote command transcripts for the jobs below were not supplied in this batch; their exact commands are not reconstructed. Student-provided execution records and locally inspected downloaded artefacts are distinguished explicitly.

| Environment / command or procedure | Purpose | Expected output / why expected | Actual result | Interpretation / limitations |
| --- | --- | --- | --- | --- |
| Windows PowerShell file inventory and SHA-256 inspection of the specified Downloads directories | Locate and fingerprint original case evidence without copying it into Git | Six additional JSON/PNG pairs and two B040 PNGs, if the reported downloads exist | All 14 files exist; eight PNGs inspected; exact paths/hashes in [manifest](../FAILURE_CASE_EVIDENCE.md) | Direct local existence/content evidence; PNG titles identify case but not model |
| Local Python standard-library JSON audit, executed inline | Check setup model/checkpoint/epoch, case, Validation scope, PASS, geometry/Z agreement and confusion-derived metrics | E10/1,430 updates; identical paired grids/settings; metrics equal count formulas | Six JSONs authenticate three pairs; all per-class Dice/IoU/counts agree within 1e-12; derived precision/recall recorded | Arithmetic audit, not rerun inference; no MRI arrays loaded; B040 JSON/full-validation JSONL not found locally |
| Local source/specification inspection and protected-file SHA-256 comparison | Preserve inference/model policies and verify documentation-only scope | Shared evaluator/metrics; executable/configuration bytes unchanged | Source contracts inspected; 29 Python/JSON files snapshotted for final comparison | Source supports methods; no independent remote source/version verification or checkpoint binary hashing |
| `git diff --check` in the local repository | Detect whitespace errors in tracked diffs | No errors, because edited Markdown uses consistent whitespace | Passed during documentation review; repeated in final checks | Does not inspect untracked report/manifest files; those require separate content/link/whitespace checks |
| Student Rangpur training/Validation records; exact original commands not supplied here | Reconcile stale run status and metric provenance | Completed budgets and full Validation results in authenticated student evidence | 648790 Standard and 649668 Improved: ten epochs/1,430 updates each; 649669 Standard and 650393 Improved: 38 successes, zero failures; Improved reported 8m04s; 649912 CANCELLED | Full pooled values/coverage are student-provided records, original 38-volume JSONL absent locally |
| Student real prediction records and downloaded outputs; exact original invocations not supplied here | Establish driver acceptance beyond synthetic tests | Real MRI/A100 PASS plus paired outputs, without Test processing | 650442 Standard and 650468 Improved B040 PASS; 650490 COMPLETED 0:0, six PNG/six JSON | Student-executed remote evidence; local artefact audit supports last three pairs, B040 model/numbers rely on student records |

The full `overall_pooled` table and case-specific results are retained in the report/manifest. Improved Bones scores were supplied rounded to six decimals; three predicted Bones voxels across two volumes are not useful segmentation or an established exact unrounded zero score. Both models still omit Prostate/Rectum. Resource fields are recorded single-case inference/GPU measurements, not new measurements performed by Codex, clinical validation or full-training speedup. Figure permissions remain unresolved; all original PNGs stay external and captions/insertion placeholders replace image embeds. Final Test evaluation, tutor check-off and submission are not claimed complete.

### Improved V2 - Weighted Cross-Entropy execution record (12 October 2026)

**Provenance:** Remote commands, hashes and results in this entry are **student-supplied Rangpur execution records**. Codex inspected local source/configuration and historical conversation evidence; it did not access Rangpur or independently reproduce these remote results during documentation. Exact commands, reported procedures and reconstructed usage are labelled separately. Scheduler timestamps are reproduced as reported, without assigning an unverified timezone.

**Evidence chronology:** The entry below retains the earlier observation of job **652649 RUNNING** at Step 14 of Epoch 1. Later student-supplied Phase C2/C3 evidence now reports completed training and full Validation; groups 11–14 close the timeline. No new Test evaluation is authorised or verified.

#### V2 command group 1 - Windows transfer and SHA-256 verification

| Field | Record |
| --- | --- |
| Environment | Windows PowerShell; destination is the independent Rangpur verification workspace. |
| Exact command / transcript status | Supplied patterns: `Get-FileHash -Algorithm SHA256` and `scp -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive ...`. Complete executed PowerShell invocation/variable expansion is not supplied here; the ellipsis is a pattern, not an executable historical quotation. |
| Purpose | Transfer the four isolated V2 files and identify their bytes. |
| Expected output | Four readable files, SHA-256 values and completed transfers. |
| Reason for expectation | Hashing identifies contents; SCP reports delivery of accessible files. |
| Actual verified output | `train_improved_v2.py`, `validate_improved_v2.py`, `test_improved_v2.py` and `improved_v2_config.json` each reported `100%` transfer. Remote hashes follow in group 2. |
| Interpretation | Student-reported delivery of the four files. |
| Limitations | Transfer completion alone does not prove code correctness, checkpoint compatibility or GPU execution. No credentials are recorded. |
| Troubleshooting | No V2 transfer failure supplied; the separate missing legacy test was discovered later. |


#### V2 command group 2 - Rangpur source and original-checkpoint authentication

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash, `$HOME/comp3710_hipmri_verify`. |
| Exact command / transcript status | Supplied pattern: `sha256sum recognition/hipmri_3d_improved_unet_hard/<filename>`. Checkpoint existence/size/hash checks were reported; their complete executed loop is unavailable in this update. |
| Purpose | Authenticate transferred files and compare the original E10 snapshot with the mutable last-checkpoint filename. |
| Expected output | Matching V2 hashes and explicit checkpoint existence/hash results. |
| Reason for expectation | SHA-256 compares file bytes independently of filenames. |
| Actual verified output | Both `$HOME/hipmri_training_runs/improved_unet_epoch10.pt` and `improved_unet_last.pt` were present, approximately 66 MB each, with identical SHA-256 `4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6`. |
| Interpretation | The two original remote files were byte-identical at inspection. The authenticated reference is `improved_unet_epoch10.pt`. |
| Limitations | This is reported remote identity evidence, not independent local checkpoint inspection. It does not mean V2 reused the original weights. No separate tensor-by-tensor comparison output was supplied. |
| Troubleshooting | The earlier prepared preflight named `improved_unet_last.pt`; explicit snapshot authentication and byte comparison resolved the ambiguity without weakening protection. |

| V2 file | Reported remote SHA-256 |
| --- | --- |
| `train_improved_v2.py` | `da3258d630e26de47ae418d07a607d1c307ebbf6bca9d5b685e33b668077c9d3` |
| `validate_improved_v2.py` | `f17f98e17f09bd501444a73edef4438c18637eecf3b8642ca640b11c52aaf8fd` |
| `test_improved_v2.py` | `7b4284ad96d01c8c405cbeaa18617cccb7b954ff7cbfab6b5b73f868362307a4` |
| `improved_v2_config.json` | `0af1acdc87b24092914db4de7c751e3623205d31a193ef6fe4f2b5b4fdafb235` |

#### V2 command group 3 - Protected CPU preflight

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash, existing Conda `torch` Python, CPU execution. |
| Exact command / transcript status | Supplied assignments/directory command below. Complete executed Python here-document is unavailable here; its verified procedure is described rather than reconstructed as verbatim. |
| Purpose | Check dependencies, source/test identity and original E10 checkpoint compatibility. |
| Expected output | Matching hashes, compatible imports, E10/1,430 metadata and PASS with exit code 0. |
| Reason for expectation | The protected original loader checks architecture, policies, source compatibility and strict state loading; explicit assertions verify the endpoint. |
| Actual verified output | PyTorch `2.13.0+cu130`, NumPy `2.4.6`, NiBabel `5.4.2`, SciPy `1.17.1`; source manifest `fe8a4ac95f1a8b1e41931c0a1d53498a73ca2a60bca6936e8b8e94c2e6a4553e`; `PASS: protected CPU preflight`; `Preflight pipeline exit code: 0`. |
| Interpretation | The reported procedure imported all four libraries, checked the source manifest/test-file hash, used the protected original Improved checkpoint loader, asserted `completed_epochs == 10` and `total_steps == 1430`, and printed V2 weights/versions. |
| Limitations | Compatibility and identity evidence only; no new training or segmentation-quality evidence. Exact executed checkpoint-path expression in the here-document is not available here; group 2 establishes the intended snapshot and byte identity. |
| Troubleshooting | `mkdir` returned `File exists`. Inspection showed only `preflight.txt`, no V2 checkpoint or job files. This was a directory-reuse warning, not a preflight failure or proof of prior training. No deletion is inferred. |

Supplied commands:

```bash
V2_RUN="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
PY="$HOME/miniconda3/envs/torch/bin/python"
mkdir "$V2_RUN"
```

#### V2 command group 4 - Initial CPU regression failure

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash, CPU-only `torch` interpreter. |
| Exact command / transcript status | The command below is assembled from the complete executed components supplied by the student and matches the previously prepared command; exact auxiliary file-inspection commands are unavailable. |
| Purpose | Run existing regressions and V2 tests together. |
| Expected output | 107 tests ending in `OK`, pipeline exit code 0. |
| Reason for expectation | 92 existing tests plus 15 V2 tests. `CUDA_VISIBLE_DEVICES` hides GPUs; `-B` suppresses bytecode; `-m unittest` loads modules; `-q` reduces verbosity; `2>&1` captures both streams; `tee` saves output; `pipefail` retains upstream failures. |
| Actual verified output | First: `Ran 86 tests in 6.576s`, `FAILED (errors=1)`, `ModuleNotFoundError: No module named 'recognition.hipmri_3d_improved_unet_hard.test_prediction'`. Retry: 86 tests in 3.643 seconds, same missing-module error. |
| Interpretation | FAIL for the combined runs. The reported 86 includes a failed-import entry, not a successful 86-test suite. |
| Limitations | No V2 implementation defect or GPU problem is established by this failure. |
| Troubleshooting | Remote inspection: `predict.py` FOUND, SHA-256 `a3a934f22bd92335c3f2c7f167bd9b1c103b9d56866badf3b0da8ce415718f8c`; `test_prediction.py` MISSING. The remote test workspace was incomplete. |

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES='' "$PY" -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_standard_unet \
  recognition.hipmri_3d_improved_unet_hard.test_improved_unet \
  recognition.hipmri_3d_improved_unet_hard.test_training \
  recognition.hipmri_3d_improved_unet_hard.test_improved_integration \
  recognition.hipmri_3d_improved_unet_hard.test_validation \
  recognition.hipmri_3d_improved_unet_hard.test_prediction \
  recognition.hipmri_3d_improved_unet_hard.test_improved_v2 \
  -q 2>&1 | tee "$V2_RUN/cpu_tests.txt"
```

#### V2 command group 5 - Targeted missing-test transfer and successful retest

| Field | Record |
| --- | --- |
| Environment | Windows PowerShell for hashing/SCP; Rangpur login Bash for hashing/retesting. |
| Exact command / transcript status | Reported procedure: compare local/remote `predict.py` hashes, SCP only `test_prediction.py`, hash it remotely, rerun group 4. Full targeted SCP invocation is unavailable; no production-file transfer is inferred. |
| Purpose | Restore the missing test without changing the verified prediction implementation. |
| Expected output | Completed transfer, matching test hash, all 107 tests passing. |
| Reason for expectation | Resolving the missing module enables its 22 tests in place of the failed-import placeholder. |
| Actual verified output | SCP `100%`, approximately 21 KB; test hash `496aac41902976900a045174724255fa179131f54631348cb3a2322915988c42`; `Ran 107 tests in 9.884s`, `OK`, `CPU regression exit code: 0`. |
| Interpretation | PASS for the student-run CPU regression after targeted workspace repair. |
| Limitations | Synthetic/mocked tests do not establish real-data segmentation quality or allocated-node CUDA compatibility. |
| Troubleshooting | Failure - missing test diagnosis - matching production hash - test-only transfer - successful retest. Old CUDA-driver warnings and expected CLI argument-rejection output were nonfatal, not test failures. |


#### V2 command group 6 - Nineteen-minute GPU scheduling eligibility

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash / Slurm. |
| Exact command / transcript status | Verbatim student-supplied command below. |
| Purpose | Check the short one-step resource request before submission. |
| Expected output | A scheduling estimate, without a real submitted job. |
| Reason for expectation | `--test-only` checks eligibility; 19 minutes is below the reported test QoS 20-minute limit. |
| Actual verified output | Estimated start `2026-10-12T08:29:37` on `a100-a`, displayed test-only number `652631`. |
| Interpretation | Eligible scheduling request at the time of inspection. |
| Limitations | Test-only did not submit a real job; number/start estimate is not execution or a guaranteed future start. |
| Troubleshooting | No rejection supplied; `a100-test` was used for the actual short diagnostic. |

```bash
sbatch --test-only --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:19:00 --job-name=hipmri-v2-smoke --wrap='true'
```

#### V2 command group 7 - Real A100 one-step acceptance

| Field | Record |
| --- | --- |
| Environment | Submitted from Rangpur login Bash; actual execution on allocated A100 node `a100-a`. |
| Exact command / transcript status | Full actual wrapper unavailable. Verified components: `sbatch --parsable`, partition `a100-test`, one A100, 2 CPUs, 19 minutes, module `recognition.hipmri_3d_improved_unet_hard.train_improved_v2`, `--diagnostic-one-step`, separate diagnostic checkpoint target. Exact `sacct`/filesystem-check invocations not supplied here. |
| Purpose | Verify one real MRI forward/loss/backward/update without a saved training checkpoint. |
| Expected output | Completed job, one step, finite gradients, parameter changes, no completed epoch/checkpoint, PASS. |
| Reason for expectation | The diagnostic validates gradients/parameters, stops after one update and bypasses epoch-boundary saving. |
| Actual verified output | Actual job **652632**: `COMPLETED 0:0 00:00:12 a100-a`. JSONL values below; filesystem check: `PASS: No diagnostic checkpoint was saved`. |
| Interpretation | PASS for real-data GPU executability and one update. The counts of 77 are parameter tensors, not scalar parameters. |
| Limitations | No convergence or Validation Dice/IoU improvement established. The 6.180867418996058-second diagnostic runtime and 12-second Slurm elapsed are different measurements. GPU peaks belong to this diagnostic configuration, not measured full-training peaks. |
| Troubleshooting | Nonfatal `max_pool3d_with_indices_backward_cuda` deterministic-algorithms warning under `warn_only=True`; best-effort settings remain enabled, without guaranteed bitwise CUDA determinism. |

| Reported field | Actual value |
| --- | --- |
| `processed_steps` | 1 |
| `epoch_completed` | false |
| `checkpoint_saved` | false |
| `changed_parameter_tensors` | 77 |
| `finite_gradient_parameter_tensors` | 77 |
| `ordinary_ce` | 1.8826422691345215 |
| `weighted_ce` | 2.556217670440674 |
| `soft_dice_loss` | 0.8652969598770142 |
| `loss` | 1.7107571363449097 |
| `runtime_seconds` | 6.180867418996058 |
| `peak_gpu_allocated_bytes` | 463996928 |
| `peak_gpu_reserved_bytes` | 503316480 |
| Final `status` / `total_steps` | PASS / 1 |

Loss components are the production deep-supervision aggregates. Printed values agree with `0.5 * weighted_ce + 0.5 * soft_dice_loss` within float32 rounding. Ordinary CE is logged separately, not added again to the objective.

#### V2 command group 8 - Partition and QoS inspection

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash / read-only Slurm inspection. |
| Exact command / transcript status | Verbatim supplied commands below. |
| Purpose | Inspect full-training resource and wall-time constraints. |
| Expected output | Partition, association and QoS settings. |
| Reason for expectation | These commands query distinct scheduler configuration layers. |
| Actual verified output | `a100`: partition QoS `spot`, DefaultTime `01:00:00`, MaxTime `UNLIMITED`, State `UP`; account `comp3710`, association QOS `normal`; test QoS MaxWall `00:20:00`; `comp3710` QoS MaxWall `12:00:00`. |
| Interpretation | The test QoS is unsuitable for multi-hour training; actual-request eligibility needs checking. |
| Limitations | Account `comp3710` does not alone prove use of the same-named QoS. Partition MaxTime does not override other constraints. These observations are time-specific. |
| Troubleshooting | `spot` motivates interruption/preemption planning and epoch-boundary resume; no actual preemption is reported. |

```bash
scontrol show partition a100
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS
sacctmgr show qos format=Name,MaxWall
```

#### V2 command group 9 - Three-hour training eligibility

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash / Slurm. |
| Exact command / transcript status | Verbatim supplied command below; reported output is an excerpt with an ellipsis. |
| Purpose | Check the three-hour A100 request before real submission. |
| Expected output | Accepted scheduling estimate. |
| Reason for expectation | `--test-only` checks the specified resource/time request without starting training. |
| Actual verified output | `sbatch: Job 652641 to start at 2026-10-12T09:26:03 ... on nodes a100-9 in partition a100`. |
| Interpretation | Eligibility at the time of the check. |
| Limitations | `652641` is not an actual scheduled training run; estimated start is not measured runtime or completion. |
| Troubleshooting | No rejection supplied; real submission followed separate student approval. |

```bash
sbatch --test-only --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=03:00:00 --job-name=hipmri-v2-train --wrap='true'
```

#### V2 command group 10 - Approved full training and early monitoring

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash for submission/monitoring; training on allocated `a100-9`. |
| Exact command / transcript status | Submission below is reconstructed usage from supplied components, NOT a verbatim transcript or instruction to submit again. Monitoring commands are supplied verbatim, with Markdown escapes removed. Exact guard commands and initial `squeue` invocation are unavailable. |
| Purpose | Start the explicitly approved fresh V2 run and inspect progress. |
| Expected output | Real job ID, correct setup, finite steps, and checkpoints after complete epochs. Ten epochs/1,430 updates are planned endpoints. |
| Reason for expectation | The trainer checks the frozen 143-volume policy and saves completed-epoch checkpoints atomically. |
| Actual verified output | `SUCCESS: Submitted V2 training job 652649`; initial `652649 RUNNING a100 a100-9`; later `652649 RUNNING 0:0 00:01:21 a100-9`; at least Step 14 of Epoch 1; `No V2 training checkpoint yet`. |
| Interpretation | Real training was submitted and RUNNING at the last observation. Interim `0:0` is not a final successful exit. |
| Limitations | No proof of final completion, ten completed epochs, 1,430 V2 updates, valid final checkpoint or V2 Validation performance. Finite early losses do not prove convergence. |
| Troubleshooting | Before submission, student checked prior training job ID/checkpoint absence, CPU evidence and diagnostic job ID, and recorded original-checkpoint hashes. No checkpoint before Epoch 1 completes is expected, not a failure. No preemption/restart is inferred. |

**Reconstructed submission usage - not verbatim; do not resubmit:**

```bash
sbatch --parsable \
  --partition=a100 --gres=gpu:a100:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --time=03:00:00 --job-name=hipmri-v2-e10 \
  --output="$V2_RUN/train_%j.jsonl" \
  --error="$V2_RUN/train_%j.err" \
  --wrap='cd "$HOME/comp3710_hipmri_verify" && CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/miniconda3/envs/torch/bin/python" -B -m recognition.hipmri_3d_improved_unet_hard.train_improved_v2 --epochs 10 --checkpoint "$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1/improved_unet_v2_last.pt"'
```

**Supplied monitoring commands:**

```bash
sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
grep -v '"event": "setup"' "$V2_RUN/train_${TRAIN_JOB}.jsonl" | tail -n 8
ls -lh "$V2_RUN"/improved_unet_v2_*.pt
```

Setup reported `diagnostic: false`, `resume_completed_epochs: 0`, experiment `improved_v2_wce_sqrt_v1`, GPU `NVIDIA A100-PCIE-40GB`, 143 steps per epoch, ten planned epochs and frozen configuration/source hashes. First-step loss **1.7107571363449097** matched the diagnostic. This is consistent with controlled fresh initialization; one matching scalar does not prove a complete historical RNG trajectory or bitwise CUDA reproducibility.

| Epoch 1 step | Reported combined loss |
| --- | --- |
| 7 | 1.5556107759475708 |
| 8 | 1.3671952486038208 |
| 9 | 0.9403958916664124 |
| 10 | 1.5255235433578491 |
| 11 | 1.1955740451812744 |
| 12 | 1.033990502357483 |
| 13 | 0.8858786821365356 |
| 14 | 1.1548893451690674 |

**Evidence boundary at this early observation:** final Slurm state/exit/elapsed, complete training records, final-checkpoint verification and full Validation had not yet been supplied. Later evidence in groups 11–14 addresses completion while retaining the log-integrity limitation. Do not infer completion from this early observation, test-only scheduling or one-step acceptance.

### Improved V2 Phase C2/C3 - Completed training and full Validation (student-supplied evidence, 12 October 2026)

These later records extend command group 10; its RUNNING status was accurate at that earlier observation. The V2 JSONL/checkpoint files remain on Rangpur under `$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1/` and were **not** inspected locally during this documentation task. The original Improved E10 Validation JSONL **was** inspected locally for the baseline values below. No Test arrays were accessed.

#### V2 command group 11 - Training completion and log-integrity audit

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash for Slurm inspection and JSONL audit; allocated NVIDIA A100 for training. |
| Exact command / transcript status | Earlier exact monitoring command: `sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList`. The complete final JSONL parsing command/transcript is unavailable locally; the student supplied its resulting values. No complete command is reconstructed as verbatim. |
| Purpose | Determine final job state and planned training-step completion. |
| Expected output | Slurm COMPLETED/0:0 and program PASS at ten epochs/1,430 updates. |
| Reason for expectation | The approved V2 trainer has 143 frozen Train volumes, one step per volume each epoch, and a ten-epoch budget. |
| Actual verified output | Student-reported job **652649 COMPLETED**, ExitCode **0:0**, elapsed **01:55:48**, ten completed epochs, **1,430** total optimiser updates, program final `PASS`. Final epoch mean training loss **0.6459554164976506**; final recorded patch loss **0.4643269181251526**. |
| Interpretation | Slurm and final program/checkpoint records support completion of the planned training budget. Step **143** is the last step *within* Epoch 10; **1,430** is the global update count. These losses are training observations, not Validation metrics or convergence proof. |
| Limitations | The remote `train_652649.jsonl` was reported to contain exactly **126,976 leading NUL bytes**, with earlier records missing from the then-current file. Its cause and missing earlier history remain unresolved. Consequently the complete epoch trajectory cannot be reconstructed from that file alone; neither the raw log nor its hash was repaired or rewritten. |
| Troubleshooting | Retain the earlier RUNNING/Step-14 observation and this later completed state chronologically. The NUL block is a log-integrity defect, not proven training failure or proven resolved. Slurm result, final program result and checkpoint metadata are distinct completion evidence. |

#### V2 command group 12 - Final checkpoint and original-checkpoint preservation

| Field | Record |
| --- | --- |
| Environment | Rangpur login Bash and CPU checkpoint inspection; no training or inference in the integrity check. |
| Exact command / transcript status | Reported procedures: `sha256sum` on final checkpoint and `torch.load(..., map_location="cpu", weights_only=True)`; original-checkpoint manifest verification. The full executed Python/manifest commands are unavailable locally. |
| Purpose | Verify V2 endpoint identity and confirm original Improved checkpoints remained unchanged. |
| Expected output | Loadable V2 checkpoint at ten epochs/1,430 updates and successful original-checkpoint hash checks. |
| Reason for expectation | V2 saves only completed epoch boundaries into its separate run directory; SHA-256 detects byte changes against the recorded manifest. |
| Actual verified output | `improved_unet_v2_last.pt`: **68,960,401 bytes**, SHA-256 `3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f`; safe CPU load PASS; `completed_epochs=10`, `total_steps=1430`. Keys: `completed_epochs`, `config`, `epoch_result`, `model`, `optimizer`, `rng`, `schema_version`, `total_steps`. Original `improved_unet_epoch10.pt: OK` and `improved_unet_last.pt: OK` against the earlier SHA-256 manifest. |
| Interpretation | The final V2 checkpoint has the reported complete-epoch metadata. The two original checkpoint files were unchanged relative to their recorded manifest at verification. |
| Limitations | This is student-supplied remote output. Safe deserialisation and metadata alone do not prove valid model inference or segmentation quality. V2 full-volume execution is separately recorded in groups 13–14. |
| Troubleshooting | V2 and original checkpoint paths remain distinct; the original E10 snapshot and last file were previously confirmed byte-identical, and their preservation checks passed after V2 training. |

#### V2 command group 13 - Full Validation execution and JSONL integrity

| Field | Record |
| --- | --- |
| Environment | Rangpur Slurm `a100` partition, compute node `a100-9`; existing Conda torch interpreter. |
| Exact command / transcript status | Student-supplied evaluation invocation below. The established interpreter is `$HOME/miniconda3/envs/torch/bin/python`; the exact executed `sbatch` wrapper/options for job 652948 are not supplied here, so the wrapper is not reconstructed as verbatim. |
| Purpose | Evaluate the frozen V2 Epoch 10 checkpoint on all 38 Validation volumes with stride 32. |
| Expected output | 38 successful Validation volume records, one setup/summary/result, final PASS and no Test processing. |
| Reason for expectation | `validate_improved_v2.py` selects the frozen Validation split, requires the V2 Epoch 10/1,430-update checkpoint and reuses the original full-volume sliding-window evaluator and metric functions. |
| Actual verified output | Job **652948 COMPLETED 0:0**, start `2026-10-12T11:18:39`, end `2026-10-12T11:26:49`, elapsed **00:08:10**; Validation loop **483.9776779509848 s**. Stderr **0 bytes**. JSONL **95,096 bytes**, **0 NUL bytes**, **41 valid records**, no invalid lines: 1 setup, 38 volume, 1 summary, 1 result. Summary: **38 successful**, **0 failed**, `failed_cases: []`, six patients, `full_38_volume_validation_completed: true`. Final result `PASS`, scope `all 38 Validation volumes`. |
| Interpretation | Student-supplied job and JSONL audit establish a complete Validation-only execution with one successful record per selected volume. This job did not train the model. |
| Limitations | The original V2 JSONL was not downloaded/independently parsed locally in this task; event counts and sizes are reported Rangpur output. This is Validation evidence from six patients, not held-out Test or clinical validation. |
| Troubleshooting | The V2 Validation JSONL had no reported NUL corruption; the separate training-log limitation in group 11 remains unresolved. |

Student-supplied evaluation invocation (environment-specific full Slurm wrapper transcript unavailable):

```bash
python -B -m recognition.hipmri_3d_improved_unet_hard.validate_improved_v2 \
  --checkpoint "$CKPT" \
  --all-validation \
  --stride 32
```

Student-supplied JSONL audit excerpt:

```text
File size: 95096 bytes
NUL bytes: 0
Valid records: 41
Invalid lines: []
setup: 1
volume: 38
summary: 1
result: 1
{"event": "result", "scope": "all 38 Validation volumes", "status": "PASS"}
```

#### V2 command group 14 - Numerical, per-volume and evidence-hash audit

| Field | Record |
| --- | --- |
| Environment | Student-executed read-only Rangpur JSONL/confusion-count audit; arithmetic below independently checked locally from the supplied pooled counts. |
| Exact command / transcript status | Exact original Python audit/here-document is unavailable locally. Reported procedures: validate distinct case/split/status identities; sum per-volume 6x6 confusion matrices; recompute pooled class counts, Dice/IoU, prediction/TP coverage and SHA-256 using `sha256sum`. |
| Purpose | Authenticate the 38-volume comparison and distinguish prediction presence from segmentation quality. |
| Expected output | 38 distinct successful Validation cases, pooled counts equal the summary, and consistent class metrics. |
| Reason for expectation | Overall pooled counts sum intersections, predicted voxels and true voxels before computing each metric; the same evaluation policy was used for original Improved and V2. |
| Actual verified output | Student audit: `PASS: 38 distinct successful Validation volumes`; for each of Bones, Bladder, Rectum and Prostate, ground truth present / predicted / TP-positive in **38/38** volumes; `PASS: All pooled confusion counts match 38 volume records`. Reported V2 pooled and patient-macro values below. Local arithmetic recomputation from the supplied pooled counts matches the four listed organs' Dice and IoU. |
| Interpretation | Weighted CE substantially reduced observed minority-class prediction collapse on this Validation cohort, with positive overlap in every volume for these four classes. Original Improved pooled values below are directly checked against its locally downloaded original Validation JSONL; V2 figures are student-supplied remote records plus independent arithmetic checks. |
| Limitations | A single TP in every volume does **not** imply high per-volume Dice. Pooled, volume-macro and patient-macro measures are different estimands. Six Validation patients, one V2 run and unchanged held-out Test status limit generalisation. The remote V2 JSONL has not been independently parsed locally. |
| Troubleshooting | Training-log NUL corruption remains unresolved. The V2 Validation JSONL and its reported hash preserve a separate clean evidence trail; the training-log fingerprint does not recover missing records. |

Student-supplied per-volume audit output:

```text
PASS: 38 distinct successful Validation volumes
Bones | GT-present volumes: 38 | Predicted volumes: 38 | TP-positive volumes: 38
Bladder | GT-present volumes: 38 | Predicted volumes: 38 | TP-positive volumes: 38
Rectum | GT-present volumes: 38 | Predicted volumes: 38 | TP-positive volumes: 38
Prostate | GT-present volumes: 38 | Predicted volumes: 38 | TP-positive volumes: 38
PASS: All pooled confusion counts match 38 volume records
```

| Class | V2 pooled Dice | V2 pooled IoU | Original Improved pooled Dice, locally inspected |
| --- | ---: | ---: | ---: |
| Background | 0.953976094324474 | 0.9120021914875768 | 0.9760118742343792 |
| Body | 0.862140529343725 | 0.7576862974533001 | 0.9067809959985655 |
| Bones | 0.36228015688892556 | 0.2212100918315336 | 3.943109603265999e-7 |
| Bladder | 0.8267709438369439 | 0.7046969553761536 | 0.7682301265834174 |
| Rectum | 0.4613784294942647 | 0.2998647869876233 | 0.0 |
| Prostate | 0.4456946436520105 | 0.28674844478386075 | 0.0 |

These are `overall_pooled` metrics computed after pooling counts across 38 volumes, not averages of volume or patient Dice. The production absent-class policy is `both_absent_null_and_excluded; one_sided_absence_zero`: both empty gives a null metric omitted from macro averaging; exactly one empty gives zero. Reported V2 patient-macro Dice (equal patient weighting after pooling each patient's Weeks) is **Background 0.9485494166346685; Body 0.8547545927370498; Bones 0.3610534978254727; Bladder 0.7799079456596424; Rectum 0.4395923340101276; Prostate 0.45397654566090667**. Patient-macro figures are not substituted into the pooled comparison.

Pooled precision = intersection / predicted voxels; recall = intersection / true voxels. The following precision/recall and FP/FN were computed locally from the student-supplied V2 counts:

| Class | TP | Predicted | True | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bones | 2795781 | 5290084 | 10144275 | 0.528495 | 0.275602 | 2494303 | 7348494 |
| Bladder | 1513515 | 1919217 | 1742051 | 0.788611 | 0.868812 | 405702 | 228536 |
| Rectum | 187841 | 314195 | 500065 | 0.597848 | 0.375633 | 126354 | 312224 |
| Prostate | 162207 | 476872 | 251012 | 0.340148 | 0.646212 | 314665 | 88805 |

Bones and Rectum retain substantial missed tissue. Prostate has higher recall but many false positives and more predicted than true voxels. Bladder Dice increased, while Background and Body Dice decreased. Positive predictions are not automatically accurate contours. The evidence supports a **trade-off**, not unconditional V2 superiority or clinical reliability.

Student-reported Rangpur SHA-256 fingerprints:

```text
3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f  improved_unet_v2_last.pt
42e61535dbf9034d19fc90a3fedbe379a9cc1198199726f870ffb6551446ea08  train_652649.jsonl
d77e0c53c13a53b8a0af0dd22cf45f6f64798c82a077f597ba9aaa29a80a215e  validation_652948.jsonl
```

The training-log SHA-256 authenticates its **current imperfect bytes**; it does not imply the lost early records were recovered. Original Improved checkpoint preservation was verified separately in group 12. Final model selection is not frozen, no Test evaluation has occurred, and no claim of clinical reliability follows from these Validation metrics.

### Improved V3 continuation - local implementation and proposed remote batch (12 October 2026)

This entry records authorised local work only. No V3 Rangpur job, real MRI step or Validation result has been executed. The exact proposed transfer/preflight/Slurm/acceptance/evidence commands and their expected outcomes are in [IMPROVED_V3_EXPERIMENT.md](../IMPROVED_V3_EXPERIMENT.md); they are not historical executed commands.

| Field | Record |
| --- | --- |
| Environment | Local Windows source inspection and WSL torch environment, CPU; Python 3.11.15, PyTorch 2.12.1+cu130. |
| Command | Final executed nine-module regression command below; separate focused V3 unittest and Python AST/JSON parsing. |
| Purpose | Verify V2-to-V3 state continuation, unchanged objective/inference reuse, selection, checkpoint boundaries, logging safeguards and existing behaviour before requesting GPU budget approval. |
| Expected output | All relevant synthetic tests end in OK; no MRI decoding or GPU execution. |
| Reason for expectation | The V3 runner reuses approved V2 loss/epoch functions, protected V2 loader/resume and original Validation mathematics. New tests exercise its state and evidence handling. |
| Actual output | Focused **19 tests in 8.064s, OK**; combined **126 tests in 10.655s, OK** (107 existing + 19 new). AST/JSON parsing passed. An earlier 18-test revision and 125-test combined run also passed before adding the actual-main restoration test and per-epoch source checks. |
| Interpretation | CPU fixtures verify restored AdamW/RNG/next-update parity, RNG-neutral Validation, class-aware selection guards, immutable epoch files, no diagnostic checkpoint, failed-Validation recovery boundary and journal corruption rejection. The actual main was exercised with mocked remote data/CUDA. |
| Limitations | Local NiBabel is absent; no real NIfTI pipeline or A100 continuation has passed yet. CUDA warn-only policy remains best effort. V2 per-volume logs/checkpoint remain remote and the planned preflight must authenticate their pinned hashes. |
| Troubleshooting | Initial WSL access was sandbox-blocked; approved local execution succeeded after tool escalation. An early interpreter probe had shell-quoting failure and was corrected using stdin. These were environment/command issues, not model test failures. Expected CLI rejection messages appeared during negative tests and did not fail the suite. |

Executed from Windows PowerShell:

```powershell
wsl --cd 'D:\Desktop\UQ26s2-COPMP3710-Pattern-Recognition-and-Analysis\Final project' -e /home/savannah/miniconda3/envs/torch/bin/python -B -m unittest recognition.hipmri_3d_improved_unet_hard.test_dataset recognition.hipmri_3d_improved_unet_hard.test_standard_unet recognition.hipmri_3d_improved_unet_hard.test_improved_unet recognition.hipmri_3d_improved_unet_hard.test_training recognition.hipmri_3d_improved_unet_hard.test_improved_integration recognition.hipmri_3d_improved_unet_hard.test_validation recognition.hipmri_3d_improved_unet_hard.test_prediction recognition.hipmri_3d_improved_unet_hard.test_improved_v2 recognition.hipmri_3d_improved_unet_hard.test_improved_v3 -q
```

The proposed first batch is one 19-minute A100 diagnostic followed, only after acceptance, by a 45-minute allocation for E11-E12 and all 38 Validation volumes. **64 GPU-minutes is a requested ceiling, not an approved or consumed budget.** The approximately 31m20s main-stage projection uses measured V2 training/Validation times and is not a V3 measurement. Default CLI stops at E12; later E14/E16/E18/E20 require separate decisions. No Test use is authorised. V2's historical 126,976 leading NUL bytes remain unresolved; new exclusive append/fsync/hash-chain logging is a safeguard, not a claimed diagnosis of that defect.

## C. AI-Assisted Development Log

This log complements the command reference above. **Repository evidence** means the current files or Git objects inspected while updating this document. **Reported terminal evidence** means command output supplied by the student from an earlier session; it was not rerun for this log. **Codex-reported** means an earlier execution report without an independent reproduction here. **Synthetic** means generated inputs: empty files for Batch 1 discovery tests, genuine small NIfTI files for Batch 2.1 loading tests, constructed 3D arrays for Batch 2.2 resampling tests, synthetic 3D arrays for Batch 2.3 normalisation tests, and constructed arrays with mocked preprocessing for Batch 2.4 Patch Dataset tests. **Real-dataset** means student-run checks on Rangpur HipMRI files; their scope is stated for each batch. A planned action has no PASS result.

### Earlier substantiated milestones — M0 and M1-B1 through M1-B3

| Milestone | Task and available evidence | AI/student roles and evidence limit |
| --- | --- | --- |
| M0 — repository and project scaffold | Git commit `3d22172` added `.gitignore`, the project README, and docstring-only `modules.py`, `dataset.py`, `train.py`, and `predict.py`. The earlier repository commit `552b902` prepared the recognition branch. | The student supplied the project scope and requested a scaffold. The commit proves the files were recorded, not who drafted each line. The exact AI execution log and an independent approval record are not available here. |
| M1-B1 — data integrity audit | The student-provided audit states 38 patients, 211 matched MRI/segmentation volumes, zero image/mask shape, affine, spacing, or orientation header mismatches, LPS orientation, 210 shapes of 256×256×128 and one of 256×256×144, with variable spacing. The current README repeats part of this audit. | **Reported**, not reproduced during this documentation update. Original commands, full output, exact prompts, dates, and AI-versus-student execution detail are **Not Available**. This audit does not establish preprocessing correctness. |
| M1-B2 — semantic label mapping | The student supplied the CSIRO-verified mapping: 0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum, 5 Prostate; the current README records it. | The source-search process and exact original citation are **Not Available** in the local Git history examined here. Record the mapping as student-provided evidence, not as a mapping inferred by AI from image appearance or class sizes. |
| M1-B3 — approved patient split | The student supplied the frozen 26/6/6 patient assignments, 143/38/30 expected volume counts, and seed provenance 3710. The lists and counts are present in `dataset.py` at commit `273cc68`. | The code proves the assignments were stored explicitly and no runtime random draw occurs. The earlier deliberation, exact prompts, consultation record, and original split-generation execution log are **Not Available** here. Real-data coverage was tested later in M1-B4 Batch 1. |

These rows retain the available facts without treating a commit author or a later README statement as proof of unaided authorship, exact prompt wording, or independent rerunning of earlier audits.

### M1-B4 Batch 1 — dataset discovery, frozen split, verification and submission

**Task and motivation.** The course project needed deterministic HipMRI MRI/mask discovery and a leakage-free patient split before image processing or model work. Earlier work had established the dataset audit, CSIRO label meanings, and approved assignments. The student requested AI help to implement the file-level checks, focused tests, and documentation, then sought help when the temporary real-data verifier was missing and Git authentication failed. The student supplied the patient lists, expected counts, dataset path, and implementation boundaries; AI did not select a new random split.

**Prompt record.** The surviving conversation contains student instructions for implementation, real-data verification, temporary-file recovery, README correction, and this audit. The log below paraphrases them to avoid presenting an excerpt as a complete historical prompt. **Prompt Summary — Not Verbatim:** implement only filename discovery, pairing, frozen assignments, patient/volume coverage and synthetic tests; verify the production API on Rangpur without loading arrays or using a GPU; restore missing temporary verification files without changing repository source; correct the README to compare Standard 3D U-Net with 3D Improved U-Net; stop before preprocessing, training, commit or push when those operations were outside the particular batch. The initial implementation prompt permitted edits to `dataset.py`, `test_dataset.py`, and the README but barred preprocessing and training. The recovery prompt permitted only temporary files outside Git and barred SSH/SCP. The student, rather than Codex, performed interactive remote login and transfer. ChatGPT assisted with review and instructions; VS Code Codex was used for reported file edits and repair. Full tool transcripts and exact wording for every intermediate ChatGPT exchange are **Not Available** in this file.

**Stage 1 — production method and source evidence.** The Codex-assisted implementation in commit `273cc68` uses a regular expression in `parse_volume_filename()` to extract patient ID and numeric Week from `*_LFOV.nii.gz` or `*_SEMANTIC.nii.gz`. `_index_files()` indexes paths by `(patient_id, week)` and rejects duplicate logical Weeks, including alternate spellings such as `Week1` and `Week01`. `discover_pairs()` checks the configured root and required directories, compares MRI/mask key sets, and returns sorted `VolumePair` records without reading image arrays. `validate_approved_split()` checks the explicit patient lists for counts, overlap, and K019's Train assignment. `assign_approved_split()` rejects unexpected or missing patients, duplicate pair keys, missing `K019_Week1`, and incorrect per-split or total volume counts; each patient's Weeks use the same lookup. `discover_dataset()` composes pairing and assignment. `test_dataset.py` creates temporary directories and empty files for success and rejection cases. The README records source, labels, verified counts, split policy, and the Standard 3D U-Net versus Improved 3D U-Net question. Git commit `273cc68` contains exactly those three project files. This code inspection establishes the implemented method; it does not itself prove execution on Rangpur.

**Stage 2 — synthetic tests.** Expected: successful deterministic discovery for a complete synthetic filename set and clear rejection of malformed names, missing partners, duplicate Weeks, unknown or absent patients, missing volumes, and missing K019 Week1. Actual: the student-run Rangpur command in Section B reported **11 tests, OK** (**PASS; reported terminal output, synthetic**). The current `test_dataset.py` contains 11 test methods and uses empty `.nii.gz` files; it never validates image bytes, NIfTI headers, voxel alignment, or training behavior.

**Stage 3 — missing verifier recovery.** Initial attempt: the verification handoff referred to Windows TEMP files. Observed problem: the student's `Test-Path -PathType Leaf` checks returned `False` for `verify_real_dataset.py` and `source_review_material.txt`. Diagnosis: the paths had been reported without the files existing there; printing a path string was insufficient. Human correction request: ChatGPT supplied a narrowly scoped recovery prompt instructing Codex to inspect `dataset.py` and `test_dataset.py`, recreate the two TEMP files, call the production `discover_dataset()` API, and avoid repository changes, SSH, commits and pushes. Codex **reported** that syntax checking and a synthetic dry run passed. Retest: the student's `Test-Path` checks returned `True` for both files. Outcome: files existed for manual transfer, but the Codex report alone was not real-data verification. The exact temporary script contents and syntax-check output are **Not Available** in the tracked project history.

**Stage 4 — real Rangpur verification.** ChatGPT provided PowerShell SSH/SCP and Linux commands. The student manually created the home-directory verification workspace, transferred `dataset.py`, `test_dataset.py`, and `verify_real_dataset.py`, logged in with interactive UQ authentication, activated the existing `torch` Conda environment, and ran the commands recorded in Section B. The first SSH attempt returned `Connection reset`; a retry succeeded, and SCP reported `100%` for each file. Expected: 211 matching pairs from 38 patients, 26/143 Train, 6/38 Validation, 6/30 Test, no leakage or missing/unexpected patients, and K019 Week1 in Train. Actual: the student-supplied Rangpur output reported all those values, no missing/unmatched paths, **11/11 synthetic tests passed**, and a final real-data verifier **PASS**. Status: **PASS for real filename discovery, pairing, approved assignment and coverage**. This was lightweight CPU-only path inspection; it did not load NIfTI arrays, test voxel-level preprocessing, use a GPU, or measure segmentation quality. The local conversation record reports the output; the remote terminal was not independently rerun while writing this log.

**Stage 5 — source review.** The student supplied `source_review_material.txt` for a ChatGPT-assisted review of production matching, frozen assignments, leakage safeguards, synthetic tests, the read-only verifier, and README consistency. The review identified an outdated 2D U-Net baseline reference and outdated real-verification status. This was **source review**, not another independently executed dataset test. The review material itself was a temporary file outside Git; its complete historical contents are **Not Available** in this repository.

**Stage 6 — documentation correction.** The student directed Codex to edit only the project README. Codex **reported** revising the engineering question to Standard 3D U-Net (Normal Difficulty) versus 3D Improved U-Net (Hard Difficulty), recording the Rangpur discovery checks, and separating them from unfinished loading, preprocessing, training, and performance evaluation. The current README and commit `273cc68` substantiate those content changes. The student manually inspected the README and Git output. Several joined words were noticed later; whether each was corrected is **Unverified** from the available review record, so this log does not claim that formatting repair passed.

**Stage 7 — Git and authentication iteration.** Initial problems reported by the student were `Not a git repository` when Git ran in the Rangpur TEMP workspace, `Author identity unknown` during a local commit attempt, a GitHub SSH private-key passphrase prompt, and Windows LF/CRLF warnings. ChatGPT explained the distinction between the Rangpur workspace and local checkout, the need for Git author identity, and GitHub authentication; it recommended HTTPS with Git Credential Manager. The student checked GCM, changed `origin` to HTTPS, configured the local helper, and performed the push. Section B records the verified commands and outcomes. The final Git evidence is commit `273cc68` (`Implement HipMRI discovery and frozen patient split`) on `topic-recognition`, containing README, `dataset.py`, and `test_dataset.py`; the reported HTTPS push advanced `Savan-na/PatternAnalysis-2026` from `3d22172` to `273cc68`. A later `git commit` returned `nothing to commit`; it did not create `273cc68`. The working tree and tracking branch were clean and synchronised after that push. LF/CRLF warnings were not test failures. Commit author metadata identifies Git attribution, not who drafted the code.

**Stage 8 — review boundary and disclosure.** M1-B4 Batch 1 was technically reviewed and approved, committed and pushed. AI assistance materially contributed the production discovery/split implementation, synthetic tests, temporary verifier recovery, README wording, source review, and debugging recommendations. The student established the approved project facts and constraints, detected missing TEMP files and documentation issues, executed the interactive Rangpur and Git verification, supplied outputs, selected the GitHub authentication change, reviewed the work, and approved the batch. The confirmed result is a validated file-level dataset index and split. Production NIfTI loading, preprocessing, model architecture implementation, training, and Dice/IoU evaluation remain **PENDING**. The next planned batch is M1-B4 Batch 2 — 3D Preprocessing & Spatial Alignment; no commands or results for it are claimed here.

### M1-B4 Batch 2.1 — NIfTI content loading and spatial validation

**Task, prior state and student request.** After Batch 1's approved 211-pair filename and patient-split validation, the student requested one-pair, CPU-compatible 3D NIfTI decoding and spatial/content checks. The student specified the `VolumePair` interface, labels 0–5, original-grid preservation, synthetic NIfTI tests, and strict exclusions of preprocessing, models, training, automatic Rangpur access and repository commits. AI assistance was requested for implementation, diagnosis of a real-data failure, targeted correction and source review. **Prompt Summary — Not Verbatim:** implement only `volume_io.py` and `test_volume_io.py`; read one MRI/mask pair, validate shape, affine, spacing, units, labels and finite values; preserve the frozen split; after observing the `mm`/`unknown` failure, accept that pattern only when all geometric checks pass and expose the inference. Complete original prompt transcripts for every ChatGPT exchange are not reproduced here.

**Initial implementation.** ChatGPT helped form the narrow loading plan. Codex created `volume_io.py` and `test_volume_io.py`, using NiBabel for real NIfTI decoding, a `LoadedVolumePair` result, scaled finite `float32` MRI data, validated `uint8` segmentation IDs, and original shape/affine/spacing/orientation metadata. It reused the approved `VolumePair` and did not alter `dataset.py`. The initial rule required MRI and mask headers to declare the same spatial unit. Local Codex testing could parse the new files and run the 11 discovery tests, but local NiBabel was unavailable; this was explicitly reported rather than counted as a NIfTI test pass.

**Synthetic verification.** The student manually transferred the loader and tests, activated Rangpur's `torch` environment, and ran the combined CPU-only command recorded in Section B. The student reported NiBabel `5.4.2`, NumPy `2.4.6`, and **27/27 PASS**: 11 existing discovery tests and 16 generated-NIfTI tests. Those tests exercised decoding and failure cases on small files created by the test suite. They did not establish that a real HipMRI pair would pass.

**Real-data failure.** The student then attempted real `K019_Week1` loading. The observed result was `ValueError: K019_Week1: spatial unit mismatch: MRI='mm', segmentation='unknown'.` This **FAIL** was retained as evidence of a real metadata assumption that the synthetic tests had not covered. It did not demonstrate that the voxel grids differed.

**Student-led header audit.** To diagnose the failure, the student—not Codex—ran a read-only audit of **all 211 real pair headers** on Rangpur. The student reported MRI=`mm` and mask=`unknown` in **211/211** pairs, LPS/LPS orientation throughout, zero shape/affine/spacing/orientation grid mismatches, zero unexpected unit combinations, and audit `PASS`. For `K019_Week1`, the effective affines were numerically identical despite differing sform/qform codes. This supported a conditional metadata interpretation; header agreement did not prove full array-content validity or clinical annotation quality. The exact original audit here-document is unavailable.

**Diagnosis and targeted correction.** ChatGPT reviewed the student-supplied failure and audit and specified the compatibility condition. Codex changed only `volume_io.py` and `test_volume_io.py`: matching explicit physical units remain accepted; MRI=`mm` with mask=`unknown` is accepted only if 3D shapes, finite valid affines and spacing, affine/spacing tolerances, and orientation agree. Explicitly conflicting units and an unknown MRI unit remain errors. `spatial_unit` retains the MRI-declared working unit, `segmentation_header_spatial_unit` retains the mask's actual header value, and `segmentation_unit_inferred` flags the conditional inference. Six genuine-NIfTI test methods were added without weakening prior mismatch or invalid-label checks. No source NIfTI was altered or resampled.

**Regression and real-data success.** The student manually reran the combined Rangpur suite with GPU visibility disabled: `Ran 33 tests in 0.226s`, then `OK` (**11/11 discovery plus 22/22 NIfTI synthetic tests**). The student then reran the one-pair real loader. MRI and mask both decoded to `(256, 256, 144)`; MRI dtype was `float32`, segmentation dtype `uint8`, working unit `mm`, original mask header unit `unknown`, inference flag `True`, axis codes `('L', 'P', 'S')`, and labels present `(0, 1, 2, 3, 4, 5)`. The reported final line was `PASS: K019 Week1 real NIfTI pair loaded and validated`. This is **real content-loading evidence for one pair** and synthetic regression evidence for the remaining code paths. It is not a full 211-pair array audit.

**Source review and disclosure.** The student supplied both source files for ChatGPT-assisted review, which reported no blocking issue within Batch 2.1's scope. AI materially assisted the loading design, authored the loader and synthetic tests through Codex, diagnosed the unit-policy problem from student evidence, implemented the correction, and assisted source review and documentation. The student set the scope and approval boundaries, performed interactive transfer, tests, the all-pair header audit and the real-pair retest, supplied the outputs, and approved the technical milestone. Git author metadata is not evidence of unaided coding. No Batch 2.1 commit is recorded here. Full 211-pair array-content verification, preprocessing, training and Dice/IoU evaluation remain **PENDING**.

### M1-B4 Batch 2.2 — 3D resampling and spatial alignment

**Task, motivation and prompt record.** Following approved one-pair NIfTI loading, the project needed an aligned physical grid before either 3D model arm could use the data. The student requested AI help designing and implementing one-pair CPU resampling, physical/label checks, a memory estimate and synthetic tests, without changing the frozen split, loader, original data, model code or training state. The student selected the training-derived spacing candidate and 512 MiB estimate guard after a training-only header and memory preflight. **Prompt Summary — Not Verbatim:** implement `make_target_grid()` and `resample_volume_pair()` in `resampling.py`, add `test_resampling.py` and `resampling_config.json`, keep MRI and mask on one affine, use linear/nearest interpolation, reject lost foreground classes, and wait for human-run Rangpur testing. The full original prompts are available in the project conversation but are not reproduced verbatim in this record; the summary must not be treated as a quotation.

**Initial AI implementation and local checks.** Codex inspected the existing `VolumePair` and `LoadedVolumePair` contracts, then created the three Batch 2.2 files. `make_target_grid()` projects the eight source voxel-edge corners onto axes derived from the source affine, rounds near-integer extents and forms one output shape and affine without reorienting LPS data; singular or sheared geometry is rejected. `resample_volume_pair()` computes each array's output-to-input affine transform, samples MRI at order 1 and mask at order 0 on that shared grid, records spatial-unit provenance and class counts/physical volumes, and rejects invalid or disappeared foreground labels. It estimates four times source-plus-output array bytes before allocating the outputs; this is a planning guard, not a peak-RSS guarantee. Codex wrote 16 synthetic tests. Local Windows lacked SciPy and NiBabel; WSL had SciPy but lacked NiBabel, so Codex ran the genuine synthetic resampling suite in WSL using an in-memory shim solely for the unused NiBabel import. **16/16** resampling and **11/11** existing dataset tests passed locally; no real NIfTI was decoded there.

**Student preflight and first remote review.** Before implementation, the student manually ran the training-only Rangpur spacing/memory audit, installed and checked SciPy in the `torch` environment, reviewed the candidate target spacing, and approved it as an initial value. The reported 143-volume planning estimates appear in Section B; no measured RSS was claimed at that stage. The student transferred the files to Rangpur and ran the initial combined suite: **49/49 PASS** (33 existing plus 16 new tests). The student also ran real K019 Week1 and S035 Week0 examples successfully. This established initial execution on two real cases, but the source review later found an untested boundary behavior. Exact original transfer, installation and real-case commands are **Not Available** here.

**Source review → reproduced defect → targeted correction.** Review noticed that `mode="constant", cval=0` can erase a face-touching mask label when an output center lies outside the source center range but inside the voxel-edge field of view. Codex reproduced the issue with a genuine synthetic 3D MRI/mask pair and SciPy: `constant` erased both tested edge faces; `grid-constant` retained them. Linear MRI sampling with `grid-constant` attenuated an edge intensity of 100 to about 95, while `nearest` retained 100. At the student's request, Codex changed only the three Batch 2.2 files: mask mode became `grid-constant` with background 0, MRI mode became `nearest`, and the JSON boundary declarations were read and validated rather than silently ignored. The grid equations, target spacing, memory cap, interpolation orders, source data and split remained unchanged. Synthetic fixture axis codes were corrected to match their artificial affines. Four tests were added, including all-six-face label preservation and background sampling beyond the source edge. The revised local resampling suite passed **20/20**; the 11 dataset tests also passed. These local tests used no real HipMRI arrays.

**Student retest, result and disclosure.** The student transferred the corrected files, reran the Rangpur combined suite (**53/53 PASS**), and reran K019 Week1 and S035 Week0 after the fix. The student supplied the final shapes, prostate voxel counts, measured peak RSS and zero exit statuses in Section B, inspected the results, requested final review and approved the technical batch. Codex did not access Rangpur or independently reproduce those remote outputs. AI materially proposed and authored the resampling code, configuration, synthetic tests, boundary correction, source-review explanation and this documentation. The student supplied requirements and training-only evidence, selected and reviewed the target spacing, installed the remote dependency, performed transfers and remote verification, examined failures and results, and approved the outcome. The available evidence covers synthetic regressions and **two** real acceptance examples. Complete 211-volume resampling, clinical validation, intensity normalisation, mask encoding, model training and Dice/IoU evaluation remain **PENDING**. No Batch 2.2 commit or push is claimed in this record.

### M1-B4 Batch 2.3 — MRI intensity normalisation

**Task and prior evidence.** After approved patient-level discovery, one-pair NIfTI loading and aligned 3D resampling, the project needed an MRI-only intensity transform before a PyTorch segmentation Dataset. The student ran an initial audit on resampled training cases K019 Week1 and S035 Week0; their zero fractions, nonzero distributions and percentiles are recorded in Section B. These were **input** measurements, not normalised output results. The student set the scope: one pair on CPU, no segmentation-derived intensity mask, no global fit on validation/test data, and no model or training work. ChatGPT assisted in selecting nonzero percentile clipping followed by nonzero Z-scoring as an initial engineering policy; neither the course nor clinical evidence was claimed to mandate that formula. **Prompt Summary — Not Verbatim:** create only `intensity_normalization.py`, `test_intensity_normalization.py` and `normalization_config.json`; use original MRI nonzeros, 0.5/99.5 clipping, float64 population statistics, finite float32 output, original-zero preservation and unchanged geometry/mask; reject degenerate inputs; stop before training or documentation. The complete original implementation prompt is in the project conversation and is not reproduced as a historical quotation here.

**Codex implementation and local evidence.** Codex created the three files and added `normalize_volume_pair()` for a validated `ResampledVolumePair`. It selects voxels from the original MRI alone, clips a float64 working vector, computes `ddof=0` mean/std, writes a separate float32 MRI and restores exact original zero locations. The result carries pair identity, copied affine, the unchanged segmentation by reference, spatial/label provenance and reproducibility statistics. The module reads and validates the frozen JSON policy. It rejects all-zero, non-finite, invalid-shape/dtype, constant and numerically degenerate cases; synthetic tests cover these and confirm that changing segmentation labels cannot change MRI statistics. Codex reported **18/18 PASS** locally in WSL using real NumPy calculations and an in-memory NiBabel import shim for the existing resampling type. That test did not decode real NIfTI data.

**Human transfer and remote verification.** The student manually copied the three new files with SCP, logged into Rangpur interactively, selected the `torch` environment and ran the exact combined command in Section B. The student supplied `Ran 71 tests in 0.214s` followed by `OK`, then ran the real-case Bash loop for K019 Week1 and S035 Week0 with `/usr/bin/time -v`. The Python verification invoked production discovery, loading, resampling and normalisation and checked finite float32 output, original-zero preservation, unchanged mask/affine and output selected mean/std within `1e-4` of 0/1. Section B contains the actual reported per-case values, measured whole-process RSS, wall times and exit statuses. Codex did not independently access Rangpur or reproduce these real-data outputs.

**Review, limits and disclosure.** ChatGPT-assisted final source review found no blocking issue within the Batch 2.3 scope, and the student requested this documentation update after technical approval. AI materially contributed algorithm planning, code generation, synthetic tests, source review and documentation. The student established priorities and constraints, supplied the initial real-data evidence, executed Rangpur transfer/tests/real-case checks, reviewed outputs and approved the technical decisions. Git authorship alone would not separate those contributions. The 71 tests and two real examples support this one-pair normalisation pipeline; they do **not** establish success on all 211 volumes, clinical contour quality, model training or Dice/IoU performance. No Batch 2.3 commit or push is claimed here.

### M1-B4 Batch 2.4 — minimal PyTorch 3D Patch Dataset

**Scope and AI instructions.** The student approved the smallest Patch Dataset needed to prepare a future Standard 3D U-Net smoke test. They required reuse of the frozen patient-disjoint split and established discovery → loading → resampling → normalisation pipeline, CPU-only on-demand processing, `(64,64,64)` patches, correct `(X,Y,Z)` to `(Z,Y,X)` tensor-axis mapping, aligned MRI/mask crops, reproducible foreground-aware training selection, deterministic validation and no model or training code. ChatGPT assisted with the implementation specification. **Prompt Summary — Not Verbatim:** create only `patch_dataset.py`, `test_patch_dataset.py`, and `patch_dataset_config.json`; use existing APIs rather than duplicate preprocessing; test synthetic alignment, shapes, masks, sampling, split isolation and one-batch DataLoader output; stop before Rangpur access, training, documentation or Git submission. The full original implementation prompt exists in the project conversation and is not presented here as a reconstructed quotation.

**Codex implementation and synthetic evidence.** Codex created those three files. `HipMRIPatchDataset` accepts only `train`/`validation`, takes their `VolumePair` records from `discover_dataset()`, and invokes `load_volume_pair()`, `resample_volume_pair()` and `normalize_volume_pair()` in `__getitem__`; it holds no processed volume cache. It transposes both arrays from `(X,Y,Z)` to `(D,H,W)=(Z,Y,X)`, applies the same crop slices, returns float32 MRI `[1,D,H,W]` and integer `torch.long` mask `[D,H,W]`, and records patient, Week, origin and sampling mode. Training uses seed/epoch/index to draw random or prostate-containing patches, with a fallback when class 5 is absent; validation uses a center crop. The config records patch shape, one training patch per volume, probability `0.5`, label `5`, seed `3710`, and center validation. Codex reported **16/16 PASS** in WSL using constructed arrays and mocked preprocessing with an in-memory NiBabel import shim. This is synthetic evidence, not a real HipMRI result.

**Student verification and review sequence.** The student manually transferred the three files with SCP, logged into Rangpur, activated `torch`, and ran the exact five-module command in Section B. The student supplied `Ran 87 tests in 8.646s` and `OK`. ChatGPT then assisted with source review; Batch 2.4 passed that review. The student next ran the timed CPU-only real-data DataLoader check with `Subset`, `batch_size=1`, and `num_workers=0`. They inspected and supplied the K019 Week1 training and B040 Week0 validation batches, including tensor shapes/dtypes, valid labels, crop origins and modes, with actual prostate label 5 in the training patch. Both passed; the final reported output and whole-process measurements appear in Section B. This sequence reflects student-executed Rangpur commands; Codex did not independently log in or rerun the cases.

**Outcome, limits and disclosure.** The student reviewed the results and approved the technical decisions, then requested this Codex documentation update. AI materially generated the Patch Dataset and synthetic tests, assisted the specification and source review, and drafted this record. The student set the scope, performed remote transfer and verification, checked the real outputs and approved the batch. The 87-test regression and two real patches support a first on-demand patch pipeline, not full 211-volume coverage. Validation center crops may miss the prostate elsewhere, and repeated full-volume preprocessing per patch may be a training bottleneck. Full-volume inference, architecture training, clinical accuracy and Dice/IoU evaluation remain **PENDING**. No Batch 2.4 commit or push is claimed here.

### Standard 3D U-Net — architecture, CPU regression and real GPU smoke test

**Task and prompts.** With the approved one-pair preprocessing and Patch Dataset pipeline complete, the student requested a conventional Normal Difficulty Standard 3D U-Net and a minimal first real GPU optimisation check. ChatGPT assisted planning, code review and interpretation. **Prompt Summary — Not Verbatim:** preserve existing meaningful `modules.py` content; use four downsampling stages, paired padded Conv3d layers, GroupNorm/ReLU, transposed-convolution upsampling and concatenated skips; return six raw logits; add focused CPU tests. Then create only `smoke_standard_unet.py`, reuse the approved APIs and K019 Week1 frozen training assignment, preprocess on CPU, require CUDA and execute exactly one float32 forward/loss/backward/SGD update. Exclude full training, model changes, validation loops, schedulers, checkpointing, automatic Rangpur access and Git submission. Complete original user instructions exist in the project conversation; this summary is not a quotation.

**AI implementation and local verification.** Codex assisted model implementation and reviewed the Standard 3D U-Net implementation already present as an uncommitted `modules.py` change when inspected. It preserved that implementation, created `test_standard_unet.py`, and reported **10/10 local synthetic CPU tests passed**, including cross-entropy/backward/parameter-update checks and a separate no-gradient 64³ shape check with reduced feature width. The default model has widths 16/32/64/128, bottleneck width 256, four decoder stages, GroupNorm/ReLU and a six-channel raw-logit head; its measured trainable scalar parameter count is **5,646,470**. Codex then created `smoke_standard_unet.py` with CUDA diagnostics, one-pair CPU preprocessing, seed 3710, finite-value/gradient checks, host parameter snapshots to verify an update and GPU memory reporting. Local syntax, whitespace and CLI-help checks passed; Codex did not execute this real GPU test locally. The student directed requirements and later supplied remote execution evidence.

**Student CPU regression and allocation troubleshooting.** The student manually transferred `modules.py` and `test_standard_unet.py` and ran the six-module regression on Rangpur: **97/97 PASS**, `Ran 97 tests in 10.212s`, then `OK`. The login-node CPU run emitted a nonfatal CUDA driver warning. ChatGPT-assisted review distinguished that warning from allocated-node GPU compatibility, which remained unverified at that point. The first interactive `srun` request used partition `comp3710` and queued as job **647036**; the student cancelled it using Ctrl+C and later executed `scancel 647036`. The second request used the same resource settings with partition `a100` and allocated Slurm job **647039** on node **a100-5**, with an **NVIDIA A100-PCIE-40GB**. The original request and cancellation command supplied by the student are preserved in Section B. The student verified PyTorch **2.13.0+cu130**, CUDA build **13.0**, driver **590.48.01**, CUDA availability and successful CUDA tensor creation in the allocated session. No dependency-upgrade action is inferred from these version observations.

**Student real GPU execution and result.** The student manually ran the exact module command in Section B. It selected `K019_Week1`, prostate crop origin `[40, 60, 64]`, labels `[0, 1, 2, 3, 5]`, MRI batch `(1, 1, 64, 64, 64)` and integer target `(1, 64, 64, 64)`. Exactly one float32 optimisation step passed: initial cross entropy **1.92833924**, all **64** trainable parameter tensors with finite gradients, and **59/64** tensors changed after SGD at **0.001**. The final PASS and allocated/reserved/peak GPU memory values are preserved in Section B. The student inspected and supplied these outputs for ChatGPT-assisted interpretation and requested this documentation update. Codex did not independently access Rangpur or rerun the reported GPU result.

**Disclosure and limits.** AI materially assisted architecture planning and implementation, authored the architecture tests and one-step smoke-test script, assisted source review and result interpretation, and drafted these documentation additions. The student specified the scope, manually transferred files, requested/cancelled GPU allocations, executed Rangpur CPU and GPU checks, inspected results and supplied the evidence. One real training patch establishes executability of this configuration; the initial loss and finite gradients do not establish convergence, segmentation quality or clinical accuracy. No full epoch, validation-performance measurement, Dice/IoU evaluation or checkpointing occurred. At this historical documentation stage, the architecture/script changes were pending Git submission. They were subsequently committed and pushed in `dc20457`: `Implement and verify Standard 3D U-Net baseline`. The current Batch 1 profiler and documentation remain uncommitted; this documentation update performs no commit or push.

### Batch 1 — preprocessing throughput and full Train/Validation coverage

**Task and motivation.** After the Standard 3D U-Net execution milestone, the student requested a bounded throughput and real-volume coverage audit before building full training. The technical question was whether the approved on-demand preprocessing pipeline executes reliably across Train/Validation and how much CPU time it consumes. Model convergence and segmentation quality were outside scope. Baseline commit: `dc20457`; no new commit is claimed for this batch.

**Instructions and roles. Prompt Summary — Not Verbatim:** create only `profile_pipeline.py`; inspect and reuse production discovery, NIfTI loading, resampling, normalisation and approved configurations; provide a small targeted mode and explicit full Train/Validation coverage mode; measure stages, validate geometry and labels, report failures and cumulative process RSS; exclude Test-array processing, GPU work, automatic remote access, caches, training, dependency installation and Git submission. ChatGPT assisted scope definition, source review and interpretation. Codex implemented the profiler and its local checks; the student defined constraints and executed all real Rangpur measurements.

**Implementation and local evidence.** Codex inspected the production APIs and configurations and implemented sequential `discover_dataset()` → `load_volume_pair()` → `resample_volume_pair()` → `normalize_volume_pair()` use, without duplicating preprocessing algorithms. The profiler uses `time.perf_counter()` for stage timings, returns scalar case records rather than retained full-volume arrays, releases current-case objects between cases, records exception details and exits non-zero on required failures. It checks finite MRI values, matching geometry, mask IDs/counts and foreground survival. Default selection is K019 Week1, S035 Week0 and B040 Week0; all 181 volumes require `--full-coverage`. RSS is explicitly cumulative process-level memory. Codex reported syntax/help checks and **11 local synthetic/CLI assertions passed**, using mocked loading/resampling and actual normalisation. Missing NiBabel and SciPy blocked local production execution; this was disclosed, not recorded as a real-data PASS. These local checks do not establish Rangpur throughput.

**Student targeted execution.** The student manually transferred the script and ran the exact targeted command recorded in Section B. All three real cases passed: K019 Week1 **4.960915867239237 s**, S035 Week0 **5.4963636212050915 s**, B040 Week0 **4.075449131429195 s**; **3 successes**, **0 failures**, session wall **15.359112702310085 s**. The false full-coverage flag correctly limited the claim to selected cases. The student supplied outputs for ChatGPT-assisted review; Codex did not independently rerun them.

**Failure → correction → full-coverage verification.** The student's first CPU Slurm submission requested `--mem=4G`; Slurm rejected that request before creating a job. The student removed the option and successfully submitted the same CPU job as **647115**, on partition **cpu**, node **vcpu-5**, requesting **2 CPUs** and a **1-hour** limit. The original submission commands and rejection transcript were subsequently supplied by the student and are preserved verbatim in Section B, together with the final `sacct` command; they were not independently executed by Codex. The student executed the full-coverage Python command in Section B, monitored the output and checked Slurm completion: **COMPLETED**, `ExitCode=0:0`, elapsed **00:14:20**, empty stderr. The real audit reported **143/143 training volumes, 26 patients**, **38/38 validation volumes, 6 patients**, **181/181 successes**, **0 failures**, full-coverage flag **true**, final **PASS**, and no Test-array processing.

**Results, interpretation and decision boundary.** The actual stage totals, means, session wall and **345210880-byte** process peak RSS are preserved in Section B and the README. Resampling consumed approximately **63.7%** of total per-case processing time, identifying a measured runtime concern for repeated on-demand volume preprocessing. ChatGPT assisted interpretation; a cache is worth evaluating, but remains an unapproved future option, not an implemented optimisation. This demonstrates complete preprocessing executability for Train/Validation only. It does not demonstrate clinical annotation accuracy, model convergence, Dice/IoU, full-volume inference correctness, Test-array success or actual training runtime. No extra experiment or runtime projection is claimed.

**Disclosure and documentation.** Codex generated the profiler, performed local synthetic/CLI checks and prepared these documentation updates. ChatGPT assisted scope definition, source review and output interpretation. The student manually transferred the script, ran the targeted audit, submitted and monitored the CPU job, verified final Slurm completion, supplied real outputs and requested documentation finalisation. Real Rangpur evidence is student-supplied terminal evidence, not independently executed by Codex. Historical milestones are preserved; no preprocessing cache, source change, additional measurement, training, commit or push is performed by this documentation batch.

### Section 5 — documentation integration, paired failure cases and evidence reconciliation

**Task, motivation and prompt.** The student requested documentation-only integration of the completed E10 comparison and representative failures against the official specification. **Prompt Summary — Not Verbatim:** inspect report/navigation/task README/traceability/AI history, original failure narrative, specification, relevant source and downloaded JSON/PNG evidence; reconcile completed/cancelled jobs, authenticate paired identities and Z slices, verify numbers and describe actual panels; preserve code/checkpoints/splits, keep images external unless permission is established, do not run remote jobs/Test inference, commit or push. The full current instruction is available in conversation; this condensed entry is not an exact quotation.

**Student and previous AI work.** The student supplied the completed training/Validation/prediction records, class coverage and B040 counts, identified the downloaded artefact locations, and set the evidence/permission boundaries. The student executed the Rangpur experiments; prior ChatGPT/Codex work supplied planning, source review, implementation and interpretation as recorded in the project conversation. No additional remote execution or model development was requested here.

**Codex actions and method.** Codex inspected current Markdown, model/checkpoint/inference/metric contracts and the local specification. It read all six additional JSONs, checked model/case/checkpoint E10/1,430-step identities, Validation/PASS status, shared affine/shape/policies/truth counts and Z indices, and recomputed per-class overlap from confusion matrices. It visually inspected all eight unmodified PNGs, authored Section 5 and a metadata-only evidence manifest, reconciled task/navigation/traceability status, and preserved the original B040 narrative. It inspected available image rules and left PNGs outside Git because explicit HipMRI public-redistribution permission was not established. No source, config, image or checkpoint was edited.

**Problem → correction → verification.** Stale documentation treated Improved 649912 as PENDING, real driver acceptance as missing, and failure figures/comparison as unavailable. Supplied records establish that 649912 was cancelled and 650393 completed; 650442/650468 passed and 650490 generated paired evidence. Codex corrected current status while retaining historical limitations. Six local JSONs verify case arithmetic and metadata; the complete 38-volume results and B040 numerical values remain student-record evidence because original JSONL/B040 JSONs were not found locally. Both B040 supplied prostate rows sum to 7,105. The report retains S022's genuine regression and avoids asserting causal or clinical success. Markdown/diff/link checks and before/after protected-file hashes provide documentation verification, not another model test.

**Outcome and disclosure.** AI materially drafted the integrated report, captions, manifest and documentation corrections and performed local source/artefact/arithmetic review. The student directed the task, performed remote experiments and supplied evidence; AI did not independently reproduce those experiments or authorise image redistribution. No new Dice/IoU experiment, Test result, tutor approval, submission or commit is invented. Existing earlier records are preserved. The local Git HEAD at this review remains `ed705bdc4ef72db282e9c53e599f150e34a303e3`; it does not identify the remote E10 source revision. Outstanding evidence includes full-run original logs, B040 JSONs, figure permission, anatomical boundary analysis, matched repeated resource profiling and remaining assessment deliverables.

### Improved V2 - controlled weighted-CE optimisation (12 October 2026)

**Task, motivation and approval.** Original Improved E10 completed ten epochs/1,430 updates but largely omitted Bones, Rectum and Prostate on full-volume Validation. Prior source/loss, Train patch exposure and selected-patch probability investigations found no demonstrated implementation defect or unique causal explanation. The student requested a bounded practical optimisation. ChatGPT assisted hypothesis ranking and proposed a single-variable weighted-CE experiment. The student approved isolated implementation, then Phase C1 CPU/GPU acceptance, then explicitly approved Phase C2 full training. These are distinct approvals; no new Test evaluation was authorised.

**Prompt Summary - Not Verbatim.** Preserve original experiments/checkpoints; introduce six fixed Train-derived inverse-square-root-frequency weights with a voxel-count CE denominator; retain Dice, architecture, deep supervision, data and training policies; isolate V2; test numerical/gradient correctness, actual initialization/RNG parity, strict checkpoint provenance and resume; stop before remote execution until approval. The subsequent documentation request permits only `ai.md` changes. Missing prompts are not reconstructed as exact quotations.

**AI implementation and method.** Codex created `train_improved_v2.py`, `validate_improved_v2.py`, `test_improved_v2.py` and `improved_v2_config.json`. Six labels remain 0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum, 5 Prostate. Train-only exposure counts `[69380520, 246261463, 34110362, 15115893, 4618511, 5379171]` cover 1,430 reconstructed patches; source evidence SHA-256 is `028a495c41e14bec4855bad5fd55c1dcea51472c890285d7c499f5f4d93b22a1`. With `f_c = n_c / sum(n)`, weights are `f_c^(-1/2) / sum_j sqrt(f_j)`; their empirical frequency-weighted mean is one. This intervention is not proof that imbalance caused collapse or that the policy is clinically optimal.

The fixed weights, in class-ID order, are:

```text
[1.1775485072270662, 0.6250278281358197, 1.6794010379962119, 2.522787537775463, 4.564010031706509, 4.229021628686302]
```

Per-head `WCE = sum_i(w[y_i] * -log p_i[y_i]) / N`, where **N is total voxel count**, not selected-weight sum. Head loss is `0.5 WCE + 0.5 original foreground Soft Dice`. Original Dice sums batch/spatial axes, averages classes 1?5, uses smoothing `1e-5` and retains the absent-class false-positive policy. Deep supervision remains `(Lmain + 0.5 Lhalf + 0.25 Lquarter) / 1.75`; auxiliary logits use original trilinear upsampling while integer masks remain unchanged. Ordinary CE is separately logged. Equal coefficients do not imply equal effective gradient contributions.

**Comparison controls.** Residual Improved architecture (5,734,114 scalar parameters), frozen Train 26 patients/143 volumes, Validation 6/38, Test 6/30, loading/resampling/normalisation, 64-cube patches, original crop sampling, seed 3710, batch size 1, float32, AdamW LR 0.001/weight decay 0.00001 and deep supervision remain unchanged. Budget: ten epochs, 143 steps each, 1,430 updates, fresh initialization rather than E10 warm-start. No new scheduler, augmentation, cache or gradient-attribution utility was introduced. The older README recommendation for gradient attribution is historical and superseded by this approved decision; README is intentionally unchanged in this documentation-only task.

**Isolation and protection.** V2 reuses production model/preprocessing/Dice/checkpoint serialization/inference/metrics without modifying original protected files. Dedicated identity, configuration digest, source hashes, V2 path checks and complete configuration comparison protect resume. Original E10 checkpoints are not accepted as V2 resumes. V2 validation requires the fixed Epoch 10/1,430-update endpoint. Diagnostic mode writes no checkpoint; atomic completed-epoch saving supports interruption recovery. This design is source-reviewed behaviour, not evidence that the current full run has finished.

**Local synthetic evidence and interruption.** Codex's CPU tests covered weighted CE and analytic gradients, all-ones equivalence, exact original Dice parity, complete deep-supervision/auxiliary gradients, finite updates, checkpoint round-trip/resume and incompatibility rejection. Tests executed both actual entry points up to optimizer construction with mocked data/CUDA, comparing default parameters, RNG state, optimizer settings and ten-epoch DataLoader ordering. Local records show 107/107 PASS in 8.128 seconds, then the final 107/107 PASS in 8.281 seconds after the endpoint safeguard; continuation reran 15/15 focused tests in 2.896 seconds. Local WSL used Python 3.11.15/PyTorch 2.12.1+cu130 on CPU; missing NiBabel limited real NIfTI verification. These are software/synthetic tests, not remote MRI or GPU outcomes. The reported GPT-6 Astra usage-limit interruption was not a test failure; continuation preserved the existing files. The prior audit reported 45 pre-existing workspace files byte-identical and original Improved source hashes matching authenticated E10 evidence.

**Student execution and repair.** The student manually transferred files, compared remote hashes and original snapshot/last identity, and ran protected CPU preflight. The existing output directory contained only `preflight.txt`; `File exists` was not proof of prior training. Preflight passed. Two combined CPU attempts failed because `test_prediction.py` was missing remotely. After checking the matching production `predict.py` hash, the student transferred only the missing test: 107 tests in 9.884 seconds, OK, exit 0. Section B preserves the failure, diagnosis, correction, retest and nonfatal warning history. This was an incomplete remote test workspace, not a demonstrated V2 implementation defect.

**Real acceptance.** Test-only number 652631 was a scheduling estimate. Actual job 652632 completed on `a100-a` in 12 seconds: one update, 77 parameter tensors with finite gradients, 77 changed, loss 1.7107571363449097, no completed epoch/checkpoint. MaxPool3D's nonfatal determinism warning was consistent with the retained warn-only policy. This establishes real MRI/GPU executability, not convergence, clinical reliability or Validation improvement. Codex did not independently run the remote experiment.

**Formal full-training approval and early state.** After acceptance and QoS review, the student explicitly approved Phase C2. Three-hour test-only number 652641 was followed by real job **652649**, with pre-submission checks for prior job/checkpoint state and acceptance evidence. Setup identified fresh initialization and the approved experiment. At the then-current observation: **RUNNING**, `0:0`, `00:01:21`, node `a100-9`, at least Step 14 of Epoch 1, no completed-epoch checkpoint yet. Section B retains exact early losses. Completion and Validation were **UNVERIFIED at that point**; later evidence is appended below. Spot interruption was a risk, not a reported event.

**Contribution and academic-integrity disclosure.** ChatGPT assisted design, code review, command preparation, troubleshooting explanations and interpretation. Codex/Astra-assisted work materially authored the V2 code/tests, performed local checks and prepared this documentation; the student is not described as independently writing AI-generated code. The student set requirements, approved stages, manually transferred files, diagnosed missing remote files, ran CPU checks, submitted/monitored GPU jobs and supplied outputs. No unrecorded separate Astra action or remote Codex execution is inferred. Local inspection, historical local tool results and student-reported remote outputs remain distinct evidence levels. No V2 commit/push, final model selection, new Test evaluation or assessment check-off is claimed.

### Improved V2 - Phase C2 training completion and Phase C3 full Validation (12 October 2026)

**Task and approval.** After the student-approved one-step A100 acceptance, the student separately approved the ten-epoch Phase C2 run and then executed full Phase C3 Validation. The scientific question remained whether changing only the CE class weights would recover useful minority-class predictions relative to the original Improved E10 model. The frozen Train/Validation/Test assignments, residual architecture, original foreground Dice, deep-supervision weights, preprocessing, patch policy, seed, optimiser and ten-epoch budget remained the comparison controls. No held-out Test evaluation or final model-selection decision was part of these phases.

**Prompt record and roles. Prompt Summary - Not Verbatim:** the student requested an isolated V2 weighted-CE experiment, protected preflight and one-step acceptance before separately approving full training; the present documentation request supplies later training, checkpoint and Validation evidence and permits an `ai.md` update only. The student submitted and monitored Rangpur jobs, inspected checkpoint and JSONL outputs, and supplied results. ChatGPT/Astra/Codex assisted the research decision, isolated implementation, local synthetic checks, command preparation, troubleshooting, evidence interpretation and this documentation. None of those AI tools independently ran the student-reported A100 jobs. Exact original commands are recorded in Section B when available; the missing full scripts are not reconstructed as quotations.

**Initial attempt to completed training.** The earlier Step-14 observation correctly recorded job **652649 RUNNING** with no completed-epoch checkpoint. Later student-supplied Slurm output reports **COMPLETED 0:0 in 01:55:48**, and the program reports ten epochs, **1,430 optimiser updates** and final PASS. Epoch 10 mean training loss was **0.6459554164976506**; the last recorded patch loss was **0.4643269181251526**. These numbers show executed optimisation, not class-specific convergence. The remote training JSONL later showed **126,976 leading NUL bytes** and missing early records. Its cause is unknown, the file was not repaired, and a complete historical epoch-loss trajectory cannot be claimed from it. This limitation is retained alongside the independent Slurm, program and checkpoint completion evidence.

**Checkpoint and source protection.** The student's read-only CPU inspection reported a **68,960,401-byte** separate V2 checkpoint, SHA-256 `3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f`, safely loaded with `weights_only=True` and `map_location="cpu"`. It recorded `completed_epochs=10` and `total_steps=1430`, with model, optimiser, RNG and configuration fields needed for the protected resume contract. The original Improved `epoch10` and `last` checkpoints both passed their existing hash-manifest checks after V2 training. The V2 checkpoint was not locally copied or independently inspected in this documentation task; its successful full-volume inference is separately supported by the student-supplied Phase C3 result.

**Full Validation and evidence authentication.** The student executed `validate_improved_v2` on the six-patient, 38-volume frozen Validation split using the V2 Epoch 10 checkpoint and stride 32. Job **652948** on `a100-9` finished **COMPLETED 0:0** in **00:08:10** with empty stderr. The student-reported Validation JSONL audit found **41 valid records** (1 setup, 38 successful volume records, 1 summary, 1 PASS result), no invalid lines and no NUL bytes. A read-only audit reported 38 distinct successful cases, no failures, and pooled confusion counts equal to the sum of all 38 per-volume matrices. For Bones, Bladder, Rectum and Prostate, every Validation volume had ground-truth, predicted and true-positive voxels. The original V2 JSONL remains remote and was not independently parsed locally here; the original Improved E10 Validation JSONL was inspected locally to verify its comparison values.

**Result and limitations.** V2 `overall_pooled` Dice was **0.362280** for Bones, **0.826771** for Bladder, **0.461378** for Rectum and **0.445695** for Prostate, compared with original Improved values of approximately **0.000000394**, **0.768230**, **0** and **0** respectively. Background and Body Dice decreased from **0.976012/0.906781** to **0.953976/0.862141**. Section B retains full-precision Dice/IoU, patient-macro Dice, pooled counts, precision, recall, FP/FN, the hash fingerprints and evidence levels. Prostate recall **0.646212** accompanies precision **0.340148** and **314,665** pooled false positives. Bones and Rectum still miss substantial true tissue. Thus the Validation evidence supports recovery of nontrivial minority-class overlap with material trade-offs, not clinical reliability or unconditional superiority. No causal attribution to CE weighting is proven beyond this one controlled run, and no Test-set result is claimed.

**Disclosure and next evidence boundary.** AI materially assisted the V2 design, implementation, local tests, interpretation and documentation. The student authorised the experiment, ran the remote CPU/GPU stages and provided the original remote-output excerpts and checksums. The training-log corruption remains unresolved. Final model selection has not been frozen, the held-out Test set has not been evaluated, and no commit, push or assessment check-off is asserted by this entry.

### Improved V3 - duration-only continuation preparation (12 October 2026)

**Task and motivation. Prompt Summary - Not Verbatim:** the student authorised isolated optimisation implementation and local tests, prioritising whether V2 benefits from longer training, while requiring separate approval for GPU budgets and keeping Test untouched. Existing V2 measurements show recovered minority overlap, Bones/Rectum under-recall and Prostate false positives. Meaningful prior patch exposure and the existing prostate branch contradict a simple no-exposure explanation. A single E10 Validation endpoint and damaged early training log do not establish a plateau. No gradient-attribution utility or extra loss/sampling/architecture change was requested or introduced.

**AI work and technical decision.** Codex inspected the source/configuration/test contracts, previous diagnosis and AI history, local original experiment evidence availability and the local assessment PDF. It created train_improved_v3.py, improved_v3_config.json and test_improved_v3.py plus the experiment runbook. The V3 runner authenticates the pinned V2 parent and baseline Validation log, uses the protected original V2 loader/resume, restores optimiser and all RNG streams, reuses V2.run_epoch and existing full-volume Validation, and protects the training RNG sequence during Validation. A separate V3 config records parent provenance, source hashes and bounded schedule; original configuration/checkpoint protections were not relaxed.

**Selection, recovery and evidence method.** Initial scope is E11-E12 plus full Validation; additional stages through E20 remain conditional. Candidate selection maximises the mean patient-macro Dice of Bones/Rectum/Prostate subject to predeclared pooled class-regression limits and TP coverage across all 38 cases. V2 is retained unless a candidate qualifies. Every epoch checkpoint is separate and immutable. Scheduled Validation finishes before its checkpoint save; interruption requires replay from the previous saved boundary and explicit accounting of extra compute. Exclusive run locks, unique append-only/fsynced JSONL records and sequence/hash-chain checks prevent program log reuse and detect common corruption. This does not explain or repair the original V2 log.

**Verification and roles.** Codex executed synthetic CPU tests locally: final focused 19/19 and combined 126/126 passed, as recorded in Section B. Tests use real small PyTorch forwards/updates and serialized state where appropriate, with mocked MRI/CUDA orchestration. They verify implementation, not segmentation performance. The student supplied authentic historical Rangpur evidence and authorised local implementation; no V3 remote execution has occurred. Codex prepared resource-inspection, CPU-preflight, diagnostic, bounded-training, Validation and evidence-audit commands for student execution after approval. The source code and tests are AI-generated assistance, not independent student authorship.

**Remaining boundary.** Request 64 A100 wall-time minutes maximum for the first diagnostic/main-stage batch, subject to current Slurm eligibility and approval. Remote source/checkpoint/log preflight, NIfTI/CUDA acceptance, actual V3 runtime and class metrics remain unverified. No new Dice, Test result, model freeze, commit or push is claimed. The specification's Test >=0.70 aggregation remains an authoritative-clarification gap. Longer training is the first controlled experiment; V4/V5 are deferred until its actual Validation evidence is reviewed.

### Future record standard

After each tested and reviewed batch, maintain both records. The **Verified Command Reference** records command → environment → purpose → expected output → why expected → actual output → interpretation, with notes about limits. The **AI-Assisted Development Log** records problem → prompt or clearly marked prompt summary → AI actions and technical method → student actions → verification and evidence level → failures/corrections/retests → result → AI/human disclosure. Preserve meaningful abandoned attempts and previous records, cite exact files, source diffs, test outputs and commits where available, and mark unavailable evidence rather than inventing it. Never include passwords, tokens, private keys, patient-identifying information or confidential dataset contents.

## Documentation Maintenance Rules

1. Append a milestone entry only after actual execution and verification.
2. Preserve all previous approved milestone records.
3. For each command, record its purpose, expected output, reason for that expectation, actual result and interpretation, with environment and notes.
4. Clearly separate Windows PowerShell commands from Rangpur Linux Bash commands.
5. Use exact observed values for numerical results.
6. Record meaningful failures and confirmed resolutions in troubleshooting notes.
7. Never record secrets or credentials.
8. Distinguish AI assistance, human execution and verified evidence.
9. Never fabricate tests, training results, metrics or completion status.
10. Keep the reference concise, searchable and reusable.

## V4 E20 - AI-Assisted Implementation and Verification (13 October 2026)

### Scope of AI assistance

ChatGPT assisted with the experimental design, Python implementation,
PowerShell and Rangpur Bash commands, source-integrity checks,
CPU numerical tests, synthetic integration tests, recovery safeguards
and draft documentation.

The student executed commands in the local Windows environment and
Rangpur environment, supplied the resulting terminal output and
reviewed PASS/FAIL results step by step.

This entry documents the reported executions; it does not imply
that ChatGPT independently accessed the student's workstation,
Rangpur account or actual experimental checkpoints.

### Experimental implementation

A paired V4 experiment was prepared from the selected V3 E12 parent.

- Control uses the original V3 loss.
- Treatment increases the Prostate Tversky alpha from 0.5 to 0.7.
- Both arms plan E13-E20, with 143 updates per epoch.
- Full-volume Validation is scheduled at E14/E16/E18/E20.
- Step journals include training identity, crop metadata and
  optimiser parameters.
- Resume and comparison routines enforce declared evidence boundaries.

The historical E14 draft was preserved separately.

### Student-executed verification

| Check | Observed output |
| --- | --- |
| Numerical loss/gradient tests | 3/3 PASS |
| Mocked resume boundaries | 7/7 PASS |
| Synthetic paired integrity | 8/8 PASS |
| Synthetic 143-step CPU training | 3/3 PASS |
| CPU checkpoint/journal integrity | 5/5 PASS |
| Synthetic failed-E13 retry | 9/9 PASS |
| Mocked lock lifecycle | 4/4 PASS |
| Final CPU regression | 3/3 PASS |

The final CPU regression was run on the latest synced E20 Runner.
It reported three successful unittest cases and exit code zero.

### Corrections during development

A previous helper-extraction script accidentally introduced two
identical definitions of `validate_failed_parent_retry()`.
An initial verification assertion also counted a function definition
and its call as though both were calls.

The student ran an AST-based diagnostic, identified the duplicate
definitions, then executed a guarded deduplication script. The
resulting code retained one definition and one call, with the
original E14 implementation unchanged.

Other focused changes added Validation-before-Checkpoint ordering,
an explicit failed-parent retry guard, source and journal auditing,
early exclusive Lock acquisition and ownership-aware Lock release.

These modifications were followed by bounded static or runtime
checks. Static PASS alone was not interpreted as proof of runtime
correctness.

### Reproducibility and integrity boundaries

Windows/Rangpur SHA-256 equality was checked after E20 source
transfers. The source identity snapshot is recorded in
`REPORT_CODE_TRACEABILITY.md`.

Rangpur login-node CPU autograd emitted a CUDA driver compatibility
warning, but the CPU tests completed successfully. This does not
establish CUDA compatibility on allocated A100 compute nodes.

Synthetic recovery tests do not prove actual Slurm SIGKILL recovery.
Damaged or ambiguous journals and orphaned locks require manual
review. No automatic deletion of historical evidence is authorised.

No V4 GPU training, E20 Validation result, independent Test score,
performance improvement, model freeze or Git push is claimed here.

### Academic and scientific responsibility

AI-generated implementation suggestions and test scripts are
development assistance, not independent experimental evidence.
The student remains responsible for understanding the code,
checking assessment AI-use requirements, executing tests,
interpreting actual MRI outcomes and documenting limitations.
