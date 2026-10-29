# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

## HipMRI segmentation submission

Open [the project README](HipMRI_UNet_s4906923/README.md) for the technical report, results and running instructions. The same report is available as [a PDF](HipMRI_UNet_s4906923/REPORT.pdf).

The submitted code runs the Standard 3D U-Net and Improved v3. Their Validation predictions have been checked against the original experiment results. Saved results from all ten model runs remain in the project's `evidence/` directory for comparison.

The historical `20261017` research workspace and private model backups are archived separately from this submission. Tests and saved-record inspection run without that archive. New MRI inference requires authorised data and trusted weights supplied through the documented command-line arguments; datasets and weight files are excluded from this folder and Git.
