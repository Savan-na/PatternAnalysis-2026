# Paper and assignment decision register

| Decision | Basis | Implementation status / deviation |
|---|---|
| Core research question, six-class HipMRI and test endpoints | COMP3710 commission and revised data-driven research direction | Raw data and sampling exposure are investigated before model changes; no new results |
| Patient-level split | Existing approved project split, 26/6/6 patients and 143/38/30 volumes | Preserved for comparison; prior Validation reuse disclosed |
| 2D U-Net | Ronneberger et al. (2015), assignment baseline option | Existing independent implementation retained as background; no required retraining |
| Standard 3D U-Net | Çiçek et al. (2016), DOI 10.1007/978-3-319-46723-8_49 | 3D encoder/decoder, skip links, convolution and pooling; same-padding/group normalization and six-class output are explicit adaptations |
| Improved 3D U-Net | Isensee et al., *Brain Tumor Segmentation and Radiomics Survival Prediction* (2018), arXiv:1802.10508 | Pre-activation residual context, stride-2 convolution, interpolation-based upsampling, localization blocks, additive deep-supervision logits; six-class HipMRI adaptation |
| CAN3D and its pelvis comparisons | Dai et al. (2022), DOI 10.1016/j.media.2022.102562 | Literature context only; not a fourth training arm or a source of directly comparable target results |
| Common image grid | Assignment specifies downsampled 3D HipMRI and alignment | Factor-two voxel-edge-aligned grid is an initial engineering choice; must be validated on real data before training |
| Loss and optimizer | Still to be finalized before GPU training | Do not infer that Dai's DSF is the original loss of either U-Net |
| Academic integrity | Course commission and source attribution requirements | Original PyTorch code written here; cite papers and consulted author code in README/report |

Primary sources:

- Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597
- Çiçek et al. (2016): https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf
- Çiçek author resources: https://lmb.informatik.uni-freiburg.de/lmbsoft/unet.en.html
- Isensee et al. (2018): https://arxiv.org/abs/1802.10508
- Isensee team code: https://github.com/MIC-DKFZ/BraTS2017
- Dai et al. (2022): https://doi.org/10.1016/j.media.2022.102562
- CSIRO HipMRI data: https://data.csiro.au/collection/csiro%3A51392

The original Çiçek network's reported 19.07M parameters and Dai's reduced
pelvis comparator's 1.196M parameters describe different instances. Matching
one number cannot establish layer-for-layer fidelity. Do not claim exact
reproduction of unavailable Dai comparator settings.

Initial implementations have 1,942,374 (2D), 1,412,550 (standard 3D), and
2,366,946 (improved 3D) trainable parameters at their documented default
widths. These counts are local code measurements, **not** paper claims or
validated full-volume memory measurements.

## Imbalance methods in the three primary papers

Ronneberger et al. (2015) use a pixel-weighted cross-entropy map to compensate
class frequency and emphasize the narrow background gaps between touching
cells; strong elastic augmentation addresses limited annotated images. The
cell-border formula and constants are specific to their microscopy task.

Çiçek et al. (2016) reduce loss weight for common background and increase it
for the less common inner tubule. They separately give *unlabelled* voxels
zero loss weight because their 3D volumes are sparsely annotated. HipMRI's
labelled Background is not an unlabelled class. Their batch normalization and
spatial/gray-value augmentation serve optimization and generalization, not
class-frequency correction.

Isensee et al. (2018) standardize each patient's MRI modality using brain-region
intensity statistics, then clip and rescale. They explicitly address their
background-to-enhancing-tumor imbalance with multiclass Dice loss. Random 3D
patches, instance normalization, augmentation, dropout and L2 regularization
are also described; the paper does not specify foreground-anchored sampling as
its own method. Its brain-specific intensity region and clipping thresholds
require HipMRI Train evidence before adaptation.

Current HipMRI G3 evidence supports keeping the validated nonzero-intensity
Z-score and measured downsampling as a starting point. G4 verified a
patient-balanced patch mechanism and exact post-transform label exposure
recorder on all 26 Train patients in one CPU dry-run epoch. The reference
sampling design for the first controlled Standard/Improved comparison is equal
patient selection and uniform crops; organ-aware placement remains a targeted
candidate because its pooled exposure gain was modest and patient-level gains
were mixed. This selection is based on HipMRI Train evidence, not prescribed
by the three papers. Actual training exposure and Validation performance must
still be measured. No Dice/weighted-CE choice has been established.

The G5 two-patch Standard diagnostic still predicted no Bladder, Rectum or
Prostate at 500 optimizer steps under the provisional unweighted CE plus
foreground Dice mixture. The follow-up configuration
`code/config/g5_weighted_ce_diagnostic.json` uses square-root inverse class
frequency relative to Body from the G4 Train-only patient-uniform Patch
exposure, capped at 10. This bounded weighting is a HipMRI diagnostic choice,
not a parameter copied from any of the three papers or a final loss decision.
