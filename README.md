# COMP3710 HipMRI Final Project

This repository contains the active HipMRI multi-class 3D MRI segmentation project under [recognition/](recognition). The [recognition overview](recognition/README.md) gives the current experiment state; the [model package README](recognition/hipmri_3d_improved_unet_hard/README.md) documents implementation and reproducibility; the [technical report](recognition/FINAL-REPORT.md) presents methods and Validation findings; and the research archive (local-only: `research_archive/ARCHIVE_README.md`) preserves dated evidence. The E20 Validation decision rejected V4 Treatment and retained V3 E12 as the protected candidate. A formal model freeze and Held-out Test evaluation have not occurred.

## Documentation map

- [Recognition overview](recognition/README.md): current status and navigation.
- [Experiment history](recognition/docs/experiments.md): Baseline through V4 protocol, results, decisions and evidence levels.
- [Final Report](recognition/FINAL-REPORT.md): course-facing methods and scientific interpretation.
- [Model package README](recognition/hipmri_3d_improved_unet_hard/README.md): implementation and reproducibility.
- [Analysis](recognition/docs/analysis/class-prediction-collapse.md) and [evidence](recognition/docs/evidence/failure-cases.md): detailed E10 diagnosis and source provenance.

The local-only historical `research_archive/` is excluded from the GitHub submission; its original plans and receipts remain on this machine.

## Original repository context

# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2025 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.

