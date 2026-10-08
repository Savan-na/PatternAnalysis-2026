# HipMRI 3D Improved U-Net Segmentation

## Project Status

This is an initial project scaffold. No experimental results exist yet.

## Problem

This project investigates volumetric MRI segmentation for prostate radiotherapy planning using HipMRI.

## Engineering Question

Does true 3D volumetric context, followed by residual learning and deep supervision, reduce clinically important boundary and slice-transition errors compared with a standard 2D baseline, and are the gains worth the additional computational cost?

## Planned Model Comparison

- 2D U-Net — baseline
- 3D U-Net — spatial-context control
- 3D Improved U-Net — final Hard Difficulty model

## Planned Evaluation

- Per-class Dice
- Per-class IoU
- Prostate base/apex analysis
- Failure-case analysis
- Peak GPU VRAM
- Inference time per case

## Repository Structure

- `modules.py`: Model components for the planned comparison.
- `dataset.py`: Dataset handling for the planned experiments.
- `train.py`: Training workflow for the planned models.
- `predict.py`: Inference workflow for the planned models.

## Artificial Intelligence Usage Disclosure

This section will be completed progressively as AI-assisted development occurs.
