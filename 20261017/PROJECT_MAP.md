# Project File Map / 项目文件总览

Updated / 更新时间: **2026-10-20 01:11:29 AEST**

**English:** This generated index lists every directory and file under `20261017`, including local logs and generated artifacts. The content column summarizes structure (headings, code definitions, fields or file type) rather than copying another document's prose. Run `python code/tools/update_project_map.py --watch` to keep it current while the watcher is active.

**中文：** 本自动目录列出 `20261017` 下的全部目录和文件，包括本地日志及生成产物。内容栏概述章节、代码定义、字段或文件类型，不重复抄录其他文档。运行 `python code/tools/update_project_map.py --watch`，并保持进程运行，即可持续更新。

Directories / 目录: **47** · Files / 文件: **404**

## Directory Tree / 目录树

```text
20261017/
├── code/
│   ├── config/
│   │   ├── arms/
│   │   │   ├── baseline_2d/
│   │   │   │   └── arm.json
│   │   │   ├── baseline_3d/
│   │   │   │   └── arm.json
│   │   │   └── improved_3d/
│   │   │       └── arm.json
│   │   ├── g3_patch_audit.json
│   │   ├── g5_2d_training_frozen.json
│   │   ├── g5_controlled_training_candidate.json
│   │   ├── g5_controlled_training_frozen.json
│   │   ├── g5_weighted_ce_diagnostic.json
│   │   ├── g7_edge_sampling_candidate.json
│   │   └── patient_split.json
│   ├── docs/
│   │   ├── EXECUTION.md
│   │   ├── RANGPUR_COMMANDS.md
│   │   ├── SOURCE_DECISIONS.md
│   │   └── STUDY_PROTOCOL.md
│   ├── hipmri_rewrite/
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_2d.py
│   │   │   ├── baseline_3d.py
│   │   │   ├── common.py
│   │   │   └── improved_3d.py
│   │   ├── __init__.py
│   │   ├── audit.py
│   │   ├── boundaries.py
│   │   ├── data.py
│   │   ├── inference.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── profile.py
│   │   ├── sampling.py
│   │   ├── sampling_2d.py
│   │   ├── sampling_edges.py
│   │   ├── splits.py
│   │   └── training.py
│   ├── tests/
│   │   ├── test_boundaries.py
│   │   ├── test_data.py
│   │   ├── test_g5_candidate.py
│   │   ├── test_metrics.py
│   │   ├── test_models_and_losses.py
│   │   ├── test_raw_audit.py
│   │   ├── test_sampling.py
│   │   ├── test_sampling_2d.py
│   │   ├── test_sampling_audit.py
│   │   ├── test_sampling_edges.py
│   │   ├── test_splits.py
│   │   ├── test_training_recovery.py
│   │   └── test_training_selection.py
│   ├── tools/
│   │   ├── __pycache__/
│   │   ├── audit_g2_train_slices.py
│   │   ├── audit_g3_candidate_sampling.py
│   │   ├── audit_g3_existing_pipeline.py
│   │   ├── audit_g4_patient_sampling.py
│   │   ├── audit_g5_2d_sampling.py
│   │   ├── audit_g5_aux_logits_validation.py
│   │   ├── audit_g5_train_patch_geometry.py
│   │   ├── audit_g5_validation_body.py
│   │   ├── audit_g7_edge_sampling.py
│   │   ├── audit_train_raw.py
│   │   ├── evaluate_g6_test.py
│   │   ├── package_transfer.py
│   │   ├── probe_g5_full_volume.py
│   │   ├── probe_g5_patch_gpu.py
│   │   ├── probe_g5_training_smoke.py
│   │   ├── review_g1_bladder_islands.py
│   │   ├── review_g1_overlays.py
│   │   ├── review_g1_train_contact_sheets.py
│   │   ├── review_g5_validation_failures.py
│   │   ├── review_g6_body_boundaries.py
│   │   ├── review_g6_test_cases.py
│   │   ├── submit_g5_2d.sh
│   │   ├── submit_g5_2d_extend_50.sh
│   │   ├── submit_g5_40.sh
│   │   ├── submit_g5_extend_50.sh
│   │   ├── submit_g6_test.sh
│   │   ├── submit_g7_edge_audit.sh
│   │   ├── summarize_all_class_review.py
│   │   ├── summarize_g2_slice_detail.py
│   │   ├── summarize_g2_train_audit.py
│   │   ├── summarize_g3_candidate_sampling.py
│   │   ├── summarize_g3_pipeline.py
│   │   ├── summarize_g4_patient_sampling.py
│   │   ├── summarize_g5_2d_validation.py
│   │   ├── summarize_g5_body_review.py
│   │   ├── summarize_g5_validation.py
│   │   ├── summarize_g6_test.py
│   │   ├── summarize_g7_edge_sampling.py
│   │   ├── tiny_overfit_g5.py
│   │   ├── train_g5_2d.py
│   │   ├── train_g5_cohort.py
│   │   └── update_project_map.py
│   ├── transfer/
│   │   ├── 20261017_source.zip
│   │   └── g7_edge_audit_20261019.zip
│   ├── README.md
│   └── requirements.txt
├── logs/
│   ├── audits/
│   │   ├── g3_patch_candidate_full_20261018/
│   │   │   ├── manifest.json
│   │   │   └── patches.csv
│   │   ├── g4_patient_full_20261018/
│   │   │   ├── manifest.json
│   │   │   └── patches.csv
│   │   ├── g5_2d_sampling_full_20261018/
│   │   │   ├── manifest.json
│   │   │   └── patches.csv
│   │   ├── g7_edge_sampling_full_20261019/
│   │   │   ├── manifest.json
│   │   │   └── patches.csv
│   │   ├── g1_train_contact_sheets_20261018.json
│   │   ├── g3_existing_pipeline_full_20261018_job671892.json
│   │   └── train_full_20261018_job671208.json
│   ├── internal/
│   │   ├── map_watcher_stderr.txt
│   │   ├── map_watcher_stdout.txt
│   │   └── project_map_watcher.pid
│   ├── profiles/
│   │   ├── g5_full_volume_probe_20261018.json
│   │   ├── g5_patch_probe_20261018.json
│   │   ├── g5_standard_500_20261018.json
│   │   ├── g5_standard_weighted_500_20261018.json
│   │   ├── g5_tiny_overfit_20261018.json
│   │   └── g5_training_smoke_20261018.json
│   ├── training/
│   │   ├── baseline_2d/
│   │   │   ├── epoch_001.json
│   │   │   ├── epoch_002.json
│   │   │   ├── epoch_003.json
│   │   │   ├── epoch_004.json
│   │   │   ├── epoch_005.json
│   │   │   ├── epoch_006.json
│   │   │   ├── epoch_007.json
│   │   │   ├── epoch_008.json
│   │   │   ├── epoch_009.json
│   │   │   ├── epoch_010.json
│   │   │   ├── epoch_011.json
│   │   │   ├── epoch_012.json
│   │   │   ├── epoch_013.json
│   │   │   ├── epoch_014.json
│   │   │   ├── epoch_015.json
│   │   │   ├── epoch_016.json
│   │   │   ├── epoch_017.json
│   │   │   ├── epoch_018.json
│   │   │   ├── epoch_019.json
│   │   │   ├── epoch_020.json
│   │   │   ├── epoch_021.json
│   │   │   ├── epoch_022.json
│   │   │   ├── epoch_023.json
│   │   │   ├── epoch_024.json
│   │   │   ├── epoch_025.json
│   │   │   ├── epoch_026.json
│   │   │   ├── epoch_027.json
│   │   │   ├── epoch_028.json
│   │   │   ├── epoch_029.json
│   │   │   ├── epoch_030.json
│   │   │   ├── epoch_031.json
│   │   │   ├── epoch_032.json
│   │   │   ├── epoch_033.json
│   │   │   ├── epoch_034.json
│   │   │   ├── epoch_035.json
│   │   │   ├── epoch_036.json
│   │   │   ├── epoch_037.json
│   │   │   ├── epoch_038.json
│   │   │   ├── epoch_039.json
│   │   │   ├── epoch_040.json
│   │   │   ├── epoch_041.json
│   │   │   ├── epoch_042.json
│   │   │   ├── epoch_043.json
│   │   │   ├── epoch_044.json
│   │   │   ├── epoch_045.json
│   │   │   ├── epoch_046.json
│   │   │   ├── epoch_047.json
│   │   │   ├── epoch_048.json
│   │   │   ├── epoch_049.json
│   │   │   ├── epoch_050.json
│   │   │   └── manifest.json
│   │   ├── baseline_3d/
│   │   │   ├── epoch_001.json
│   │   │   ├── epoch_002.json
│   │   │   ├── epoch_003.json
│   │   │   ├── epoch_004.json
│   │   │   ├── epoch_005.json
│   │   │   ├── epoch_006.json
│   │   │   ├── epoch_007.json
│   │   │   ├── epoch_008.json
│   │   │   ├── epoch_009.json
│   │   │   ├── epoch_010.json
│   │   │   ├── epoch_011.json
│   │   │   ├── epoch_012.json
│   │   │   ├── epoch_013.json
│   │   │   ├── epoch_014.json
│   │   │   ├── epoch_015.json
│   │   │   ├── epoch_016.json
│   │   │   ├── epoch_017.json
│   │   │   ├── epoch_018.json
│   │   │   ├── epoch_019.json
│   │   │   ├── epoch_020.json
│   │   │   ├── epoch_021.json
│   │   │   ├── epoch_022.json
│   │   │   ├── epoch_023.json
│   │   │   ├── epoch_024.json
│   │   │   ├── epoch_025.json
│   │   │   ├── epoch_026.json
│   │   │   ├── epoch_027.json
│   │   │   ├── epoch_028.json
│   │   │   ├── epoch_029.json
│   │   │   ├── epoch_030.json
│   │   │   ├── epoch_031.json
│   │   │   ├── epoch_032.json
│   │   │   ├── epoch_033.json
│   │   │   ├── epoch_034.json
│   │   │   ├── epoch_035.json
│   │   │   ├── epoch_036.json
│   │   │   ├── epoch_037.json
│   │   │   ├── epoch_038.json
│   │   │   ├── epoch_039.json
│   │   │   ├── epoch_040.json
│   │   │   ├── epoch_041.json
│   │   │   ├── epoch_042.json
│   │   │   ├── epoch_043.json
│   │   │   ├── epoch_044.json
│   │   │   ├── epoch_045.json
│   │   │   ├── epoch_046.json
│   │   │   ├── epoch_047.json
│   │   │   ├── epoch_048.json
│   │   │   ├── epoch_049.json
│   │   │   ├── epoch_050.json
│   │   │   └── manifest.json
│   │   └── improved_3d/
│   │       ├── epoch_001.json
│   │       ├── epoch_002.json
│   │       ├── epoch_003.json
│   │       ├── epoch_004.json
│   │       ├── epoch_005.json
│   │       ├── epoch_006.json
│   │       ├── epoch_007.json
│   │       ├── epoch_008.json
│   │       ├── epoch_009.json
│   │       ├── epoch_010.json
│   │       ├── epoch_011.json
│   │       ├── epoch_012.json
│   │       ├── epoch_013.json
│   │       ├── epoch_014.json
│   │       ├── epoch_015.json
│   │       ├── epoch_016.json
│   │       ├── epoch_017.json
│   │       ├── epoch_018.json
│   │       ├── epoch_019.json
│   │       ├── epoch_020.json
│   │       ├── epoch_021.json
│   │       ├── epoch_022.json
│   │       ├── epoch_023.json
│   │       ├── epoch_024.json
│   │       ├── epoch_025.json
│   │       ├── epoch_026.json
│   │       ├── epoch_027.json
│   │       ├── epoch_028.json
│   │       ├── epoch_029.json
│   │       ├── epoch_030.json
│   │       ├── epoch_031.json
│   │       ├── epoch_032.json
│   │       ├── epoch_033.json
│   │       ├── epoch_034.json
│   │       ├── epoch_035.json
│   │       ├── epoch_036.json
│   │       ├── epoch_037.json
│   │       ├── epoch_038.json
│   │       ├── epoch_039.json
│   │       ├── epoch_040.json
│   │       ├── epoch_041.json
│   │       ├── epoch_042.json
│   │       ├── epoch_043.json
│   │       ├── epoch_044.json
│   │       ├── epoch_045.json
│   │       ├── epoch_046.json
│   │       ├── epoch_047.json
│   │       ├── epoch_048.json
│   │       ├── epoch_049.json
│   │       ├── epoch_050.json
│   │       └── manifest.json
│   ├── validation/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   └── improved_3d/
│   └── README.md
├── results/
│   ├── case_studies/
│   │   └── comparisons/
│   │       ├── g5_body_validation_B040_Week4.png
│   │       ├── g5_body_validation_J026_Week4.png
│   │       ├── g5_body_validation_K018_Week4.png
│   │       ├── g5_validation_B040_Week1_prostate.png
│   │       ├── g5_validation_J026_Week1_rectum.png
│   │       ├── g5_validation_K018_Week3_rectum.png
│   │       ├── g5_validation_S022_Week6_prostate.png
│   │       ├── g5_validation_selected_cases.json
│   │       ├── g6_body_L011_Week5.png
│   │       ├── g6_body_M015_Week3.png
│   │       ├── g6_body_selected_cases.json
│   │       ├── g6_body_W041_Week0.png
│   │       ├── g6_test_M015_Week7_bladder.png
│   │       ├── g6_test_M020_Week1_prostate.png
│   │       ├── g6_test_selected_cases.json
│   │       ├── g6_test_W029_Week6_prostate.png
│   │       └── README.md
│   ├── data_analysis/
│   │   ├── g1_bladder_review/
│   │   │   ├── g1_bladder_island_D031_Week7.png
│   │   │   ├── g1_bladder_island_N010_Week3.png
│   │   │   └── README.md
│   │   ├── g1_patient_screen/
│   │   │   ├── g1_train_patient_B006.png
│   │   │   ├── g1_train_patient_B038.png
│   │   │   ├── g1_train_patient_C032.png
│   │   │   ├── g1_train_patient_D031.png
│   │   │   ├── g1_train_patient_G021.png
│   │   │   ├── g1_train_patient_H007.png
│   │   │   ├── g1_train_patient_H017.png
│   │   │   ├── g1_train_patient_J005.png
│   │   │   ├── g1_train_patient_K019.png
│   │   │   ├── g1_train_patient_K042.png
│   │   │   ├── g1_train_patient_M004.png
│   │   │   ├── g1_train_patient_M013.png
│   │   │   ├── g1_train_patient_M023.png
│   │   │   ├── g1_train_patient_M036.png
│   │   │   ├── g1_train_patient_N010.png
│   │   │   ├── g1_train_patient_O025.png
│   │   │   ├── g1_train_patient_R016.png
│   │   │   ├── g1_train_patient_R024.png
│   │   │   ├── g1_train_patient_R039.png
│   │   │   ├── g1_train_patient_S028.png
│   │   │   ├── g1_train_patient_S033.png
│   │   │   ├── g1_train_patient_S035.png
│   │   │   ├── g1_train_patient_T009.png
│   │   │   ├── g1_train_patient_T014.png
│   │   │   ├── g1_train_patient_V027.png
│   │   │   ├── g1_train_patient_W012.png
│   │   │   └── README.md
│   │   ├── g1_selected_cases/
│   │   │   ├── g1_overlay_B038_Week0.png
│   │   │   ├── g1_overlay_D031_Week7.png
│   │   │   ├── g1_overlay_H017_Week7.png
│   │   │   ├── g1_overlay_K019_Week1.png
│   │   │   ├── g1_overlay_N010_Week3.png
│   │   │   ├── g1_overlay_S033_Week1.png
│   │   │   └── README.md
│   │   ├── g2_train_summary/
│   │   │   ├── class_distribution.csv
│   │   │   ├── class_frequency_log.svg
│   │   │   ├── intensity_case_summary.svg
│   │   │   ├── intensity_statistics.csv
│   │   │   ├── longitudinal_organ_variation.csv
│   │   │   ├── organ_bbox_centers.svg
│   │   │   ├── organ_slices_per_volume.svg
│   │   │   ├── patient_rare_organ_share.svg
│   │   │   ├── per_patient_distribution.csv
│   │   │   ├── per_slice_class_coverage.csv
│   │   │   ├── per_volume_distribution.csv
│   │   │   ├── per_volume_intensity_histogram.csv
│   │   │   ├── per_volume_slice_coverage.csv
│   │   │   ├── pooled_positive_intensity_histogram.csv
│   │   │   ├── pooled_positive_intensity_histogram.svg
│   │   │   ├── raw_axial_slice_hit_rate.svg
│   │   │   ├── README.md
│   │   │   ├── slice_presence_summary.csv
│   │   │   ├── spatial_statistics.csv
│   │   │   └── voxel_spacing.svg
│   │   ├── g3_pipeline/
│   │   │   ├── candidate_patch_class_exposure.csv
│   │   │   ├── candidate_patch_exposure.svg
│   │   │   ├── candidate_patch_patient_exposure.csv
│   │   │   ├── organ_physical_volume_ratio.svg
│   │   │   ├── organ_preservation_by_volume.csv
│   │   │   └── README.md
│   │   ├── g4_patient_sampling/
│   │   │   ├── class_exposure.csv
│   │   │   ├── g5_train_patch_geometry.json
│   │   │   ├── patient_balanced_exposure.svg
│   │   │   ├── patient_exposure.csv
│   │   │   └── README.md
│   │   ├── g7_edge_sampling/
│   │   │   ├── class_exposure.csv
│   │   │   ├── g7_crop_origins.png
│   │   │   ├── g7_crop_origins.svg
│   │   │   ├── g7_exposure_and_geometry.png
│   │   │   ├── g7_exposure_and_geometry.svg
│   │   │   ├── patient_class_exposure.csv
│   │   │   ├── README.md
│   │   │   └── verification.json
│   │   ├── historical_train/
│   │   │   ├── HipMRI_Pretraining_Data_Analysis.previous-41201a5e.bak
│   │   │   ├── HipMRI_Pretraining_Data_Analysis.previous-5dc166bd.bak
│   │   │   ├── historical_patch_vs_full_volume.svg
│   │   │   ├── historical_resampling_volume_ratio.svg
│   │   │   ├── historical_train_case_profile.csv
│   │   │   ├── historical_train_statistics.json
│   │   │   ├── README.md
│   │   │   ├── train_class_distribution.svg
│   │   │   ├── train_foreground_distribution.svg
│   │   │   ├── train_patient_examinations.svg
│   │   │   └── train_rare_organ_volumes.svg
│   │   └── README.md
│   ├── figures/
│   │   ├── baseline_2d/
│   │   │   ├── g5_2d_validation_all_classes.png
│   │   │   ├── g5_2d_validation_all_classes.svg
│   │   │   └── README.md
│   │   └── comparisons/
│   │       ├── g5_3d_validation_all_classes.png
│   │       ├── g5_3d_validation_all_classes.svg
│   │       ├── g5_three_arm_training_trajectories.png
│   │       ├── g5_three_arm_training_trajectories.svg
│   │       ├── g5_three_arm_validation_all_classes.png
│   │       ├── g5_three_arm_validation_all_classes.svg
│   │       ├── g5_validation_body_boundaries.png
│   │       ├── g5_validation_body_boundaries.svg
│   │       ├── g5_validation_patient_class_dice.png
│   │       ├── g5_validation_patient_class_dice.svg
│   │       ├── g6_test_all_classes.png
│   │       ├── g6_test_all_classes.svg
│   │       ├── g6_test_failure_patterns.png
│   │       ├── g6_test_failure_patterns.svg
│   │       ├── g6_test_hd95_all_foreground.png
│   │       ├── g6_test_hd95_all_foreground.svg
│   │       └── README.md
│   ├── reports/
│   │   ├── HipMRI_P1_独立研究审查_ZH.md
│   │   ├── HipMRI_Pretraining_Data_Analysis.md
│   │   ├── HipMRI_Research_Report.md
│   │   └── HipMRI_模型复盘与下一轮实验_ZH.md
│   ├── tables/
│   │   └── comparisons/
│   │       ├── g5_validation_aux_logits.json
│   │       ├── g5_validation_body_audit.json
│   │       ├── g5_validation_case_class_metrics.csv
│   │       ├── g6_test_3d.json
│   │       ├── g6_test_case_class_metrics.csv
│   │       └── README.md
│   └── README.md
├── .gitignore
├── PROJECT_MAP.md
└── README.md
```

## File Contents / 文件内容概览

| File / 文件 | English content overview | 中文内容概览 |
|---|---|---|
| `.gitignore` | Git rules for local artifacts; UTF-8 text with 22 lines; 482 bytes | 本地运行产物的 Git 规则；UTF-8 文本，22 行；482 字节 |
| `code/config/arms/baseline_2d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/baseline_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/improved_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/g3_patch_audit.json` | Frozen candidate-crop audit design, not a training configuration; JSON fields: scope, seed, patch_shape_xyz, draws_per_case_per_arm, arm_schedules | 固定的候选裁剪审计设计，非训练配置；JSON 字段：scope, seed, patch_shape_xyz, draws_per_case_per_arm, arm_schedules |
| `code/config/g5_2d_training_frozen.json` | JSON fields: status, purpose, model, model_defaults, split_sha256, train_cases, train_patients, validation_cases, validation_patients, seed, preprocessing, sampling | JSON 字段：status, purpose, model, model_defaults, split_sha256, train_cases, train_patients, validation_cases, validation_patients, seed, preprocessing, sampling |
| `code/config/g5_controlled_training_candidate.json` | Historical candidate for the paired 3D cohort training protocol; JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | 双模型队列训练方案的历史候选配置；JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/g5_controlled_training_frozen.json` | Frozen paired 3D cohort training protocol with resumable checkpoint selection; JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | 已冻结的双模型训练方案，含可恢复权重和验证集选择规则；JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/g5_weighted_ce_diagnostic.json` | Train-only weighted CE configuration for the Standard tiny learning diagnostic; JSON fields: scope, source_table, source_table_sha256, source_policy, formula, ce_weight_in_combined_loss, class_weights | 仅用训练数据为 Standard 微型学习诊断配置加权交叉熵；JSON 字段：scope, source_table, source_table_sha256, source_policy, formula, ce_weight_in_combined_loss, class_weights |
| `code/config/g7_edge_sampling_candidate.json` | Predeclared edge crop search and Train exposure gates, not model training config; JSON fields: status, seed, epochs, patch_xyz, patches_per_patient_epoch, preserved_uniform_draws, edge, training_gate, baseline_geometry_counts, scope, fallback, negative_definition | 预先定义边缘裁剪搜索及训练曝光门槛，非模型训练配置；JSON 字段：status, seed, epochs, patch_xyz, patches_per_patient_epoch, preserved_uniform_draws, edge, training_gate, baseline_geometry_counts, scope, fallback, negative_definition |
| `code/config/patient_split.json` | Frozen patient groups and expected counts; JSON fields: provenance, train, validation, test, expected_volumes | 固定患者分组与预期数量；JSON 字段：provenance, train, validation, test, expected_volumes |
| `code/docs/EXECUTION.md` | Evidence and execution gates; Sections: Execution and evidence checklist; Evidence identity; G1: raw integrity gate; G2: completed Train-only raw EDA gate; G3: existing-pipeline diagnosis (completed) | 证据与执行关口；章节标题：Execution and evidence checklist; Evidence identity; G1: raw integrity gate; G2: completed Train-only raw EDA gate; G3: existing-pipeline diagnosis (completed) |
| `code/docs/RANGPUR_COMMANDS.md` | Separate Windows and Rangpur commands; Sections: Rangpur G2 and G3 command handoff; Transfer the G2 slice audit script; Two-case smoke run on a CPU node; Full Train run after the smoke check; G3 existing-pipeline smoke run | 分开的 Windows 与 Rangpur 命令；章节标题：Rangpur G2 and G3 command handoff; Transfer the G2 slice audit script; Two-case smoke run on a CPU node; Full Train run after the smoke check; G3 existing-pipeline smoke run |
| `code/docs/SOURCE_DECISIONS.md` | Paper sources and implementation deviations; Sections: Paper and assignment decision register; Imbalance methods in the three primary papers | 论文来源与实现差异；章节标题：Paper and assignment decision register; Imbalance methods in the three primary papers |
| `code/docs/STUDY_PROTOCOL.md` | Research hypotheses, endpoints, and freeze rules; Sections: HipMRI research protocol — data-first revision; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints; Assessment and independent-review update — 20 October 2026 | 研究假设、终点指标与冻结规则；章节标题：HipMRI research protocol — data-first revision; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints; Assessment and independent-review update — 20 October 2026 |
| `code/hipmri_rewrite/__init__.py` | Six-class package constants; Definitions: module-level code | 六类别包常量；定义：module-level code |
| `code/hipmri_rewrite/audit.py` | Train/Validation-only real NIfTI preflight; Definitions: audit, main | 仅针对训练与验证数据的真实 NIfTI 预检；定义：audit, main |
| `code/hipmri_rewrite/boundaries.py` | Physical base/apex, slice and surface error analysis; Definitions: prostate_axial_regions, slice_dice, symmetric_surface_distances, prostate_organ_confusion | 物理位置的基底/尖端、切片和表面误差分析；定义：prostate_axial_regions, slice_dice, symmetric_surface_distances, prostate_organ_confusion |
| `code/hipmri_rewrite/data.py` | Paired loading, common-grid downsampling and mask encoding; Definitions: Volume, load_case, downsample, zscore_nonzero, one_hot, to_3d_model_arrays, from_3d_model_prediction | 配对读取、共用网格降采样及标签编码；定义：Volume, load_case, downsample, zscore_nonzero, one_hot, to_3d_model_arrays, from_3d_model_prediction |
| `code/hipmri_rewrite/inference.py` | 2D/3D predictions restored to the input XYZ grid; Definitions: predict_volume | 将 2D/3D 预测还原至输入 XYZ 网格；定义：predict_volume |
| `code/hipmri_rewrite/losses.py` | Six-class cross-entropy and soft-Dice objectives; Definitions: _check, foreground_soft_dice_loss, ce_dice_loss | 六类别交叉熵与 Soft Dice 目标函数；定义：_check, foreground_soft_dice_loss, ce_dice_loss |
| `code/hipmri_rewrite/metrics.py` | Confusion counts, Dice, IoU, precision and recall; Definitions: confusion, overlap | 混淆计数、Dice、IoU、精确率与召回率；定义：confusion, overlap |
| `code/hipmri_rewrite/models/__init__.py` | Six-class package constants; Definitions: module-level code | 六类别包常量；定义：module-level code |
| `code/hipmri_rewrite/models/baseline_2d.py` | Slice-wise 2D U-Net baseline; Definitions: UNet2D | 逐切片 2D U-Net 基线；定义：UNet2D |
| `code/hipmri_rewrite/models/baseline_3d.py` | Standard volumetric 3D U-Net baseline; Definitions: UNet3D | 标准体数据 3D U-Net 基线；定义：UNet3D |
| `code/hipmri_rewrite/models/common.py` | Shared standard U-Net convolutional blocks; Definitions: _DoubleConv, _StandardUNet | 标准 U-Net 共用卷积模块；定义：_DoubleConv, _StandardUNet |
| `code/hipmri_rewrite/models/improved_3d.py` | Residual 3D U-Net with deep supervision; Definitions: _ContextResidual, _Localization, ImprovedUNet3D | 含残差结构及深监督的 3D U-Net；定义：_ContextResidual, _Localization, ImprovedUNet3D |
| `code/hipmri_rewrite/profile.py` | Train-only GPU forward/backward capacity probe; Definitions: profile, main | 仅使用训练数据的 GPU 前反向容量检查；定义：profile, main |
| `code/hipmri_rewrite/sampling.py` | Patient-balanced Train patch selection and exact label exposure; Definitions: _rng, epoch_cases, crop_origin, crop_for_3d_model, ExposureRecorder | 按患者均衡抽取训练 Patch 并记录实际标签曝光；定义：_rng, epoch_cases, crop_origin, crop_for_3d_model, ExposureRecorder |
| `code/hipmri_rewrite/sampling_2d.py` | Deterministic Train-only axial crop selection for the optional 2D arm; Definitions: slice_crop_origin, crop_for_2d_model | 可选 2D 模型使用的确定性训练集轴向切片裁剪；定义：slice_crop_origin, crop_for_2d_model |
| `code/hipmri_rewrite/sampling_edges.py` | Candidate Train corner crops with physical hard negatives and uniform fallback; Definitions: boundary_hits, EdgeSelection, EdgePatchSelector | 候选训练集角落裁剪，含物理距离负样本及均匀回退；定义：boundary_hits, EdgeSelection, EdgePatchSelector |
| `code/hipmri_rewrite/splits.py` | Frozen patient identity and MRI/mask file pairing; Definitions: Case, read_split, discover | 固定患者身份及 MRI/mask 文件配对；定义：Case, read_split, discover |
| `code/hipmri_rewrite/training.py` | Predeclared equal-patient Validation selection score; Definitions: patient_small_organ_score | 预先定义患者等权的验证集模型选择指标；定义：patient_small_organ_score |
| `code/README.md` | Entry point or directory guide; Sections: Code and Methods | 项目入口或目录说明；章节标题：Code and Methods |
| `code/requirements.txt` | Python dependency ranges; UTF-8 text with 5 lines; 79 bytes | Python 依赖版本范围；UTF-8 文本，5 行；79 字节 |
| `code/tests/test_boundaries.py` | Physical direction and empty-surface tests; Definitions: BoundaryTests, SurfaceTests | 物理方向及空表面测试；定义：BoundaryTests, SurfaceTests |
| `code/tests/test_data.py` | NIfTI alignment, downsampling and label tests; Definitions: DataContractTests, DownsamplingGeometryTests, NiftiPairTests | NIfTI 对齐、降采样与标签测试；定义：DataContractTests, DownsamplingGeometryTests, NiftiPairTests |
| `code/tests/test_g5_candidate.py` | Check paired G5 candidate budget and Train-only weight provenance; Definitions: G5CandidateTests | 核验 G5 双模型候选预算及仅训练集权重来源；定义：G5CandidateTests |
| `code/tests/test_metrics.py` | Class-confusion and overlap metric tests; Definitions: OverlapTests | 类别混淆与重叠指标测试；定义：OverlapTests |
| `code/tests/test_models_and_losses.py` | Shape, gradient, supervision and inference tests; Definitions: NetworkTests | 形状、梯度、深监督与推理测试；定义：NetworkTests |
| `code/tests/test_raw_audit.py` | Train inventory and file-hash audit tests; Definitions: RawAuditInventoryTests | 训练集文件盘点与文件哈希审计测试；定义：RawAuditInventoryTests |
| `code/tests/test_sampling.py` | Patient cycle, paired crop and exposure-record tests; Definitions: case, PatientBalancedSamplingTests | 患者轮换、配对裁剪及曝光记录测试；定义：case, PatientBalancedSamplingTests |
| `code/tests/test_sampling_2d.py` | 2D crop target, bounds and MRI/mask alignment checks; Definitions: TwoDSamplingTests | 核验 2D 裁剪目标、边界和影像标签对齐；定义：TwoDSamplingTests |
| `code/tests/test_sampling_audit.py` | Candidate crop bounds and repeatability checks; Definitions: CandidateCropTests | 候选裁剪边界与可重复性检查；定义：CandidateCropTests |
| `code/tests/test_sampling_edges.py` | Verify physical negatives, paired geometry, Train isolation and uniform fallback; Definitions: volume, selector, EdgeSamplingTests | 核验物理距离负样本、配对几何、训练集隔离及均匀回退；定义：volume, selector, EdgeSamplingTests |
| `code/tests/test_splits.py` | Patient isolation and file pairing tests; Definitions: PatientSplitTests | 患者隔离与文件配对测试；定义：PatientSplitTests |
| `code/tests/test_training_recovery.py` | Check atomic batch checkpoints and recovery after an interrupted epoch pointer; Definitions: RecoveryTests | 核验原子批次权重及轮次指针中断后的恢复；定义：RecoveryTests |
| `code/tests/test_training_selection.py` | Check patient-balanced small-organ checkpoint selection; Definitions: PatientSelectionTests | 核验患者等权的小器官权重选择规则；定义：PatientSelectionTests |
| `code/tools/audit_g2_train_slices.py` | Measure Train axial label coverage and MRI intensity bins on Rangpur CPU; Definitions: sha256_file, write_csv, main | 在 Rangpur CPU 测量训练集逐切片标签覆盖与 MRI 强度分箱；定义：sha256_file, write_csv, main |
| `code/tools/audit_g3_candidate_sampling.py` | Compare fixed Train-only uniform and organ-aware candidate crops; Definitions: sha256_file, fixed_rng, uniform_origin, target_origin, patch_counts, main | 比较固定的训练集均匀与器官感知候选裁剪；定义：sha256_file, fixed_rng, uniform_origin, target_origin, patch_counts, main |
| `code/tools/audit_g3_existing_pipeline.py` | Audit current Train downsampling, normalization and whole-volume arrays; Definitions: sha256_file, class_counts, occupied_z_slices, software_versions, main | 审计当前训练集降采样、归一化与整卷张量；定义：sha256_file, class_counts, occupied_z_slices, software_versions, main |
| `code/tools/audit_g4_patient_sampling.py` | Dry-run patient-balanced model-ready Train patches on CPU; Definitions: sha256_file, main | 在 CPU 上预演患者均衡的训练模型输入 Patch；定义：sha256_file, main |
| `code/tools/audit_g5_2d_sampling.py` | CPU audit of patient-balanced uniform and organ-aware 2D slice exposure; Definitions: sha256_file, main | 在 CPU 审计按患者均衡的 2D 均匀及器官感知切片曝光；定义：sha256_file, main |
| `code/tools/audit_g5_aux_logits_validation.py` | Compare selected Improved 3D summed and main-only logits on Validation; Definitions: predictions, main | 在验证集比较选定 Improved 3D 的相加输出和单独主输出；定义：predictions, main |
| `code/tools/audit_g5_train_patch_geometry.py` | Reconstruct frozen 3D Train patch face and corner coverage; Definitions: digest, touches, main | 重建固定方案的 3D 训练 Patch 边缘及角落覆盖；定义：digest, touches, main |
| `code/tools/audit_g5_validation_body.py` | Measure fixed-checkpoint Validation Body boundaries and draw outliers; Definitions: draw, main | 测量固定权重的验证集 Body 边界并绘制异常案例；定义：draw, main |
| `code/tools/audit_g7_edge_sampling.py` | Replay frozen Train epochs and compare 9+1 edge sampling exposure on CPU; Definitions: sha, main | 在 CPU 重放固定训练轮次并比较 9+1 边缘采样曝光；定义：sha, main |
| `code/tools/audit_train_raw.py` | Train-only raw MRI/mask integrity audit; Definitions: train_inventory, sha256_file, audit_case, main | 仅对训练集原始 MRI/Mask 做完整性审计；定义：train_inventory, sha256_file, audit_case, main |
| `code/tools/evaluate_g6_test.py` | Definitions: sha256, selected_model, boundary_summary, main | 定义：sha256, selected_model, boundary_summary, main |
| `code/tools/package_transfer.py` | Build and verify a source-only transfer ZIP; Definitions: main | 创建并校验仅含源码的传输包；定义：main |
| `code/tools/probe_g5_full_volume.py` | Measure full-volume Train inference geometry, time and A100 memory with random weights; Definitions: sha256_file, main | 用随机权重测量完整训练体积推理的几何、时间及 A100 显存；定义：sha256_file, main |
| `code/tools/probe_g5_patch_gpu.py` | Measure both 3D models' real Train patch gradients and A100 memory; Definitions: sha256_file, main | 测量两个 3D 模型真实训练 Patch 的梯度和 A100 显存；定义：sha256_file, main |
| `code/tools/probe_g5_training_smoke.py` | Check paired two-patient weighted Train batches, gradients and actual exposure without a checkpoint; Definitions: sha256_file, check_config, main | 不保存权重，检查双模型两患者加权训练批次、梯度和实际曝光；定义：sha256_file, check_config, main |
| `code/tools/review_g1_bladder_islands.py` | Locate and visualize two flagged Train Bladder components; Definitions: inspect_case, main | 定位并可视化两例训练集膀胱小连通区域；定义：inspect_case, main |
| `code/tools/review_g1_overlays.py` | Render raw Train MRI and ground-truth overlays for G1 review; Definitions: render_case, main | 绘制训练集原始 MRI 与真实标签叠加图供 G1 审查；定义：render_case, main |
| `code/tools/review_g1_train_contact_sheets.py` | Render two-slice Train patient sheets for G1 alignment screening; Definitions: render_patient, main | 绘制训练集患者双切面图供 G1 对齐筛查；定义：render_patient, main |
| `code/tools/review_g5_validation_failures.py` | Definitions: sha256, load_selected, selected_slice, render, main | 定义：sha256, load_selected, selected_slice, render, main |
| `code/tools/review_g6_body_boundaries.py` | Reproduce selected Test cases and draw Body boundary errors; Definitions: planes, draw, main | 复现选定测试案例并绘制 Body 边界错误；定义：planes, draw, main |
| `code/tools/review_g6_test_cases.py` | Reproduce selected held-out Test 3D predictions and draw descriptive overlays; Definitions: axial_slice, render, main | 复现选定独立测试样本的 3D 预测并绘制描述性叠加图；定义：axial_slice, render, main |
| `code/tools/submit_g5_2d.sh` | UTF-8 text with 77 lines; 3,122 bytes | UTF-8 文本，77 行；3,122 字节 |
| `code/tools/submit_g5_2d_extend_50.sh` | Verify and queue a frozen 2D continuation to epoch 50; UTF-8 text with 81 lines; 2,914 bytes | 核验并排队固定配置的 2D 模型续训至 50 轮；UTF-8 文本，81 行；2,914 字节 |
| `code/tools/submit_g5_40.sh` | Submit CPU recovery check and dependency-gated Standard/Improved 40-epoch A100 jobs; UTF-8 text with 95 lines; 3,818 bytes | 提交 CPU 恢复测试及依赖测试通过的双模型 40 轮 A100 作业；UTF-8 文本，95 行；3,818 字节 |
| `code/tools/submit_g5_extend_50.sh` | UTF-8 text with 109 lines; 4,866 bytes | UTF-8 文本，109 行；4,866 字节 |
| `code/tools/submit_g6_test.sh` | UTF-8 text with 35 lines; 1,550 bytes | UTF-8 文本，35 行；1,550 字节 |
| `code/tools/submit_g7_edge_audit.sh` | Submit smoke or smoke-gated full Train-only CPU edge audit; UTF-8 text with 66 lines; 2,607 bytes | 提交小规模或通过小规模门槛的完整训练集 CPU 边缘审计；UTF-8 文本，66 行；2,607 字节 |
| `code/tools/summarize_all_class_review.py` | Verify selected records and draw six-class, training and failure figures; Definitions: save, paired_bars, validation_figures, test_figures_and_table, three_arm_validation_figures, main | 核验选定记录并绘制六类、训练及失误图；定义：save, paired_bars, validation_figures, test_figures_and_table, three_arm_validation_figures, main |
| `code/tools/summarize_g2_slice_detail.py` | Reconcile G2 slice and intensity CSVs with G1 and draw local plots; Definitions: read_csv, write_csv, check_hash, main | 将 G2 切片及强度 CSV 与 G1 核对并绘制本地图；定义：read_csv, write_csv, check_hash, main |
| `code/tools/summarize_g2_train_audit.py` | Summarize verified raw Train audit into G2 tables and plots; Definitions: write_csv, main | 将已核验的训练集原始审计汇总为 G2 表格与图；定义：write_csv, main |
| `code/tools/summarize_g3_candidate_sampling.py` | Verify full candidate patch records and summarize exposure; Definitions: sha256_file, write_csv, main | 核验完整候选 Patch 记录并汇总曝光；定义：sha256_file, write_csv, main |
| `code/tools/summarize_g3_pipeline.py` | Definitions: verified_json, main | 定义：verified_json, main |
| `code/tools/summarize_g4_patient_sampling.py` | Verify G4 tensor dry run and summarize class/patient exposure; Definitions: sha256_file, write_csv, main | 核验 G4 张量预演并汇总类别及患者曝光；定义：sha256_file, write_csv, main |
| `code/tools/summarize_g5_2d_validation.py` | Definitions: sha256, main | 定义：sha256, main |
| `code/tools/summarize_g5_body_review.py` | Verify Validation Body audit and plot patient boundary errors; Definitions: main | 核验验证集 Body 审计并绘制患者边界错误；定义：main |
| `code/tools/summarize_g5_validation.py` | Definitions: sha256, same_metric, load_checked_run, main | 定义：sha256, same_metric, load_checked_run, main |
| `code/tools/summarize_g6_test.py` | Verify one-time held-out Test metrics; Definitions: sha256, same, check, main | 核验一次性独立测试指标；定义：sha256, same, check, main |
| `code/tools/summarize_g7_edge_sampling.py` | Verify paired G7 crop CSV against 50 training epochs and draw six-class spatial exposure; Definitions: sha, save, main | 对照 50 轮训练核验 G7 配对裁剪 CSV 并绘制六类空间曝光；定义：sha, save, main |
| `code/tools/tiny_overfit_g5.py` | Check both 3D models can learn two fixed Train patches on GPU; Definitions: sha256_file, measure, main | 检查两个 3D 模型能否在 GPU 上学习两个固定训练 Patch；定义：sha256_file, measure, main |
| `code/tools/train_g5_2d.py` | Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main | 定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main |
| `code/tools/train_g5_cohort.py` | Train a frozen 3D arm with patient-balanced exposure, Validation selection and resumable checkpoints; Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main | 按固定方案训练 3D 模型，记录患者均衡曝光、验证集选择及可恢复权重；定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main |
| `code/tools/update_project_map.py` | Generate and watch this bilingual inventory; Definitions: _files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main | 生成并持续更新本双语目录；定义：_files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main |
| `code/transfer/20261017_source.zip` | Generated source-only transfer package; ZIP archive with 66 members; 52,512 bytes | 生成的仅含源码传输包；ZIP 压缩包，含 66 个成员；52,512 字节 |
| `code/transfer/g7_edge_audit_20261019.zip` | Five-file source-only G7 audit transfer package; ZIP archive with 5 members; 10,348 bytes | 仅含五个 G7 审计源码文件的传输包；ZIP 压缩包，含 5 个成员；10,348 字节 |
| `logs/audits/g1_train_contact_sheets_20261018.json` | JSON fields: scope, screen, patients | JSON 字段：scope, screen, patients |
| `logs/audits/g3_existing_pipeline_full_20261018_job671892.json` | Binary or large file; 299,809 bytes | 二进制或大文件；299,809 字节 |
| `logs/audits/g3_patch_candidate_full_20261018/manifest.json` | JSON fields: audit_script_sha256, config, config_sha256, data_code_sha256, full_train, g1_sha256, g3_sha256, patients, scope, split_code_sha256, summary, train_cases | JSON 字段：audit_script_sha256, config, config_sha256, data_code_sha256, full_train, g1_sha256, g3_sha256, patients, scope, split_code_sha256, summary, train_cases |
| `logs/audits/g3_patch_candidate_full_20261018/patches.csv` | Binary or large file; 248,893 bytes | 二进制或大文件；248,893 字节 |
| `logs/audits/g4_patient_full_20261018/manifest.json` | JSON fields: complete_patient_epoch, data_code_sha256, epoch, g1_sha256, g3_sha256, patch_shape_xyz, patches_per_policy, patients, policies, sampling_code_sha256, scope, script_sha256 | JSON 字段：complete_patient_epoch, data_code_sha256, epoch, g1_sha256, g3_sha256, patch_shape_xyz, patches_per_policy, patients, policies, sampling_code_sha256, scope, script_sha256 |
| `logs/audits/g4_patient_full_20261018/patches.csv` | Table with 520 data rows; columns: ﻿epoch, policy, patient, case, draw, target, origin_x, origin_y, origin_z, background_voxels, body_voxels, bones_voxels | 表格含 520 条数据行；列：﻿epoch, policy, patient, case, draw, target, origin_x, origin_y, origin_z, background_voxels, body_voxels, bones_voxels |
| `logs/audits/g5_2d_sampling_full_20261018/manifest.json` | JSON fields: complete_patient_epoch, data_sha256, epoch, g1_sha256, g3_sha256, patch_xy, patches_per_policy, patients, policies, sampling_2d_sha256, scope, script_sha256 | JSON 字段：complete_patient_epoch, data_sha256, epoch, g1_sha256, g3_sha256, patch_xy, patches_per_policy, patients, policies, sampling_2d_sha256, scope, script_sha256 |
| `logs/audits/g5_2d_sampling_full_20261018/patches.csv` | Table with 520 data rows; columns: ﻿epoch, policy, patient, case, draw, target, origin_x, origin_y, slice_z, background_voxels, body_voxels, bones_voxels | 表格含 520 条数据行；列：﻿epoch, policy, patient, case, draw, target, origin_x, origin_y, slice_z, background_voxels, body_voxels, bones_voxels |
| `logs/audits/g7_edge_sampling_full_20261019/manifest.json` | Binary or large file; 856,983 bytes | 二进制或大文件；856,983 字节 |
| `logs/audits/g7_edge_sampling_full_20261019/patches.csv` | Binary or large file; 4,235,680 bytes | 二进制或大文件；4,235,680 字节 |
| `logs/audits/train_full_20261018_job671208.json` | Binary or large file; 563,622 bytes | 二进制或大文件；563,622 字节 |
| `logs/internal/map_watcher_stderr.txt` | UTF-8 text with 0 lines; 0 bytes | UTF-8 文本，0 行；0 字节 |
| `logs/internal/map_watcher_stdout.txt` | UTF-8 text with 0 lines; 0 bytes | UTF-8 文本，0 行；0 字节 |
| `logs/internal/project_map_watcher.pid` | UTF-8 text with 1 lines; 7 bytes | UTF-8 文本，1 行；7 字节 |
| `logs/profiles/g5_full_volume_probe_20261018.json` | JSON fields: case, data_code_sha256, g1_sha256, gpu, gpu_total_bytes, inference_code_sha256, job_id, kind, mask_sha256, mri_sha256, numpy, padded_input_shape_dhw | JSON 字段：case, data_code_sha256, g1_sha256, gpu, gpu_total_bytes, inference_code_sha256, job_id, kind, mask_sha256, mri_sha256, numpy, padded_input_shape_dhw |
| `logs/profiles/g5_patch_probe_20261018.json` | JSON fields: batch_size, case, cuda_build, data_code_sha256, gpu, gpu_total_bytes, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy | JSON 字段：batch_size, case, cuda_build, data_code_sha256, gpu, gpu_total_bytes, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy |
| `logs/profiles/g5_standard_500_20261018.json` | JSON fields: case, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy | JSON 字段：case, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy |
| `logs/profiles/g5_standard_weighted_500_20261018.json` | JSON fields: case, class_weights, class_weights_config_sha256, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256 | JSON 字段：case, class_weights, class_weights_config_sha256, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256 |
| `logs/profiles/g5_tiny_overfit_20261018.json` | JSON fields: case, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy | JSON 字段：case, cuda_build, data_code_sha256, g1_sha256, gpu, input_shape, job_id, kind, loss_code_sha256, mask_sha256, mri_sha256, numpy |
| `logs/profiles/g5_training_smoke_20261018.json` | JSON fields: cases, config_sha256, data_code_sha256, g1_sha256, gpu, job_id, kind, loss_code_sha256, numpy, patches, results, sampling_code_sha256 | JSON 字段：cases, config_sha256, data_code_sha256, g1_sha256, gpu, job_id, kind, loss_code_sha256, numpy, patches, results, sampling_code_sha256 |
| `logs/README.md` | Entry point or directory guide; Sections: Experiment Logs | 项目入口或目录说明；章节标题：Experiment Logs |
| `logs/training/baseline_2d/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_005.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_010.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_015.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_020.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_025.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_030.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_035.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_040.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_045.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/epoch_050.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_2d/manifest.json` | JSON fields: config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 | JSON 字段：config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 |
| `logs/training/baseline_3d/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_005.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_010.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_015.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_020.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_025.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_030.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_035.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_040.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_045.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/epoch_050.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/baseline_3d/manifest.json` | JSON fields: config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 | JSON 字段：config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 |
| `logs/training/improved_3d/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_005.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_010.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_015.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_020.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_025.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_030.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_035.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_040.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_045.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/epoch_050.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `logs/training/improved_3d/manifest.json` | JSON fields: config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 | JSON 字段：config_sha256, data_sha256, dataset_root, extensions, g1_sha256, gpu, inference_sha256, job_id_initial, kind, loss_sha256, max_epochs, metrics_sha256 |
| `PROJECT_MAP.md` | Generated bilingual tree and file content index; Self-entry; regenerated from current tree and file structure | 自动生成的双语文件树与内容索引；本文件条目；根据当前文件树和文件结构重新生成 |
| `README.md` | Entry point or directory guide; Sections: HipMRI Data-Driven Model Development; Where things belong | 项目入口或目录说明；章节标题：HipMRI Data-Driven Model Development; Where things belong |
| `results/case_studies/comparisons/g5_body_validation_B040_Week4.png` | Binary or large file; 268,835 bytes | 二进制或大文件；268,835 字节 |
| `results/case_studies/comparisons/g5_body_validation_J026_Week4.png` | Binary or large file; 222,090 bytes | 二进制或大文件；222,090 字节 |
| `results/case_studies/comparisons/g5_body_validation_K018_Week4.png` | Binary or large file; 249,121 bytes | 二进制或大文件；249,121 字节 |
| `results/case_studies/comparisons/g5_validation_B040_Week1_prostate.png` | Binary or large file; 99,219 bytes | 二进制或大文件；99,219 字节 |
| `results/case_studies/comparisons/g5_validation_J026_Week1_rectum.png` | Binary or large file; 100,764 bytes | 二进制或大文件；100,764 字节 |
| `results/case_studies/comparisons/g5_validation_K018_Week3_rectum.png` | Binary or large file; 89,639 bytes | 二进制或大文件；89,639 字节 |
| `results/case_studies/comparisons/g5_validation_S022_Week6_prostate.png` | Binary or large file; 89,736 bytes | 二进制或大文件；89,736 字节 |
| `results/case_studies/comparisons/g5_validation_selected_cases.json` | JSON fields: kind, split, cases, sources | JSON 字段：kind, split, cases, sources |
| `results/case_studies/comparisons/g6_body_L011_Week5.png` | Binary or large file; 218,157 bytes | 二进制或大文件；218,157 字节 |
| `results/case_studies/comparisons/g6_body_M015_Week3.png` | Binary or large file; 266,760 bytes | 二进制或大文件；266,760 字节 |
| `results/case_studies/comparisons/g6_body_selected_cases.json` | JSON fields: kind, split, test_result_sha256, cases | JSON 字段：kind, split, test_result_sha256, cases |
| `results/case_studies/comparisons/g6_body_W041_Week0.png` | Binary or large file; 244,148 bytes | 二进制或大文件；244,148 字节 |
| `results/case_studies/comparisons/g6_test_M015_Week7_bladder.png` | Binary or large file; 130,570 bytes | 二进制或大文件；130,570 字节 |
| `results/case_studies/comparisons/g6_test_M020_Week1_prostate.png` | Binary or large file; 94,741 bytes | 二进制或大文件；94,741 字节 |
| `results/case_studies/comparisons/g6_test_selected_cases.json` | JSON fields: kind, split, test_result_sha256, cases | JSON 字段：kind, split, test_result_sha256, cases |
| `results/case_studies/comparisons/g6_test_W029_Week6_prostate.png` | Binary or large file; 83,023 bytes | 二进制或大文件；83,023 字节 |
| `results/case_studies/comparisons/README.md` | Entry point or directory guide; Sections: Selected segmentation overlays | 项目入口或目录说明；章节标题：Selected segmentation overlays |
| `results/data_analysis/g1_bladder_review/g1_bladder_island_D031_Week7.png` | Binary or large file; 250,160 bytes | 二进制或大文件；250,160 字节 |
| `results/data_analysis/g1_bladder_review/g1_bladder_island_N010_Week3.png` | Binary or large file; 254,248 bytes | 二进制或大文件；254,248 字节 |
| `results/data_analysis/g1_bladder_review/README.md` | Entry point or directory guide; Sections: G1 Bladder component review | 项目入口或目录说明；章节标题：G1 Bladder component review |
| `results/data_analysis/g1_patient_screen/g1_train_patient_B006.png` | Binary or large file; 263,728 bytes | 二进制或大文件；263,728 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_B038.png` | Binary or large file; 134,754 bytes | 二进制或大文件；134,754 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_C032.png` | Binary or large file; 1,001,604 bytes | 二进制或大文件；1,001,604 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_D031.png` | Binary or large file; 979,042 bytes | 二进制或大文件；979,042 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_G021.png` | Binary or large file; 794,896 bytes | 二进制或大文件；794,896 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_H007.png` | Binary or large file; 108,239 bytes | 二进制或大文件；108,239 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_H017.png` | Binary or large file; 895,730 bytes | 二进制或大文件；895,730 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_J005.png` | Binary or large file; 568,074 bytes | 二进制或大文件；568,074 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_K019.png` | Binary or large file; 1,185,795 bytes | 二进制或大文件；1,185,795 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_K042.png` | Binary or large file; 128,697 bytes | 二进制或大文件；128,697 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_M004.png` | Binary or large file; 1,218,442 bytes | 二进制或大文件；1,218,442 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_M013.png` | Binary or large file; 989,753 bytes | 二进制或大文件；989,753 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_M023.png` | Binary or large file; 123,435 bytes | 二进制或大文件；123,435 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_M036.png` | Binary or large file; 123,922 bytes | 二进制或大文件；123,922 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_N010.png` | Binary or large file; 1,179,116 bytes | 二进制或大文件；1,179,116 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_O025.png` | Binary or large file; 1,050,411 bytes | 二进制或大文件；1,050,411 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_R016.png` | Binary or large file; 1,037,894 bytes | 二进制或大文件；1,037,894 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_R024.png` | Binary or large file; 1,151,048 bytes | 二进制或大文件；1,151,048 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_R039.png` | Binary or large file; 1,029,032 bytes | 二进制或大文件；1,029,032 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_S028.png` | Binary or large file; 135,808 bytes | 二进制或大文件；135,808 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_S033.png` | Binary or large file; 1,144,701 bytes | 二进制或大文件；1,144,701 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_S035.png` | Binary or large file; 107,138 bytes | 二进制或大文件；107,138 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_T009.png` | Binary or large file; 120,824 bytes | 二进制或大文件；120,824 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_T014.png` | Binary or large file; 996,731 bytes | 二进制或大文件；996,731 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_V027.png` | Binary or large file; 887,276 bytes | 二进制或大文件；887,276 字节 |
| `results/data_analysis/g1_patient_screen/g1_train_patient_W012.png` | Binary or large file; 1,046,773 bytes | 二进制或大文件；1,046,773 字节 |
| `results/data_analysis/g1_patient_screen/README.md` | Entry point or directory guide; Sections: G1 full Train patient screen | 项目入口或目录说明；章节标题：G1 full Train patient screen |
| `results/data_analysis/g1_selected_cases/g1_overlay_B038_Week0.png` | Binary or large file; 477,710 bytes | 二进制或大文件；477,710 字节 |
| `results/data_analysis/g1_selected_cases/g1_overlay_D031_Week7.png` | Binary or large file; 393,632 bytes | 二进制或大文件；393,632 字节 |
| `results/data_analysis/g1_selected_cases/g1_overlay_H017_Week7.png` | Binary or large file; 365,557 bytes | 二进制或大文件；365,557 字节 |
| `results/data_analysis/g1_selected_cases/g1_overlay_K019_Week1.png` | Binary or large file; 471,124 bytes | 二进制或大文件；471,124 字节 |
| `results/data_analysis/g1_selected_cases/g1_overlay_N010_Week3.png` | Binary or large file; 453,764 bytes | 二进制或大文件；453,764 字节 |
| `results/data_analysis/g1_selected_cases/g1_overlay_S033_Week1.png` | Binary or large file; 443,971 bytes | 二进制或大文件；443,971 字节 |
| `results/data_analysis/g1_selected_cases/README.md` | Entry point or directory guide; Sections: G1 selected raw MRI overlays | 项目入口或目录说明；章节标题：G1 selected raw MRI overlays |
| `results/data_analysis/g2_train_summary/class_distribution.csv` | Table with 6 data rows; columns: ﻿class, voxel_count, share_pct, denominator_voxels, train_volumes | 表格含 6 条数据行；列：﻿class, voxel_count, share_pct, denominator_voxels, train_volumes |
| `results/data_analysis/g2_train_summary/class_frequency_log.svg` | UTF-8 text with 544 lines; 20,424 bytes | UTF-8 文本，544 行；20,424 字节 |
| `results/data_analysis/g2_train_summary/intensity_case_summary.svg` | UTF-8 text with 675 lines; 28,992 bytes | UTF-8 文本，675 行；28,992 字节 |
| `results/data_analysis/g2_train_summary/intensity_statistics.csv` | Table with 143 data rows; columns: ﻿case, patient, finite_min, finite_median, finite_max, zero_fraction, nonzero_mean, nonzero_std, nonzero_p0_5, nonzero_p1, nonzero_p50, nonzero_p99 | 表格含 143 条数据行；列：﻿case, patient, finite_min, finite_median, finite_max, zero_fraction, nonzero_mean, nonzero_std, nonzero_p0_5, nonzero_p1, nonzero_p50, nonzero_p99 |
| `results/data_analysis/g2_train_summary/longitudinal_organ_variation.csv` | Table with 26 data rows; columns: ﻿patient, volume_count, bladder_min_ml, bladder_max_ml, bladder_max_min_ratio, rectum_min_ml, rectum_max_ml, rectum_max_min_ratio, prostate_min_ml, prostate_max_ml, prostate_max_min_ratio | 表格含 26 条数据行；列：﻿patient, volume_count, bladder_min_ml, bladder_max_ml, bladder_max_min_ratio, rectum_min_ml, rectum_max_ml, rectum_max_min_ratio, prostate_min_ml, prostate_max_ml, prostate_max_min_ratio |
| `results/data_analysis/g2_train_summary/organ_bbox_centers.svg` | UTF-8 text with 1210 lines; 95,981 bytes | UTF-8 文本，1210 行；95,981 字节 |
| `results/data_analysis/g2_train_summary/organ_slices_per_volume.svg` | UTF-8 text with 679 lines; 23,727 bytes | UTF-8 文本，679 行；23,727 字节 |
| `results/data_analysis/g2_train_summary/patient_rare_organ_share.svg` | UTF-8 text with 657 lines; 34,529 bytes | UTF-8 文本，657 行；34,529 字节 |
| `results/data_analysis/g2_train_summary/per_patient_distribution.csv` | Table with 26 data rows; columns: ﻿patient, volume_count, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct, rectum_voxels | 表格含 26 条数据行；列：﻿patient, volume_count, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct, rectum_voxels |
| `results/data_analysis/g2_train_summary/per_slice_class_coverage.csv` | Binary or large file; 1,494,705 bytes | 二进制或大文件；1,494,705 字节 |
| `results/data_analysis/g2_train_summary/per_volume_distribution.csv` | Table with 143 data rows; columns: ﻿case, patient, visit, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct | 表格含 143 条数据行；列：﻿case, patient, visit, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct |
| `results/data_analysis/g2_train_summary/per_volume_intensity_histogram.csv` | Binary or large file; 491,370 bytes | 二进制或大文件；491,370 字节 |
| `results/data_analysis/g2_train_summary/per_volume_slice_coverage.csv` | Table with 143 data rows; columns: ﻿case, patient, axial_slices, background_present_slices, background_slice_hit_pct, body_present_slices, body_slice_hit_pct, bones_present_slices, bones_slice_hit_pct, bladder_present_slices, bladder_slice_hit_pct, rectum_present_slices | 表格含 143 条数据行；列：﻿case, patient, axial_slices, background_present_slices, background_slice_hit_pct, body_present_slices, body_slice_hit_pct, bones_present_slices, bones_slice_hit_pct, bladder_present_slices, bladder_slice_hit_pct, rectum_present_slices |
| `results/data_analysis/g2_train_summary/pooled_positive_intensity_histogram.csv` | Table with 120 data rows; columns: ﻿lower_inclusive, upper_exclusive, positive_voxels | 表格含 120 条数据行；列：﻿lower_inclusive, upper_exclusive, positive_voxels |
| `results/data_analysis/g2_train_summary/pooled_positive_intensity_histogram.svg` | UTF-8 text with 298 lines; 13,306 bytes | UTF-8 文本，298 行；13,306 字节 |
| `results/data_analysis/g2_train_summary/raw_axial_slice_hit_rate.svg` | UTF-8 text with 262 lines; 12,836 bytes | UTF-8 文本，262 行；12,836 字节 |
| `results/data_analysis/g2_train_summary/README.md` | Entry point or directory guide; Sections: G2 raw Train summary | 项目入口或目录说明；章节标题：G2 raw Train summary |
| `results/data_analysis/g2_train_summary/slice_presence_summary.csv` | Table with 6 data rows; columns: ﻿class, present_slices, total_slices, slice_hit_pct, median_present_slices_per_volume, min_present_slices_per_volume, max_present_slices_per_volume | 表格含 6 条数据行；列：﻿class, present_slices, total_slices, slice_hit_pct, median_present_slices_per_volume, min_present_slices_per_volume, max_present_slices_per_volume |
| `results/data_analysis/g2_train_summary/spatial_statistics.csv` | Table with 143 data rows; columns: ﻿case, patient, shape_x, shape_y, shape_z, spacing_x_mm, spacing_y_mm, spacing_z_mm, voxel_volume_mm3, orientation, bladder_volume_ml, bladder_bbox_min_x_voxel | 表格含 143 条数据行；列：﻿case, patient, shape_x, shape_y, shape_z, spacing_x_mm, spacing_y_mm, spacing_z_mm, voxel_volume_mm3, orientation, bladder_volume_ml, bladder_bbox_min_x_voxel |
| `results/data_analysis/g2_train_summary/voxel_spacing.svg` | UTF-8 text with 372 lines; 32,876 bytes | UTF-8 文本，372 行；32,876 字节 |
| `results/data_analysis/g3_pipeline/candidate_patch_class_exposure.csv` | Table with 12 data rows; columns: ﻿arm, class, patches, patch_hits, patch_hit_pct, patch_voxels, voxel_share_pct, complete_organ_patches | 表格含 12 条数据行；列：﻿arm, class, patches, patch_hits, patch_hit_pct, patch_voxels, voxel_share_pct, complete_organ_patches |
| `results/data_analysis/g3_pipeline/candidate_patch_exposure.svg` | UTF-8 text with 1709 lines; 53,196 bytes | UTF-8 文本，1709 行；53,196 字节 |
| `results/data_analysis/g3_pipeline/candidate_patch_patient_exposure.csv` | Table with 52 data rows; columns: ﻿patient, arm, volumes, patches, bladder_hit_pct, rectum_hit_pct, prostate_hit_pct | 表格含 52 条数据行；列：﻿patient, arm, volumes, patches, bladder_hit_pct, rectum_hit_pct, prostate_hit_pct |
| `results/data_analysis/g3_pipeline/organ_physical_volume_ratio.svg` | UTF-8 text with 1439 lines; 44,212 bytes | UTF-8 文本，1439 行；44,212 字节 |
| `results/data_analysis/g3_pipeline/organ_preservation_by_volume.csv` | Table with 429 data rows; columns: ﻿case, patient, organ, raw_voxels, downsampled_voxels, raw_axial_slices, downsampled_axial_slices, raw_physical_volume_mm3, downsampled_physical_volume_mm3, physical_volume_ratio | 表格含 429 条数据行；列：﻿case, patient, organ, raw_voxels, downsampled_voxels, raw_axial_slices, downsampled_axial_slices, raw_physical_volume_mm3, downsampled_physical_volume_mm3, physical_volume_ratio |
| `results/data_analysis/g3_pipeline/README.md` | Entry point or directory guide; Sections: G3 existing-pipeline organ preservation | 项目入口或目录说明；章节标题：G3 existing-pipeline organ preservation |
| `results/data_analysis/g4_patient_sampling/class_exposure.csv` | Table with 12 data rows; columns: ﻿policy, class, patches, patch_hits, patch_hit_rate, patch_voxels, voxel_share | 表格含 12 条数据行；列：﻿policy, class, patches, patch_hits, patch_hit_rate, patch_voxels, voxel_share |
| `results/data_analysis/g4_patient_sampling/g5_train_patch_geometry.json` | JSON fields: kind, scope, g1_sha256, config_sha256, run_manifest_sha256, definition, patch_xyz, epochs, counts, rates, by_patient | JSON 字段：kind, scope, g1_sha256, config_sha256, run_manifest_sha256, definition, patch_xyz, epochs, counts, rates, by_patient |
| `results/data_analysis/g4_patient_sampling/patient_balanced_exposure.svg` | UTF-8 text with 1714 lines; 51,622 bytes | UTF-8 文本，1714 行；51,622 字节 |
| `results/data_analysis/g4_patient_sampling/patient_exposure.csv` | Table with 52 data rows; columns: ﻿policy, patient, case, patches, bladder_hits, rectum_hits, prostate_hits, bladder_voxel_share, rectum_voxel_share, prostate_voxel_share | 表格含 52 条数据行；列：﻿policy, patient, case, patches, bladder_hits, rectum_hits, prostate_hits, bladder_voxel_share, rectum_voxel_share, prostate_voxel_share |
| `results/data_analysis/g4_patient_sampling/README.md` | Entry point or directory guide; Sections: G4 patient-balanced patch exposure | 项目入口或目录说明；章节标题：G4 patient-balanced patch exposure |
| `results/data_analysis/g7_edge_sampling/class_exposure.csv` | Table with 12 data rows; columns: ﻿policy, class, patch_hits, patch_hit_rate, patch_voxels, voxel_share | 表格含 12 条数据行；列：﻿policy, class, patch_hits, patch_hit_rate, patch_voxels, voxel_share |
| `results/data_analysis/g7_edge_sampling/g7_crop_origins.png` | Binary or large file; 214,581 bytes | 二进制或大文件；214,581 字节 |
| `results/data_analysis/g7_edge_sampling/g7_crop_origins.svg` | Binary or large file; 175,989 bytes | 二进制或大文件；175,989 字节 |
| `results/data_analysis/g7_edge_sampling/g7_exposure_and_geometry.png` | Binary or large file; 140,171 bytes | 二进制或大文件；140,171 字节 |
| `results/data_analysis/g7_edge_sampling/g7_exposure_and_geometry.svg` | UTF-8 text with 2714 lines; 90,154 bytes | UTF-8 文本，2714 行；90,154 字节 |
| `results/data_analysis/g7_edge_sampling/patient_class_exposure.csv` | Table with 312 data rows; columns: ﻿policy, patient, class, patches, patch_hits, patch_voxels, voxel_share | 表格含 312 条数据行；列：﻿policy, patient, class, patches, patch_hits, patch_voxels, voxel_share |
| `results/data_analysis/g7_edge_sampling/README.md` | Entry point or directory guide; Sections: G7：原均匀采样与 9+1 边缘采样 | 项目入口或目录说明；章节标题：G7：原均匀采样与 9+1 边缘采样 |
| `results/data_analysis/g7_edge_sampling/verification.json` | JSON fields: kind, source_hashes, csv_rows, preserved_draw_pairs, baseline_epochs_reproduced, accepted_edge_draws, eligible_under_provisional_gates, inference_or_model_quality_established | JSON 字段：kind, source_hashes, csv_rows, preserved_draw_pairs, baseline_epochs_reproduced, accepted_edge_draws, eligible_under_provisional_gates, inference_or_model_quality_established |
| `results/data_analysis/historical_train/HipMRI_Pretraining_Data_Analysis.previous-41201a5e.bak` | UTF-8 text with 106 lines; 14,485 bytes | UTF-8 文本，106 行；14,485 字节 |
| `results/data_analysis/historical_train/HipMRI_Pretraining_Data_Analysis.previous-5dc166bd.bak` | UTF-8 text with 93 lines; 12,305 bytes | UTF-8 文本，93 行；12,305 字节 |
| `results/data_analysis/historical_train/historical_patch_vs_full_volume.svg` | UTF-8 text with 49 lines; 4,617 bytes | UTF-8 文本，49 行；4,617 字节 |
| `results/data_analysis/historical_train/historical_resampling_volume_ratio.svg` | UTF-8 text with 768 lines; 59,609 bytes | UTF-8 文本，768 行；59,609 字节 |
| `results/data_analysis/historical_train/historical_train_case_profile.csv` | Table with 143 data rows; columns: case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels | 表格含 143 条数据行；列：case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels |
| `results/data_analysis/historical_train/historical_train_statistics.json` | JSON fields: scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts | JSON 字段：scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts |
| `results/data_analysis/historical_train/README.md` | Entry point or directory guide; Sections: Historical Train statistics | 项目入口或目录说明；章节标题：Historical Train statistics |
| `results/data_analysis/historical_train/train_class_distribution.svg` | UTF-8 text with 35 lines; 3,679 bytes | UTF-8 文本，35 行；3,679 字节 |
| `results/data_analysis/historical_train/train_foreground_distribution.svg` | UTF-8 text with 32 lines; 3,368 bytes | UTF-8 文本，32 行；3,368 字节 |
| `results/data_analysis/historical_train/train_patient_examinations.svg` | UTF-8 text with 72 lines; 7,370 bytes | UTF-8 文本，72 行；7,370 字节 |
| `results/data_analysis/historical_train/train_rare_organ_volumes.svg` | UTF-8 text with 320 lines; 25,203 bytes | UTF-8 文本，320 行；25,203 字节 |
| `results/data_analysis/README.md` | Entry point or directory guide; Sections: Data analysis image groups | 项目入口或目录说明；章节标题：Data analysis image groups |
| `results/figures/baseline_2d/g5_2d_validation_all_classes.png` | Binary or large file; 58,445 bytes | 二进制或大文件；58,445 字节 |
| `results/figures/baseline_2d/g5_2d_validation_all_classes.svg` | UTF-8 text with 1331 lines; 38,132 bytes | UTF-8 文本，1331 行；38,132 字节 |
| `results/figures/baseline_2d/README.md` | Entry point or directory guide; Sections: Supplementary 2D Validation figure | 项目入口或目录说明；章节标题：Supplementary 2D Validation figure |
| `results/figures/comparisons/g5_3d_validation_all_classes.png` | Binary or large file; 68,380 bytes | 二进制或大文件；68,380 字节 |
| `results/figures/comparisons/g5_3d_validation_all_classes.svg` | UTF-8 text with 1503 lines; 44,330 bytes | UTF-8 文本，1503 行；44,330 字节 |
| `results/figures/comparisons/g5_three_arm_training_trajectories.png` | Binary or large file; 182,705 bytes | 二进制或大文件；182,705 字节 |
| `results/figures/comparisons/g5_three_arm_training_trajectories.svg` | UTF-8 text with 1875 lines; 60,281 bytes | UTF-8 文本，1875 行；60,281 字节 |
| `results/figures/comparisons/g5_three_arm_validation_all_classes.png` | Binary or large file; 56,788 bytes | 二进制或大文件；56,788 字节 |
| `results/figures/comparisons/g5_three_arm_validation_all_classes.svg` | UTF-8 text with 1415 lines; 40,455 bytes | UTF-8 文本，1415 行；40,455 字节 |
| `results/figures/comparisons/g5_validation_body_boundaries.png` | Binary or large file; 100,677 bytes | 二进制或大文件；100,677 字节 |
| `results/figures/comparisons/g5_validation_body_boundaries.svg` | UTF-8 text with 1975 lines; 61,632 bytes | UTF-8 文本，1975 行；61,632 字节 |
| `results/figures/comparisons/g5_validation_patient_class_dice.png` | Binary or large file; 172,263 bytes | 二进制或大文件；172,263 字节 |
| `results/figures/comparisons/g5_validation_patient_class_dice.svg` | UTF-8 text with 2229 lines; 83,604 bytes | UTF-8 文本，2229 行；83,604 字节 |
| `results/figures/comparisons/g6_test_all_classes.png` | Binary or large file; 70,228 bytes | 二进制或大文件；70,228 字节 |
| `results/figures/comparisons/g6_test_all_classes.svg` | UTF-8 text with 1505 lines; 44,031 bytes | UTF-8 文本，1505 行；44,031 字节 |
| `results/figures/comparisons/g6_test_failure_patterns.png` | Binary or large file; 128,918 bytes | 二进制或大文件；128,918 字节 |
| `results/figures/comparisons/g6_test_failure_patterns.svg` | UTF-8 text with 2063 lines; 69,179 bytes | UTF-8 文本，2063 行；69,179 字节 |
| `results/figures/comparisons/g6_test_hd95_all_foreground.png` | Binary or large file; 64,275 bytes | 二进制或大文件；64,275 字节 |
| `results/figures/comparisons/g6_test_hd95_all_foreground.svg` | UTF-8 text with 1507 lines; 42,799 bytes | UTF-8 文本，1507 行；42,799 字节 |
| `results/figures/comparisons/README.md` | Entry point or directory guide; Sections: Model comparison figures | 项目入口或目录说明；章节标题：Model comparison figures |
| `results/README.md` | Entry point or directory guide; Sections: Results and Report Artifacts | 项目入口或目录说明；章节标题：Results and Report Artifacts |
| `results/reports/HipMRI_P1_独立研究审查_ZH.md` | Sections: P1：HipMRI 研究与课程交付独立审查; 1. 课程约束已核对，交付仍有明确缺口; 2. 源码复核带来的主要新判断; 3. G7 已完成什么，尚未证明什么; 4. 竞争假设与优先级 | 章节标题：P1：HipMRI 研究与课程交付独立审查; 1. 课程约束已核对，交付仍有明确缺口; 2. 源码复核带来的主要新判断; 3. G7 已完成什么，尚未证明什么; 4. 竞争假设与优先级 |
| `results/reports/HipMRI_Pretraining_Data_Analysis.md` | Sections: HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 1. Independent verification of the available record; 2. What the raw label statistics show; 3. Unusual cases to inspect first; 4. What remains unknown | 章节标题：HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 1. Independent verification of the available record; 2. What the raw label statistics show; 3. Unusual cases to inspect first; 4. What remains unknown |
| `results/reports/HipMRI_Research_Report.md` | Sections: HipMRI Research Report; Dataset Characteristics; Problems Identified; Interventions; Experimental Results | 章节标题：HipMRI Research Report; Dataset Characteristics; Problems Identified; Interventions; Experimental Results |
| `results/reports/HipMRI_模型复盘与下一轮实验_ZH.md` | Sections: HipMRI 模型复盘与下一轮实验决策; 1. 结论先明确; 2. 六类别与边界：它究竟赢在哪里、输在哪里; 3. 为什么没有达到预期：证据、解释与尚未证明之处; 4. 下一轮：先验证采样假设，再谈新模型胜利 | 章节标题：HipMRI 模型复盘与下一轮实验决策; 1. 结论先明确; 2. 六类别与边界：它究竟赢在哪里、输在哪里; 3. 为什么没有达到预期：证据、解释与尚未证明之处; 4. 下一轮：先验证采样假设，再谈新模型胜利 |
| `results/tables/comparisons/g5_validation_aux_logits.json` | JSON fields: kind, split, interpretation, checkpoint, body_audit_sha256, cases, summary | JSON 字段：kind, split, interpretation, checkpoint, body_audit_sha256, cases, summary |
| `results/tables/comparisons/g5_validation_body_audit.json` | JSON fields: kind, split, selection, cases, sources, pictured_cases | JSON 字段：kind, split, selection, cases, sources, pictured_cases |
| `results/tables/comparisons/g5_validation_case_class_metrics.csv` | Table with 456 data rows; columns: split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou | 表格含 456 条数据行；列：split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou |
| `results/tables/comparisons/g6_test_3d.json` | Binary or large file; 220,443 bytes | 二进制或大文件；220,443 字节 |
| `results/tables/comparisons/g6_test_case_class_metrics.csv` | Table with 360 data rows; columns: split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou | 表格含 360 条数据行；列：split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou |
| `results/tables/comparisons/README.md` | Entry point or directory guide; Sections: Held-out Test comparison | 项目入口或目录说明；章节标题：Held-out Test comparison |

Generated by `code/tools/update_project_map.py`; edit the source files, then regenerate this index. / 由 `code/tools/update_project_map.py` 生成；修改源文件后重新生成本目录。
