# Project File Map / 项目文件总览

Updated / 更新时间: **2026-10-21 07:06:44 AEST**

**English:** This generated index lists every directory and file under `20261017`, including local logs and generated artifacts. The content column summarizes structure (headings, code definitions, fields or file type) rather than copying another document's prose. Run `python code/tools/update_project_map.py --watch` to keep it current while the watcher is active.

**中文：** 本自动目录列出 `20261017` 下的全部目录和文件，包括本地日志及生成产物。内容栏概述章节、代码定义、字段或文件类型，不重复抄录其他文档。运行 `python code/tools/update_project_map.py --watch`，并保持进程运行，即可持续更新。

Directories / 目录: **110** · Files / 文件: **1013**

## Directory Tree / 目录树

```text
20261017/
├── archives/
│   └── remote_training_backups/
│       ├── improved_3d_v2_ds_20261020/
│       │   ├── best.pt
│       │   ├── config_frozen.json
│       │   ├── epoch_001.json
│       │   ├── epoch_001.pt
│       │   ├── epoch_002.json
│       │   ├── epoch_002.pt
│       │   ├── epoch_003.json
│       │   ├── epoch_003.pt
│       │   ├── epoch_004.json
│       │   ├── epoch_004.pt
│       │   ├── epoch_005.json
│       │   ├── epoch_005.pt
│       │   ├── epoch_006.json
│       │   ├── epoch_006.pt
│       │   ├── epoch_007.json
│       │   ├── epoch_007.pt
│       │   ├── epoch_008.json
│       │   ├── epoch_008.pt
│       │   ├── epoch_009.json
│       │   ├── epoch_009.pt
│       │   ├── epoch_010.json
│       │   ├── epoch_010.pt
│       │   ├── epoch_011.json
│       │   ├── epoch_011.pt
│       │   ├── epoch_012.json
│       │   ├── epoch_012.pt
│       │   ├── epoch_013.json
│       │   ├── epoch_013.pt
│       │   ├── epoch_014.json
│       │   ├── epoch_014.pt
│       │   ├── epoch_015.json
│       │   ├── epoch_015.pt
│       │   ├── epoch_016.json
│       │   ├── epoch_016.pt
│       │   ├── epoch_017.json
│       │   ├── epoch_017.pt
│       │   ├── epoch_018.json
│       │   ├── epoch_018.pt
│       │   ├── epoch_019.json
│       │   ├── epoch_019.pt
│       │   ├── epoch_020.json
│       │   ├── epoch_020.pt
│       │   ├── epoch_021.json
│       │   ├── epoch_021.pt
│       │   ├── epoch_022.json
│       │   ├── epoch_022.pt
│       │   ├── epoch_023.json
│       │   ├── epoch_023.pt
│       │   ├── epoch_024.json
│       │   ├── epoch_024.pt
│       │   ├── epoch_025.json
│       │   ├── epoch_025.pt
│       │   ├── epoch_026.json
│       │   ├── epoch_026.pt
│       │   ├── epoch_027.json
│       │   ├── epoch_027.pt
│       │   ├── epoch_028.json
│       │   ├── epoch_028.pt
│       │   ├── epoch_029.json
│       │   ├── epoch_029.pt
│       │   ├── epoch_030.json
│       │   ├── epoch_030.pt
│       │   ├── epoch_031.json
│       │   ├── epoch_031.pt
│       │   ├── epoch_032.json
│       │   ├── epoch_032.pt
│       │   ├── epoch_033.json
│       │   ├── epoch_033.pt
│       │   ├── epoch_034.json
│       │   ├── epoch_034.pt
│       │   ├── epoch_035.json
│       │   ├── epoch_035.pt
│       │   ├── epoch_036.json
│       │   ├── epoch_036.pt
│       │   ├── epoch_037.json
│       │   ├── epoch_037.pt
│       │   ├── epoch_038.json
│       │   ├── epoch_038.pt
│       │   ├── epoch_039.json
│       │   ├── epoch_039.pt
│       │   ├── epoch_040.json
│       │   ├── epoch_040.pt
│       │   ├── epoch_041.json
│       │   ├── epoch_041.pt
│       │   ├── epoch_042.json
│       │   ├── epoch_042.pt
│       │   ├── epoch_043.json
│       │   ├── epoch_043.pt
│       │   ├── epoch_044.json
│       │   ├── epoch_044.pt
│       │   ├── epoch_045.json
│       │   ├── epoch_045.pt
│       │   ├── epoch_046.json
│       │   ├── epoch_046.pt
│       │   ├── epoch_047.json
│       │   ├── epoch_047.pt
│       │   ├── epoch_048.json
│       │   ├── epoch_048.pt
│       │   ├── epoch_049.json
│       │   ├── epoch_049.pt
│       │   ├── epoch_050.json
│       │   ├── epoch_050.pt
│       │   ├── initial.pt
│       │   ├── latest.pt
│       │   └── manifest.json
│       └── improved_3d_v2_ds_20261020.sha256
├── code/
│   ├── config/
│   │   ├── arms/
│   │   │   ├── baseline_2d/
│   │   │   │   └── arm.json
│   │   │   ├── baseline_3d/
│   │   │   │   └── arm.json
│   │   │   └── improved_3d/
│   │   │       └── arm.json
│   │   ├── all_model_assessment_audit.json
│   │   ├── all_model_audit_checksums.json
│   │   ├── artifact_paths.json
│   │   ├── best_checkpoint_audit.json
│   │   ├── best_checkpoint_transfer_checksums.json
│   │   ├── g3_patch_audit.json
│   │   ├── g5_2d_training_frozen.json
│   │   ├── g5_controlled_training_candidate.json
│   │   ├── g5_controlled_training_frozen.json
│   │   ├── g5_weighted_ce_diagnostic.json
│   │   ├── g7_edge_sampling_candidate.json
│   │   ├── p2_inference_context.json
│   │   ├── p2b_prediction_mechanisms.json
│   │   ├── patient_split.json
│   │   ├── v2_independent_supervision_frozen.json
│   │   ├── v2_transfer_checksums.json
│   │   ├── v3_gn1_frozen.json
│   │   ├── v3_transfer_checksums.json
│   │   ├── v4_a0_frozen.json
│   │   ├── v4_a1_frozen.json
│   │   └── v4_transfer_checksums.json
│   ├── docs/
│   │   ├── ALL_MODEL_ASSESSMENT_AUDIT.md
│   │   ├── BEST_CHECKPOINT_AUDIT.md
│   │   ├── INFERENCE_CONTEXT_DIAGNOSTIC_ZH.md
│   │   ├── METHODS_EXECUTION_HISTORY.md
│   │   ├── PREDICTION_MECHANISM_DIAGNOSTIC_ZH.md
│   │   ├── README.md
│   │   ├── STUDY_PROTOCOL.md
│   │   ├── V2_TRAINING.md
│   │   ├── V3_TRAINING.md
│   │   └── V4_Paired_Training_ZH.md
│   ├── hipmri_rewrite/
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_2d.py
│   │   │   ├── baseline_3d.py
│   │   │   ├── common.py
│   │   │   └── improved_3d.py
│   │   ├── __init__.py
│   │   ├── audit.py
│   │   ├── augmentation_v4.py
│   │   ├── boundaries.py
│   │   ├── data.py
│   │   ├── improved_v2.py
│   │   ├── improved_v3.py
│   │   ├── inference.py
│   │   ├── inference_context.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── prediction_diagnostics.py
│   │   ├── profile.py
│   │   ├── sampling.py
│   │   ├── sampling_2d.py
│   │   ├── sampling_edges.py
│   │   ├── splits.py
│   │   └── training.py
│   ├── tests/
│   │   ├── test_all_model_audit.py
│   │   ├── test_best_checkpoint_audit.py
│   │   ├── test_boundaries.py
│   │   ├── test_data.py
│   │   ├── test_g5_candidate.py
│   │   ├── test_improved_v2.py
│   │   ├── test_improved_v3.py
│   │   ├── test_inference_context.py
│   │   ├── test_metrics.py
│   │   ├── test_models_and_losses.py
│   │   ├── test_prediction_diagnostics.py
│   │   ├── test_raw_audit.py
│   │   ├── test_sampling.py
│   │   ├── test_sampling_2d.py
│   │   ├── test_sampling_audit.py
│   │   ├── test_sampling_edges.py
│   │   ├── test_splits.py
│   │   ├── test_training_recovery.py
│   │   ├── test_training_selection.py
│   │   └── test_v4_training.py
│   ├── tools/
│   │   ├── audit_all_models.py
│   │   ├── audit_best_checkpoints.py
│   │   ├── audit_g2_train_slices.py
│   │   ├── audit_g3_candidate_sampling.py
│   │   ├── audit_g3_existing_pipeline.py
│   │   ├── audit_g4_patient_sampling.py
│   │   ├── audit_g5_2d_sampling.py
│   │   ├── audit_g5_aux_logits_validation.py
│   │   ├── audit_g5_train_patch_geometry.py
│   │   ├── audit_g5_validation_body.py
│   │   ├── audit_g7_edge_sampling.py
│   │   ├── audit_local_generalization.py
│   │   ├── audit_markdown.py
│   │   ├── audit_p2_inference_context.py
│   │   ├── audit_p2b_prediction_mechanisms.py
│   │   ├── audit_p2b_replay_numerics.py
│   │   ├── audit_train_raw.py
│   │   ├── audit_v4_augmentation.py
│   │   ├── build_best_checkpoint_review.py
│   │   ├── evaluate_g6_test.py
│   │   ├── organize_result_artifacts.py
│   │   ├── package_transfer.py
│   │   ├── package_v4_paired.py
│   │   ├── probe_g5_full_volume.py
│   │   ├── probe_g5_patch_gpu.py
│   │   ├── probe_g5_training_smoke.py
│   │   ├── review_g1_bladder_islands.py
│   │   ├── review_g1_overlays.py
│   │   ├── review_g1_train_contact_sheets.py
│   │   ├── review_g5_validation_failures.py
│   │   ├── review_g6_body_boundaries.py
│   │   ├── review_g6_test_cases.py
│   │   ├── review_v2_validation.py
│   │   ├── submit_all_model_audit.sh
│   │   ├── submit_best_checkpoint_audit.sh
│   │   ├── submit_g5_2d.sh
│   │   ├── submit_g5_2d_extend_50.sh
│   │   ├── submit_g5_40.sh
│   │   ├── submit_g5_extend_50.sh
│   │   ├── submit_g6_test.sh
│   │   ├── submit_g7_edge_audit.sh
│   │   ├── submit_p2_inference_context.sh
│   │   ├── submit_p2b_prediction_mechanisms.sh
│   │   ├── submit_v2_50.sh
│   │   ├── submit_v3_50.sh
│   │   ├── submit_v4_paired.sh
│   │   ├── summarize_all_class_review.py
│   │   ├── summarize_all_model_audit.py
│   │   ├── summarize_all_model_validation.py
│   │   ├── summarize_best_checkpoint_audit.py
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
│   │   ├── summarize_p2_inference_context.py
│   │   ├── summarize_p2b_prediction_mechanisms.py
│   │   ├── summarize_v2_results.py
│   │   ├── summarize_v3_results.py
│   │   ├── summarize_v4_pair.py
│   │   ├── tiny_overfit_g5.py
│   │   ├── train_g5_2d.py
│   │   ├── train_g5_cohort.py
│   │   ├── train_v2_cohort.py
│   │   ├── train_v3_cohort.py
│   │   ├── train_v4_cohort.py
│   │   └── update_project_map.py
│   ├── transfer/
│   │   ├── 20261017_source.zip
│   │   ├── all_model_assessment_20261021.zip
│   │   ├── best_checkpoint_audit_current_20261020.zip
│   │   └── v4_paired_spatial_20261020.zip
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
│   │   ├── inference_context/
│   │   │   └── p2_four_cases_20261020/
│   │   │       ├── B040_Week4_improved_3d.json
│   │   │       ├── B040_Week4_standard_3d.json
│   │   │       ├── checksums.json
│   │   │       ├── J026_Week4_improved_3d.json
│   │   │       ├── J026_Week4_standard_3d.json
│   │   │       ├── K008_Week0_improved_3d.json
│   │   │       ├── K008_Week0_standard_3d.json
│   │   │       ├── K018_Week4_improved_3d.json
│   │   │       ├── K018_Week4_standard_3d.json
│   │   │       └── manifest.json
│   │   └── README.md
│   └── README.md
├── results/
│   ├── case_studies/
│   │   └── comparisons/
│   │       ├── test/
│   │       │   ├── body/
│   │       │   │   ├── g6_body_L011_Week5.png
│   │       │   │   ├── g6_body_M015_Week3.png
│   │       │   │   ├── g6_body_selected_cases.json
│   │       │   │   ├── g6_body_W041_Week0.png
│   │       │   │   └── README.md
│   │       │   └── small_organs/
│   │       │       ├── g6_test_M015_Week7_bladder.png
│   │       │       ├── g6_test_M020_Week1_prostate.png
│   │       │       ├── g6_test_selected_cases.json
│   │       │       ├── g6_test_W029_Week6_prostate.png
│   │       │       └── README.md
│   │       ├── validation/
│   │       │   ├── body/
│   │       │   │   ├── g5_body_validation_B040_Week4.png
│   │       │   │   ├── g5_body_validation_J026_Week4.png
│   │       │   │   ├── g5_body_validation_K018_Week4.png
│   │       │   │   └── README.md
│   │       │   └── small_organs/
│   │       │       ├── g5_validation_B040_Week1_prostate.png
│   │       │       ├── g5_validation_J026_Week1_rectum.png
│   │       │       ├── g5_validation_K018_Week3_rectum.png
│   │       │       ├── g5_validation_S022_Week6_prostate.png
│   │       │       ├── g5_validation_selected_cases.json
│   │       │       └── README.md
│   │       └── README.md
│   ├── data_analysis/
│   │   ├── boundary_sampling_audit/
│   │   │   ├── class_exposure.csv
│   │   │   ├── crop_origin_distribution.png
│   │   │   ├── crop_origin_distribution.svg
│   │   │   ├── patient_class_exposure.csv
│   │   │   ├── README.md
│   │   │   ├── sampling_exposure_and_geometry.png
│   │   │   ├── sampling_exposure_and_geometry.svg
│   │   │   └── verification.json
│   │   ├── label_component_review/
│   │   │   ├── g1_bladder_island_D031_Week7.png
│   │   │   ├── g1_bladder_island_N010_Week3.png
│   │   │   └── README.md
│   │   ├── legacy_training_statistics/
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
│   │   ├── patient_alignment_screening/
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
│   │   ├── patient_balanced_sampling/
│   │   │   ├── class_exposure.csv
│   │   │   ├── g5_train_patch_geometry.json
│   │   │   ├── patient_balanced_exposure.svg
│   │   │   ├── patient_exposure.csv
│   │   │   └── README.md
│   │   ├── preprocessing_validation/
│   │   │   ├── candidate_patch_class_exposure.csv
│   │   │   ├── candidate_patch_exposure.svg
│   │   │   ├── candidate_patch_patient_exposure.csv
│   │   │   ├── organ_physical_volume_ratio.svg
│   │   │   ├── organ_preservation_by_volume.csv
│   │   │   └── README.md
│   │   ├── representative_alignment_review/
│   │   │   ├── g1_overlay_B038_Week0.png
│   │   │   ├── g1_overlay_D031_Week7.png
│   │   │   ├── g1_overlay_H017_Week7.png
│   │   │   ├── g1_overlay_K019_Week1.png
│   │   │   ├── g1_overlay_N010_Week3.png
│   │   │   ├── g1_overlay_S033_Week1.png
│   │   │   └── README.md
│   │   ├── training_distribution/
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
│   │   └── README.md
│   ├── diagnostics/
│   │   └── inference_context/
│   │       ├── context_sensitivity_20261020/
│   │       │   ├── B040_Week4_head_margins.png
│   │       │   ├── B040_Week4_improved_3d_body.png
│   │       │   ├── B040_Week4_standard_3d_body.png
│   │       │   ├── checksums.json
│   │       │   ├── CONTEXT_SENSITIVITY_REVIEW_ZH.md
│   │       │   ├── J026_Week4_head_margins.png
│   │       │   ├── J026_Week4_improved_3d_body.png
│   │       │   ├── J026_Week4_standard_3d_body.png
│   │       │   ├── K008_Week0_head_margins.png
│   │       │   ├── K008_Week0_improved_3d_body.png
│   │       │   ├── K008_Week0_standard_3d_body.png
│   │       │   ├── K018_Week4_head_margins.png
│   │       │   ├── K018_Week4_improved_3d_body.png
│   │       │   ├── K018_Week4_standard_3d_body.png
│   │       │   ├── README.md
│   │       │   ├── six_class_context_comparison.png
│   │       │   ├── six_class_context_comparison.svg
│   │       │   └── six_class_metrics.csv
│   │       ├── prediction_mechanisms_20261020/
│   │       │   ├── NUMERICAL_REPLAY_REVIEW_ZH.md
│   │       │   ├── README.md
│   │       │   ├── same_voxel_logit_evidence.png
│   │       │   ├── same_voxel_logit_evidence.svg
│   │       │   └── spatial_logit_evidence.csv
│   │       └── README.md
│   ├── documentation_audit/
│   │   ├── audit_20261020.json
│   │   ├── before_20261020.json
│   │   ├── cleanup_validation_20261020.json
│   │   ├── consolidation_20261020.json
│   │   ├── naming_migration_20261020.json
│   │   ├── publication_audit_20261020.json
│   │   ├── README.md
│   │   └── relocations_20261020.json
│   ├── experiments/
│   │   ├── improved_3d_v2/
│   │   │   ├── case_studies/
│   │   │   │   └── validation/
│   │   │   │       └── review_45_50/
│   │   │   │           ├── J026_Week4_body_three_planes.png
│   │   │   │           ├── J026_Week4_six_classes.png
│   │   │   │           ├── K018_Week4_body_three_planes.png
│   │   │   │           ├── K018_Week4_six_classes.png
│   │   │   │           ├── local_verification.json
│   │   │   │           ├── M030_Week0_body_three_planes.png
│   │   │   │           ├── M030_Week0_six_classes.png
│   │   │   │           ├── provenance.json
│   │   │   │           ├── README.md
│   │   │   │           ├── S022_Week2_body_three_planes.png
│   │   │   │           └── S022_Week2_six_classes.png
│   │   │   ├── figures/
│   │   │   │   ├── all_six_classes.png
│   │   │   │   ├── all_six_classes.svg
│   │   │   │   ├── patient_comparison.png
│   │   │   │   ├── patient_comparison.svg
│   │   │   │   ├── README.md
│   │   │   │   ├── S022_Week2_error_counts.png
│   │   │   │   ├── S022_Week2_error_counts.svg
│   │   │   │   ├── training_and_boundary_trajectory.png
│   │   │   │   └── training_and_boundary_trajectory.svg
│   │   │   ├── provenance/
│   │   │   │   └── local_source_checksums.json
│   │   │   ├── reports/
│   │   │   │   └── RESEARCH_REPORT_ZH.md
│   │   │   ├── tables/
│   │   │   │   ├── analysis_summary.json
│   │   │   │   ├── body_case_comparison.csv
│   │   │   │   ├── epoch45_to_50_patient_delta.csv
│   │   │   │   ├── patient_comparison.csv
│   │   │   │   ├── S022_Week2_error_counts.csv
│   │   │   │   ├── six_class_comparison.csv
│   │   │   │   └── validation_trajectory.csv
│   │   │   ├── training_records/
│   │   │   │   ├── config_frozen.json
│   │   │   │   ├── epoch_001.json
│   │   │   │   ├── epoch_002.json
│   │   │   │   ├── epoch_003.json
│   │   │   │   ├── epoch_004.json
│   │   │   │   ├── epoch_005.json
│   │   │   │   ├── epoch_006.json
│   │   │   │   ├── epoch_007.json
│   │   │   │   ├── epoch_008.json
│   │   │   │   ├── epoch_009.json
│   │   │   │   ├── epoch_010.json
│   │   │   │   ├── epoch_011.json
│   │   │   │   ├── epoch_012.json
│   │   │   │   ├── epoch_013.json
│   │   │   │   ├── epoch_014.json
│   │   │   │   ├── epoch_015.json
│   │   │   │   ├── epoch_016.json
│   │   │   │   ├── epoch_017.json
│   │   │   │   ├── epoch_018.json
│   │   │   │   ├── epoch_019.json
│   │   │   │   ├── epoch_020.json
│   │   │   │   ├── epoch_021.json
│   │   │   │   ├── epoch_022.json
│   │   │   │   ├── epoch_023.json
│   │   │   │   ├── epoch_024.json
│   │   │   │   ├── epoch_025.json
│   │   │   │   ├── epoch_026.json
│   │   │   │   ├── epoch_027.json
│   │   │   │   ├── epoch_028.json
│   │   │   │   ├── epoch_029.json
│   │   │   │   ├── epoch_030.json
│   │   │   │   ├── epoch_031.json
│   │   │   │   ├── epoch_032.json
│   │   │   │   ├── epoch_033.json
│   │   │   │   ├── epoch_034.json
│   │   │   │   ├── epoch_035.json
│   │   │   │   ├── epoch_036.json
│   │   │   │   ├── epoch_037.json
│   │   │   │   ├── epoch_038.json
│   │   │   │   ├── epoch_039.json
│   │   │   │   ├── epoch_040.json
│   │   │   │   ├── epoch_041.json
│   │   │   │   ├── epoch_042.json
│   │   │   │   ├── epoch_043.json
│   │   │   │   ├── epoch_044.json
│   │   │   │   ├── epoch_045.json
│   │   │   │   ├── epoch_046.json
│   │   │   │   ├── epoch_047.json
│   │   │   │   ├── epoch_048.json
│   │   │   │   ├── epoch_049.json
│   │   │   │   ├── epoch_050.json
│   │   │   │   └── manifest.json
│   │   │   └── README.md
│   │   ├── improved_3d_v3/
│   │   │   ├── provenance/
│   │   │   │   └── local_source_checksums.json
│   │   │   ├── reports/
│   │   │   │   └── RESEARCH_REPORT_ZH.md
│   │   │   ├── tables/
│   │   │   │   ├── analysis_summary.json
│   │   │   │   ├── body_case_comparison.csv
│   │   │   │   ├── foreground_surfaces_v3.csv
│   │   │   │   ├── patient_comparison.csv
│   │   │   │   ├── six_class_comparison.csv
│   │   │   │   ├── v3_minus_standard_patient_delta.csv
│   │   │   │   ├── v3_minus_v1_patient_delta.csv
│   │   │   │   └── validation_trajectory.csv
│   │   │   ├── training_records/
│   │   │   │   ├── config_frozen.json
│   │   │   │   ├── epoch_001.json
│   │   │   │   ├── epoch_002.json
│   │   │   │   ├── epoch_003.json
│   │   │   │   ├── epoch_004.json
│   │   │   │   ├── epoch_005.json
│   │   │   │   ├── epoch_006.json
│   │   │   │   ├── epoch_007.json
│   │   │   │   ├── epoch_008.json
│   │   │   │   ├── epoch_009.json
│   │   │   │   ├── epoch_010.json
│   │   │   │   ├── epoch_011.json
│   │   │   │   ├── epoch_012.json
│   │   │   │   ├── epoch_013.json
│   │   │   │   ├── epoch_014.json
│   │   │   │   ├── epoch_015.json
│   │   │   │   ├── epoch_016.json
│   │   │   │   ├── epoch_017.json
│   │   │   │   ├── epoch_018.json
│   │   │   │   ├── epoch_019.json
│   │   │   │   ├── epoch_020.json
│   │   │   │   ├── epoch_021.json
│   │   │   │   ├── epoch_022.json
│   │   │   │   ├── epoch_023.json
│   │   │   │   ├── epoch_024.json
│   │   │   │   ├── epoch_025.json
│   │   │   │   ├── epoch_026.json
│   │   │   │   ├── epoch_027.json
│   │   │   │   ├── epoch_028.json
│   │   │   │   ├── epoch_029.json
│   │   │   │   ├── epoch_030.json
│   │   │   │   ├── epoch_031.json
│   │   │   │   ├── epoch_032.json
│   │   │   │   ├── epoch_033.json
│   │   │   │   ├── epoch_034.json
│   │   │   │   ├── epoch_035.json
│   │   │   │   ├── epoch_036.json
│   │   │   │   ├── epoch_037.json
│   │   │   │   ├── epoch_038.json
│   │   │   │   ├── epoch_039.json
│   │   │   │   ├── epoch_040.json
│   │   │   │   ├── epoch_041.json
│   │   │   │   ├── epoch_042.json
│   │   │   │   ├── epoch_043.json
│   │   │   │   ├── epoch_044.json
│   │   │   │   ├── epoch_045.json
│   │   │   │   ├── epoch_046.json
│   │   │   │   ├── epoch_047.json
│   │   │   │   ├── epoch_048.json
│   │   │   │   ├── epoch_049.json
│   │   │   │   ├── epoch_050.json
│   │   │   │   └── manifest.json
│   │   │   └── README.md
│   │   └── improved_3d_v4/
│   │       ├── a0/
│   │       │   └── training_records/
│   │       │       ├── config_frozen.json
│   │       │       ├── epoch_001.json
│   │       │       ├── epoch_002.json
│   │       │       ├── epoch_003.json
│   │       │       ├── epoch_004.json
│   │       │       ├── epoch_005.json
│   │       │       ├── epoch_006.json
│   │       │       ├── epoch_007.json
│   │       │       ├── epoch_008.json
│   │       │       ├── epoch_009.json
│   │       │       ├── epoch_010.json
│   │       │       ├── epoch_011.json
│   │       │       ├── epoch_012.json
│   │       │       ├── epoch_013.json
│   │       │       ├── epoch_014.json
│   │       │       ├── epoch_015.json
│   │       │       ├── epoch_016.json
│   │       │       ├── epoch_017.json
│   │       │       ├── epoch_018.json
│   │       │       ├── epoch_019.json
│   │       │       ├── epoch_020.json
│   │       │       ├── epoch_021.json
│   │       │       ├── epoch_022.json
│   │       │       ├── epoch_023.json
│   │       │       ├── epoch_024.json
│   │       │       ├── epoch_025.json
│   │       │       ├── epoch_026.json
│   │       │       ├── epoch_027.json
│   │       │       ├── epoch_028.json
│   │       │       ├── epoch_029.json
│   │       │       ├── epoch_030.json
│   │       │       ├── epoch_031.json
│   │       │       ├── epoch_032.json
│   │       │       ├── epoch_033.json
│   │       │       ├── epoch_034.json
│   │       │       ├── epoch_035.json
│   │       │       ├── epoch_036.json
│   │       │       ├── epoch_037.json
│   │       │       ├── epoch_038.json
│   │       │       ├── epoch_039.json
│   │       │       ├── epoch_040.json
│   │       │       ├── epoch_041.json
│   │       │       ├── epoch_042.json
│   │       │       ├── epoch_043.json
│   │       │       ├── epoch_044.json
│   │       │       ├── epoch_045.json
│   │       │       ├── epoch_046.json
│   │       │       ├── epoch_047.json
│   │       │       ├── epoch_048.json
│   │       │       ├── epoch_049.json
│   │       │       ├── epoch_050.json
│   │       │       └── manifest.json
│   │       ├── a1/
│   │       │   └── training_records/
│   │       │       ├── config_frozen.json
│   │       │       ├── epoch_001.json
│   │       │       ├── epoch_002.json
│   │       │       ├── epoch_003.json
│   │       │       ├── epoch_004.json
│   │       │       ├── epoch_005.json
│   │       │       ├── epoch_006.json
│   │       │       ├── epoch_007.json
│   │       │       ├── epoch_008.json
│   │       │       ├── epoch_009.json
│   │       │       ├── epoch_010.json
│   │       │       ├── epoch_011.json
│   │       │       ├── epoch_012.json
│   │       │       ├── epoch_013.json
│   │       │       ├── epoch_014.json
│   │       │       ├── epoch_015.json
│   │       │       ├── epoch_016.json
│   │       │       ├── epoch_017.json
│   │       │       ├── epoch_018.json
│   │       │       ├── epoch_019.json
│   │       │       ├── epoch_020.json
│   │       │       ├── epoch_021.json
│   │       │       ├── epoch_022.json
│   │       │       ├── epoch_023.json
│   │       │       ├── epoch_024.json
│   │       │       ├── epoch_025.json
│   │       │       ├── epoch_026.json
│   │       │       ├── epoch_027.json
│   │       │       ├── epoch_028.json
│   │       │       ├── epoch_029.json
│   │       │       ├── epoch_030.json
│   │       │       ├── epoch_031.json
│   │       │       ├── epoch_032.json
│   │       │       ├── epoch_033.json
│   │       │       ├── epoch_034.json
│   │       │       ├── epoch_035.json
│   │       │       ├── epoch_036.json
│   │       │       ├── epoch_037.json
│   │       │       ├── epoch_038.json
│   │       │       ├── epoch_039.json
│   │       │       ├── epoch_040.json
│   │       │       ├── epoch_041.json
│   │       │       ├── epoch_042.json
│   │       │       ├── epoch_043.json
│   │       │       ├── epoch_044.json
│   │       │       ├── epoch_045.json
│   │       │       ├── epoch_046.json
│   │       │       ├── epoch_047.json
│   │       │       ├── epoch_048.json
│   │       │       ├── epoch_049.json
│   │       │       ├── epoch_050.json
│   │       │       └── manifest.json
│   │       ├── assessment_analysis/
│   │       │   ├── assessment_required/
│   │       │   │   ├── figures/
│   │       │   │   │   ├── body_boundary_distribution.png
│   │       │   │   │   ├── body_boundary_distribution.svg
│   │       │   │   │   ├── prostate_base_apex.png
│   │       │   │   │   ├── prostate_base_apex.svg
│   │       │   │   │   ├── prostate_boundary_tails.png
│   │       │   │   │   ├── prostate_boundary_tails.svg
│   │       │   │   │   ├── prostate_oar_tradeoff.png
│   │       │   │   │   ├── prostate_oar_tradeoff.svg
│   │       │   │   │   ├── resource_profile.png
│   │       │   │   │   ├── resource_profile.svg
│   │       │   │   │   ├── six_class_validation_test.png
│   │       │   │   │   ├── six_class_validation_test.svg
│   │       │   │   │   ├── validation_test_generalization.png
│   │       │   │   │   └── validation_test_generalization.svg
│   │       │   │   └── tables/
│   │       │   │       ├── boundary_errors.csv
│   │       │   │       ├── model_summary.csv
│   │       │   │       ├── per_case_inference.csv
│   │       │   │       ├── prostate_regions.csv
│   │       │   │       ├── resources.csv
│   │       │   │       ├── six_class_metrics.csv
│   │       │   │       └── surface_summary.csv
│   │       │   ├── supplementary_research/
│   │       │   │   ├── figures/
│   │       │   │   │   ├── distant_body_false_positives.png
│   │       │   │   │   ├── distant_body_false_positives.svg
│   │       │   │   │   ├── patient_variability.png
│   │       │   │   │   ├── patient_variability.svg
│   │       │   │   │   ├── prostate_continuity.png
│   │       │   │   │   └── prostate_continuity.svg
│   │       │   │   └── tables/
│   │       │   │       ├── continuity.csv
│   │       │   │       ├── paired_patient_sensitivity.csv
│   │       │   │       ├── per_case_class_metrics.csv
│   │       │   │       ├── per_case_regions.csv
│   │       │   │       ├── per_patient.csv
│   │       │   │       └── surface_distances.csv
│   │       │   ├── coverage.json
│   │       │   └── integrity_verification.json
│   │       ├── augmentation_audit_20261020/
│   │       │   ├── audit.json
│   │       │   ├── checksums.json
│   │       │   ├── preview_01_O025_Week6.png
│   │       │   ├── preview_02_O025_Week6.png
│   │       │   ├── preview_03_O025_Week6.png
│   │       │   ├── preview_04_O025_Week6.png
│   │       │   ├── preview_05_R039_Week2.png
│   │       │   ├── preview_06_R039_Week2.png
│   │       │   └── README.md
│   │       ├── comprehensive_audit_20261021/
│   │       │   ├── images/
│   │       │   │   ├── test/
│   │       │   │   │   ├── M015_Week7_Bladder.png
│   │       │   │   │   ├── M020_Week1_Prostate.png
│   │       │   │   │   ├── M020_Week1_Prostate_base_apex.png
│   │       │   │   │   ├── README.md
│   │       │   │   │   ├── W029_Week6_Body.png
│   │       │   │   │   ├── W029_Week6_Bones.png
│   │       │   │   │   ├── W029_Week6_Prostate.png
│   │       │   │   │   └── W029_Week6_Prostate_base_apex.png
│   │       │   │   └── validation/
│   │       │   │       ├── B040_Week1_Background.png
│   │       │   │       ├── B040_Week1_Bladder.png
│   │       │   │       ├── B040_Week1_Body.png
│   │       │   │       ├── B040_Week1_Body_three_planes.png
│   │       │   │       ├── B040_Week1_Bones.png
│   │       │   │       ├── B040_Week1_Prostate.png
│   │       │   │       ├── B040_Week1_Prostate_base_apex.png
│   │       │   │       ├── B040_Week1_Rectum.png
│   │       │   │       ├── J026_Week4_Body_three_planes.png
│   │       │   │       ├── J026_Week4_Rectum.png
│   │       │   │       ├── K008_Week0_Bladder.png
│   │       │   │       ├── K008_Week0_Body_three_planes.png
│   │       │   │       ├── K008_Week0_Prostate.png
│   │       │   │       ├── K008_Week0_Prostate_base_apex.png
│   │       │   │       ├── K018_Week4_Background.png
│   │       │   │       ├── K018_Week4_Body.png
│   │       │   │       ├── K018_Week4_Body_three_planes.png
│   │       │   │       ├── M030_Week4_Body.png
│   │       │   │       ├── M030_Week4_Body_three_planes.png
│   │       │   │       ├── M030_Week4_Bones.png
│   │       │   │       ├── README.md
│   │       │   │       ├── S022_Week2_Bladder.png
│   │       │   │       ├── S022_Week2_Body_three_planes.png
│   │       │   │       ├── S022_Week2_Prostate.png
│   │       │   │       └── S022_Week2_Prostate_base_apex.png
│   │       │   ├── checksums.json
│   │       │   ├── input_provenance.json
│   │       │   ├── metrics.json
│   │       │   ├── metrics_partial.json
│   │       │   └── README.md
│   │       ├── paired_analysis/
│   │       │   ├── all_models/
│   │       │   │   ├── selection.csv
│   │       │   │   ├── seven_model_selection.png
│   │       │   │   ├── seven_model_selection.svg
│   │       │   │   ├── seven_model_six_class_dice.png
│   │       │   │   ├── seven_model_six_class_dice.svg
│   │       │   │   ├── six_class_validation.csv
│   │       │   │   └── source_checksums.json
│   │       │   ├── learning_curves.csv
│   │       │   ├── learning_curves.png
│   │       │   ├── learning_curves.svg
│   │       │   ├── patient_metrics.csv
│   │       │   ├── README.md
│   │       │   ├── record_checksums.json
│   │       │   ├── six_class_dice.png
│   │       │   ├── six_class_dice.svg
│   │       │   ├── six_class_metrics.csv
│   │       │   └── summary.json
│   │       ├── reports/
│   │       │   ├── OPTIMIZATION_REVIEW_ZH.md
│   │       │   └── RESEARCH_REPORT_ZH.md
│   │       └── README.md
│   ├── figures/
│   │   ├── baseline_2d/
│   │   │   ├── g5_2d_validation_all_classes.png
│   │   │   ├── g5_2d_validation_all_classes.svg
│   │   │   └── README.md
│   │   └── comparisons/
│   │       ├── test/
│   │       │   ├── g6_test_all_classes.png
│   │       │   ├── g6_test_all_classes.svg
│   │       │   ├── g6_test_failure_patterns.png
│   │       │   ├── g6_test_failure_patterns.svg
│   │       │   ├── g6_test_hd95_all_foreground.png
│   │       │   ├── g6_test_hd95_all_foreground.svg
│   │       │   └── README.md
│   │       ├── training/
│   │       │   ├── g5_three_arm_training_trajectories.png
│   │       │   ├── g5_three_arm_training_trajectories.svg
│   │       │   └── README.md
│   │       ├── validation/
│   │       │   ├── g5_3d_validation_all_classes.png
│   │       │   ├── g5_3d_validation_all_classes.svg
│   │       │   ├── g5_three_arm_validation_all_classes.png
│   │       │   ├── g5_three_arm_validation_all_classes.svg
│   │       │   ├── g5_validation_body_boundaries.png
│   │       │   ├── g5_validation_body_boundaries.svg
│   │       │   ├── g5_validation_patient_class_dice.png
│   │       │   ├── g5_validation_patient_class_dice.svg
│   │       │   └── README.md
│   │       └── README.md
│   ├── model_evaluation/
│   │   ├── best_checkpoint_comparison/
│   │   │   ├── assessment_required/
│   │   │   │   ├── figures/
│   │   │   │   │   ├── body_boundary_by_case.png
│   │   │   │   │   ├── body_boundary_by_case.svg
│   │   │   │   │   ├── learning_curves_and_lr.png
│   │   │   │   │   ├── learning_curves_and_lr.svg
│   │   │   │   │   ├── six_class_dice_iou.png
│   │   │   │   │   └── six_class_dice_iou.svg
│   │   │   │   ├── source_review/
│   │   │   │   │   ├── assessment_page_14.png
│   │   │   │   │   ├── assessment_page_5.png
│   │   │   │   │   └── assessment_page_6.png
│   │   │   │   ├── tables/
│   │   │   │   │   ├── body_boundary_available.csv
│   │   │   │   │   ├── resources_measured_and_pending.csv
│   │   │   │   │   ├── selected_checkpoints.json
│   │   │   │   │   └── six_class_best_metrics.csv
│   │   │   │   └── README.md
│   │   │   ├── supplementary_research/
│   │   │   │   ├── figures/
│   │   │   │   │   ├── distant_body_fp.png
│   │   │   │   │   ├── distant_body_fp.svg
│   │   │   │   │   ├── normalized_confusion.png
│   │   │   │   │   ├── normalized_confusion.svg
│   │   │   │   │   ├── patient_small_organ_dice.png
│   │   │   │   │   ├── patient_small_organ_dice.svg
│   │   │   │   │   ├── precision_recall.png
│   │   │   │   │   ├── precision_recall.svg
│   │   │   │   │   ├── volume_bias.png
│   │   │   │   │   └── volume_bias.svg
│   │   │   │   ├── tables/
│   │   │   │   │   ├── per_patient_best.csv
│   │   │   │   │   └── relative_volume_bias.csv
│   │   │   │   └── README.md
│   │   │   ├── README.md
│   │   │   └── source_checksums.json
│   │   └── generalization_diagnostics/
│   │       ├── figures/
│   │       │   ├── patient_influence.png
│   │       │   └── patient_influence.svg
│   │       ├── provenance/
│   │       │   └── local_source_checksums.json
│   │       ├── tables/
│   │       │   ├── audit.json
│   │       │   ├── historical_test_patient_influence.csv
│   │       │   └── validation_error_sources.csv
│   │       └── README.md
│   ├── reports/
│   │   ├── DATA_AND_METHODS_REVIEW_ZH.md
│   │   ├── README.md
│   │   ├── RESEARCH_GOVERNANCE_ZH.md
│   │   └── RESEARCH_REPORT.md
│   ├── tables/
│   │   └── comparisons/
│   │       ├── g5_validation_aux_logits.json
│   │       ├── g5_validation_body_audit.json
│   │       ├── g5_validation_case_class_metrics.csv
│   │       ├── g6_test_3d.json
│   │       ├── g6_test_case_class_metrics.csv
│   │       └── README.md
│   ├── README.md
│   └── STORAGE_GUIDE.md
├── .gitignore
├── DOCUMENTATION_INDEX.md
├── PROJECT_MAP.md
└── README.md
```

## File Contents / 文件内容概览

| File / 文件 | English content overview | 中文内容概览 |
|---|---|---|
| `.gitignore` | Git rules for local artifacts; UTF-8 text with 42 lines; 2,019 bytes | 本地运行产物的 Git 规则；UTF-8 文本，42 行；2,019 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/best.pt` | Binary or large file; 28,547,995 bytes | 二进制或大文件；28,547,995 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/config_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_001.pt` | Binary or large file; 28,529,051 bytes | 二进制或大文件；28,529,051 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_002.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_003.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_004.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_005.json` | Binary or large file; 141,022 bytes | 二进制或大文件；141,022 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_005.pt` | Binary or large file; 28,548,315 bytes | 二进制或大文件；28,548,315 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_006.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_007.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_008.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_009.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_010.json` | Binary or large file; 140,803 bytes | 二进制或大文件；140,803 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_010.pt` | Binary or large file; 28,548,187 bytes | 二进制或大文件；28,548,187 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_011.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_012.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_013.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_014.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_015.json` | Binary or large file; 140,483 bytes | 二进制或大文件；140,483 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_015.pt` | Binary or large file; 28,548,123 bytes | 二进制或大文件；28,548,123 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_016.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_017.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_018.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_019.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_020.json` | Binary or large file; 140,535 bytes | 二进制或大文件；140,535 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_020.pt` | Binary or large file; 28,548,123 bytes | 二进制或大文件；28,548,123 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_021.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_022.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_023.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_024.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_025.json` | Binary or large file; 140,871 bytes | 二进制或大文件；140,871 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_025.pt` | Binary or large file; 28,548,251 bytes | 二进制或大文件；28,548,251 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_026.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_027.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_028.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_029.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_030.json` | Binary or large file; 140,249 bytes | 二进制或大文件；140,249 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_030.pt` | Binary or large file; 28,548,059 bytes | 二进制或大文件；28,548,059 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_031.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_032.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_033.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_034.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_035.json` | Binary or large file; 140,084 bytes | 二进制或大文件；140,084 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_035.pt` | Binary or large file; 28,548,059 bytes | 二进制或大文件；28,548,059 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_036.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_037.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_038.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_039.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_040.json` | Binary or large file; 140,292 bytes | 二进制或大文件；140,292 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_040.pt` | Binary or large file; 28,548,059 bytes | 二进制或大文件；28,548,059 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_041.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_042.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_043.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_044.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_045.json` | Binary or large file; 140,101 bytes | 二进制或大文件；140,101 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_045.pt` | Binary or large file; 28,547,995 bytes | 二进制或大文件；28,547,995 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_046.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_047.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_048.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_049.pt` | Binary or large file; 28,528,987 bytes | 二进制或大文件；28,528,987 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_050.json` | Binary or large file; 140,261 bytes | 二进制或大文件；140,261 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/epoch_050.pt` | Binary or large file; 28,548,059 bytes | 二进制或大文件；28,548,059 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/initial.pt` | Binary or large file; 9,504,373 bytes | 二进制或大文件；9,504,373 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/latest.pt` | Binary or large file; 28,548,059 bytes | 二进制或大文件；28,548,059 字节 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020/manifest.json` | JSON fields: boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 | JSON 字段：boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 |
| `archives/remote_training_backups/improved_3d_v2_ds_20261020.sha256` | UTF-8 text with 105 lines; 8,497 bytes | UTF-8 文本，105 行；8,497 字节 |
| `code/config/all_model_assessment_audit.json` | JSON fields: status, selected, v4_pretest_plan_sha256, historical_test_sha256, v4_pretest_plan_path, evaluation_policy | JSON 字段：status, selected, v4_pretest_plan_sha256, historical_test_sha256, v4_pretest_plan_path, evaluation_policy |
| `code/config/all_model_audit_checksums.json` | JSON fields: tools/audit_all_models.py, tools/submit_all_model_audit.sh, tests/test_all_model_audit.py, config/all_model_assessment_audit.json, docs/ALL_MODEL_ASSESSMENT_AUDIT.md | JSON 字段：tools/audit_all_models.py, tools/submit_all_model_audit.sh, tests/test_all_model_audit.py, config/all_model_assessment_audit.json, docs/ALL_MODEL_ASSESSMENT_AUDIT.md |
| `code/config/arms/baseline_2d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/baseline_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/improved_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/artifact_paths.json` | JSON fields: schema_version, purpose, directory_renames, file_renames, unchanged_contracts | JSON 字段：schema_version, purpose, directory_renames, file_renames, unchanged_contracts |
| `code/config/best_checkpoint_audit.json` | JSON fields: status, selected, v4_pretest_plan_sha256, historical_test_sha256, v4_pretest_plan_path | JSON 字段：status, selected, v4_pretest_plan_sha256, historical_test_sha256, v4_pretest_plan_path |
| `code/config/best_checkpoint_transfer_checksums.json` | JSON fields: tools/audit_best_checkpoints.py, tests/test_best_checkpoint_audit.py, config/best_checkpoint_audit.json, tools/submit_best_checkpoint_audit.sh, docs/V4_Paired_Training_ZH.md, docs/BEST_CHECKPOINT_AUDIT.md | JSON 字段：tools/audit_best_checkpoints.py, tests/test_best_checkpoint_audit.py, config/best_checkpoint_audit.json, tools/submit_best_checkpoint_audit.sh, docs/V4_Paired_Training_ZH.md, docs/BEST_CHECKPOINT_AUDIT.md |
| `code/config/g3_patch_audit.json` | Frozen candidate-crop audit design, not a training configuration; JSON fields: scope, seed, patch_shape_xyz, draws_per_case_per_arm, arm_schedules | 固定的候选裁剪审计设计，非训练配置；JSON 字段：scope, seed, patch_shape_xyz, draws_per_case_per_arm, arm_schedules |
| `code/config/g5_2d_training_frozen.json` | JSON fields: status, purpose, model, model_defaults, split_sha256, train_cases, train_patients, validation_cases, validation_patients, seed, preprocessing, sampling | JSON 字段：status, purpose, model, model_defaults, split_sha256, train_cases, train_patients, validation_cases, validation_patients, seed, preprocessing, sampling |
| `code/config/g5_controlled_training_candidate.json` | Historical candidate for the paired 3D cohort training protocol; JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | 双模型队列训练方案的历史候选配置；JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/g5_controlled_training_frozen.json` | Frozen paired 3D cohort training protocol with resumable checkpoint selection; JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | 已冻结的双模型训练方案，含可恢复权重和验证集选择规则；JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/g5_weighted_ce_diagnostic.json` | Train-only weighted CE configuration for the Standard tiny learning diagnostic; JSON fields: scope, source_table, source_table_sha256, source_policy, formula, ce_weight_in_combined_loss, class_weights | 仅用训练数据为 Standard 微型学习诊断配置加权交叉熵；JSON 字段：scope, source_table, source_table_sha256, source_policy, formula, ce_weight_in_combined_loss, class_weights |
| `code/config/g7_edge_sampling_candidate.json` | Predeclared edge crop search and Train exposure gates, not model training config; JSON fields: status, seed, epochs, patch_xyz, patches_per_patient_epoch, preserved_uniform_draws, edge, training_gate, baseline_geometry_counts, scope, fallback, negative_definition | 预先定义边缘裁剪搜索及训练曝光门槛，非模型训练配置；JSON 字段：status, seed, epochs, patch_xyz, patches_per_patient_epoch, preserved_uniform_draws, edge, training_gate, baseline_geometry_counts, scope, fallback, negative_definition |
| `code/config/p2_inference_context.json` | JSON fields: experiment_id, status, split, cases, case_selection, models, checkpoint, window_xyz, overlap_fraction, fusion, preprocessing, padding | JSON 字段：experiment_id, status, split, cases, case_selection, models, checkpoint, window_xyz, overlap_fraction, fusion, preprocessing, padding |
| `code/config/p2b_prediction_mechanisms.json` | JSON fields: experiment_id, cases, split, checkpoints, window_xyz, overlap, fusion, replay, self_replay_max_abs_logit_error, self_replay_max_argmax_disagreement_fraction, mechanism_gate, confidence_bins_descriptive_only | JSON 字段：experiment_id, cases, split, checkpoints, window_xyz, overlap, fusion, replay, self_replay_max_abs_logit_error, self_replay_max_argmax_disagreement_fraction, mechanism_gate, confidence_bins_descriptive_only |
| `code/config/patient_split.json` | Frozen patient groups and expected counts; JSON fields: provenance, train, validation, test, expected_volumes | 固定患者分组与预期数量；JSON 字段：provenance, train, validation, test, expected_volumes |
| `code/config/v2_independent_supervision_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/v2_transfer_checksums.json` | JSON fields: hipmri_rewrite/improved_v2.py, tools/train_v2_cohort.py, config/v2_independent_supervision_frozen.json, tests/test_improved_v2.py, tools/submit_v2_50.sh, docs/V2_TRAINING.md | JSON 字段：hipmri_rewrite/improved_v2.py, tools/train_v2_cohort.py, config/v2_independent_supervision_frozen.json, tests/test_improved_v2.py, tools/submit_v2_50.sh, docs/V2_TRAINING.md |
| `code/config/v3_gn1_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/v3_transfer_checksums.json` | JSON fields: hipmri_rewrite/improved_v3.py, tools/train_v3_cohort.py, config/v3_gn1_frozen.json, tests/test_improved_v3.py, tools/submit_v3_50.sh, docs/V3_TRAINING.md | JSON 字段：hipmri_rewrite/improved_v3.py, tools/train_v3_cohort.py, config/v3_gn1_frozen.json, tests/test_improved_v3.py, tools/submit_v3_50.sh, docs/V3_TRAINING.md |
| `code/config/v4_a0_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/v4_a1_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `code/config/v4_transfer_checksums.json` | JSON fields: hipmri_rewrite/augmentation_v4.py, tools/train_v4_cohort.py, tools/audit_v4_augmentation.py, tests/test_v4_training.py, config/v4_a0_frozen.json, config/v4_a1_frozen.json, tools/submit_v4_paired.sh, docs/V4_Paired_Training_ZH.md | JSON 字段：hipmri_rewrite/augmentation_v4.py, tools/train_v4_cohort.py, tools/audit_v4_augmentation.py, tests/test_v4_training.py, config/v4_a0_frozen.json, config/v4_a1_frozen.json, tools/submit_v4_paired.sh, docs/V4_Paired_Training_ZH.md |
| `code/docs/ALL_MODEL_ASSESSMENT_AUDIT.md` | Sections: Seven-model fixed-checkpoint assessment audit; Measurements; Execution and storage; Assessment limitations | 章节标题：Seven-model fixed-checkpoint assessment audit; Measurements; Execution and storage; Assessment limitations |
| `code/docs/BEST_CHECKPOINT_AUDIT.md` | Sections: Best-checkpoint assessment audit | 章节标题：Best-checkpoint assessment audit |
| `code/docs/INFERENCE_CONTEXT_DIAGNOSTIC_ZH.md` | Sections: 固定权重推理范围诊断; 当前状态; 固定设计; 运行与输出; 结果解释关口 | 章节标题：固定权重推理范围诊断; 当前状态; 固定设计; 运行与输出; 结果解释关口 |
| `code/docs/METHODS_EXECUTION_HISTORY.md` | Sections: Methods and execution history — reference, not a submission queue; 原文：SOURCE_DECISIONS.md; Paper and assignment decision register; 原文：EXECUTION.md; Execution and evidence checklist | 章节标题：Methods and execution history — reference, not a submission queue; 原文：SOURCE_DECISIONS.md; Paper and assignment decision register; 原文：EXECUTION.md; Execution and evidence checklist |
| `code/docs/PREDICTION_MECHANISM_DIAGNOSTIC_ZH.md` | Sections: 远处误检预测机制诊断; 状态与研究问题; 已核对的推理与 Deep Supervision; 固定设计与判别标准; 新测量与后处理 | 章节标题：远处误检预测机制诊断; 状态与研究问题; 已核对的推理与 Deep Supervision; 固定设计与判别标准; 新测量与后处理 |
| `code/docs/README.md` | Entry point or directory guide; Sections: 方法与执行文档 | 项目入口或目录说明；章节标题：方法与执行文档 |
| `code/docs/STUDY_PROTOCOL.md` | Research hypotheses, endpoints, and freeze rules; Sections: HipMRI research protocol — data-first revision; Current execution scope — 20 October 2026; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints | 研究假设、终点指标与冻结规则；章节标题：HipMRI research protocol — data-first revision; Current execution scope — 20 October 2026; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints |
| `code/docs/V2_TRAINING.md` | Sections: Improved v2 independent-supervision run; Frozen experiment; Submission and recovery on Rangpur; Local verification | 章节标题：Improved v2 independent-supervision run; Frozen experiment; Submission and recovery on Rangpur; Local verification |
| `code/docs/V3_TRAINING.md` | Sections: Improved v3 GN1 experiment; Identity and storage; Submission on Rangpur; Evaluation; Local checks and limits | 章节标题：Improved v3 GN1 experiment; Identity and storage; Submission on Rangpur; Evaluation; Local checks and limits |
| `code/docs/V4_Paired_Training_ZH.md` | Sections: V4 A0/A1 配对训练协议; 两个实验; 几何与曝光; 评价与完成标准; 恢复与日志 | 章节标题：V4 A0/A1 配对训练协议; 两个实验; 几何与曝光; 评价与完成标准; 恢复与日志 |
| `code/hipmri_rewrite/__init__.py` | Six-class package constants; Definitions: module-level code | 六类别包常量；定义：module-level code |
| `code/hipmri_rewrite/audit.py` | Train/Validation-only real NIfTI preflight; Definitions: audit, main | 仅针对训练与验证数据的真实 NIfTI 预检；定义：audit, main |
| `code/hipmri_rewrite/augmentation_v4.py` | Definitions: learning_rate, make_optimizer, paired_patch | 定义：learning_rate, make_optimizer, paired_patch |
| `code/hipmri_rewrite/boundaries.py` | Physical base/apex, slice and surface error analysis; Definitions: prostate_axial_regions, slice_dice, symmetric_surface_distances, prostate_organ_confusion | 物理位置的基底/尖端、切片和表面误差分析；定义：prostate_axial_regions, slice_dice, symmetric_surface_distances, prostate_organ_confusion |
| `code/hipmri_rewrite/data.py` | Paired loading, common-grid downsampling and mask encoding; Definitions: Volume, load_case, downsample, zscore_nonzero, one_hot, to_3d_model_arrays, from_3d_model_prediction | 配对读取、共用网格降采样及标签编码；定义：Volume, load_case, downsample, zscore_nonzero, one_hot, to_3d_model_arrays, from_3d_model_prediction |
| `code/hipmri_rewrite/improved_v2.py` | Definitions: ImprovedUNet3DV2, supervised_loss, state_digest | 定义：ImprovedUNet3DV2, supervised_loss, state_digest |
| `code/hipmri_rewrite/improved_v3.py` | Definitions: ImprovedUNet3DV3, state_digest | 定义：ImprovedUNet3DV3, state_digest |
| `code/hipmri_rewrite/inference.py` | 2D/3D predictions restored to the input XYZ grid; Definitions: predict_volume | 将 2D/3D 预测还原至输入 XYZ 网格；定义：predict_volume |
| `code/hipmri_rewrite/inference_context.py` | Definitions: axis_starts, infer_logits | 定义：axis_starts, infer_logits |
| `code/hipmri_rewrite/losses.py` | Six-class cross-entropy and soft-Dice objectives; Definitions: _check, foreground_soft_dice_loss, ce_dice_loss | 六类别交叉熵与 Soft Dice 目标函数；定义：_check, foreground_soft_dice_loss, ce_dice_loss |
| `code/hipmri_rewrite/metrics.py` | Confusion counts, Dice, IoU, precision and recall; Definitions: confusion, overlap | 混淆计数、Dice、IoU、精确率与召回率；定义：confusion, overlap |
| `code/hipmri_rewrite/models/__init__.py` | Six-class package constants; Definitions: module-level code | 六类别包常量；定义：module-level code |
| `code/hipmri_rewrite/models/baseline_2d.py` | Slice-wise 2D U-Net baseline; Definitions: UNet2D | 逐切片 2D U-Net 基线；定义：UNet2D |
| `code/hipmri_rewrite/models/baseline_3d.py` | Standard volumetric 3D U-Net baseline; Definitions: UNet3D | 标准体数据 3D U-Net 基线；定义：UNet3D |
| `code/hipmri_rewrite/models/common.py` | Shared standard U-Net convolutional blocks; Definitions: _DoubleConv, _StandardUNet | 标准 U-Net 共用卷积模块；定义：_DoubleConv, _StandardUNet |
| `code/hipmri_rewrite/models/improved_3d.py` | Residual 3D U-Net with deep supervision; Definitions: _ContextResidual, _Localization, ImprovedUNet3D | 含残差结构及深监督的 3D U-Net；定义：_ContextResidual, _Localization, ImprovedUNet3D |
| `code/hipmri_rewrite/prediction_diagnostics.py` | Definitions: model_digest, capture_instance_stats, replay_instance_stats, quantiles, confidence, validate_grid, geometry, conservative_component_filter | 定义：model_digest, capture_instance_stats, replay_instance_stats, quantiles, confidence, validate_grid, geometry, conservative_component_filter |
| `code/hipmri_rewrite/profile.py` | Train-only GPU forward/backward capacity probe; Definitions: profile, main | 仅使用训练数据的 GPU 前反向容量检查；定义：profile, main |
| `code/hipmri_rewrite/sampling.py` | Patient-balanced Train patch selection and exact label exposure; Definitions: _rng, epoch_cases, crop_origin, crop_for_3d_model, ExposureRecorder | 按患者均衡抽取训练 Patch 并记录实际标签曝光；定义：_rng, epoch_cases, crop_origin, crop_for_3d_model, ExposureRecorder |
| `code/hipmri_rewrite/sampling_2d.py` | Deterministic Train-only axial crop selection for the optional 2D arm; Definitions: slice_crop_origin, crop_for_2d_model | 可选 2D 模型使用的确定性训练集轴向切片裁剪；定义：slice_crop_origin, crop_for_2d_model |
| `code/hipmri_rewrite/sampling_edges.py` | Candidate Train corner crops with physical hard negatives and uniform fallback; Definitions: boundary_hits, EdgeSelection, EdgePatchSelector | 候选训练集角落裁剪，含物理距离负样本及均匀回退；定义：boundary_hits, EdgeSelection, EdgePatchSelector |
| `code/hipmri_rewrite/splits.py` | Frozen patient identity and MRI/mask file pairing; Definitions: Case, read_split, discover | 固定患者身份及 MRI/mask 文件配对；定义：Case, read_split, discover |
| `code/hipmri_rewrite/training.py` | Predeclared equal-patient Validation selection score; Definitions: patient_small_organ_score | 预先定义患者等权的验证集模型选择指标；定义：patient_small_organ_score |
| `code/README.md` | Entry point or directory guide; Sections: Code and Methods; Task-oriented entry points; Current documentation entry points — 20 October 2026 | 项目入口或目录说明；章节标题：Code and Methods; Task-oriented entry points; Current documentation entry points — 20 October 2026 |
| `code/requirements.txt` | Python dependency ranges; UTF-8 text with 5 lines; 79 bytes | Python 依赖版本范围；UTF-8 文本，5 行；79 字节 |
| `code/tests/test_all_model_audit.py` | Definitions: GeneralizationAuditTests | 定义：GeneralizationAuditTests |
| `code/tests/test_best_checkpoint_audit.py` | Definitions: GeneralizationAuditTests | 定义：GeneralizationAuditTests |
| `code/tests/test_boundaries.py` | Physical direction and empty-surface tests; Definitions: BoundaryTests, SurfaceTests | 物理方向及空表面测试；定义：BoundaryTests, SurfaceTests |
| `code/tests/test_data.py` | NIfTI alignment, downsampling and label tests; Definitions: DataContractTests, DownsamplingGeometryTests, NiftiPairTests | NIfTI 对齐、降采样与标签测试；定义：DataContractTests, DownsamplingGeometryTests, NiftiPairTests |
| `code/tests/test_g5_candidate.py` | Check paired G5 candidate budget and Train-only weight provenance; Definitions: G5CandidateTests | 核验 G5 双模型候选预算及仅训练集权重来源；定义：G5CandidateTests |
| `code/tests/test_improved_v2.py` | Definitions: V2Tests | 定义：V2Tests |
| `code/tests/test_improved_v3.py` | Definitions: V3Tests | 定义：V3Tests |
| `code/tests/test_inference_context.py` | Definitions: Pointwise, ContextTests | 定义：Pointwise, ContextTests |
| `code/tests/test_metrics.py` | Class-confusion and overlap metric tests; Definitions: OverlapTests | 类别混淆与重叠指标测试；定义：OverlapTests |
| `code/tests/test_models_and_losses.py` | Shape, gradient, supervision and inference tests; Definitions: NetworkTests | 形状、梯度、深监督与推理测试；定义：NetworkTests |
| `code/tests/test_prediction_diagnostics.py` | Definitions: PredictionDiagnosticTests | 定义：PredictionDiagnosticTests |
| `code/tests/test_raw_audit.py` | Train inventory and file-hash audit tests; Definitions: RawAuditInventoryTests | 训练集文件盘点与文件哈希审计测试；定义：RawAuditInventoryTests |
| `code/tests/test_sampling.py` | Patient cycle, paired crop and exposure-record tests; Definitions: case, PatientBalancedSamplingTests | 患者轮换、配对裁剪及曝光记录测试；定义：case, PatientBalancedSamplingTests |
| `code/tests/test_sampling_2d.py` | 2D crop target, bounds and MRI/mask alignment checks; Definitions: TwoDSamplingTests | 核验 2D 裁剪目标、边界和影像标签对齐；定义：TwoDSamplingTests |
| `code/tests/test_sampling_audit.py` | Candidate crop bounds and repeatability checks; Definitions: CandidateCropTests | 候选裁剪边界与可重复性检查；定义：CandidateCropTests |
| `code/tests/test_sampling_edges.py` | Verify physical negatives, paired geometry, Train isolation and uniform fallback; Definitions: volume, selector, EdgeSamplingTests | 核验物理距离负样本、配对几何、训练集隔离及均匀回退；定义：volume, selector, EdgeSamplingTests |
| `code/tests/test_splits.py` | Patient isolation and file pairing tests; Definitions: PatientSplitTests | 患者隔离与文件配对测试；定义：PatientSplitTests |
| `code/tests/test_training_recovery.py` | Check atomic batch checkpoints and recovery after an interrupted epoch pointer; Definitions: RecoveryTests | 核验原子批次权重及轮次指针中断后的恢复；定义：RecoveryTests |
| `code/tests/test_training_selection.py` | Check patient-balanced small-organ checkpoint selection; Definitions: PatientSelectionTests | 核验患者等权的小器官权重选择规则；定义：PatientSelectionTests |
| `code/tests/test_v4_training.py` | Definitions: V4Tests | 定义：V4Tests |
| `code/tools/audit_all_models.py` | Definitions: sha, write, geometry, diagnostics, load_model, panel, body_three_planes, main | 定义：sha, write, geometry, diagnostics, load_model, panel, body_three_planes, main |
| `code/tools/audit_best_checkpoints.py` | Definitions: sha, write, geometry, diagnostics, load_model, panel, main | 定义：sha, write, geometry, diagnostics, load_model, panel, main |
| `code/tools/audit_g2_train_slices.py` | Measure Train axial label coverage and MRI intensity bins on Rangpur CPU; Definitions: sha256_file, write_csv, main | 在 Rangpur CPU 测量训练集逐切片标签覆盖与 MRI 强度分箱；定义：sha256_file, write_csv, main |
| `code/tools/audit_g3_candidate_sampling.py` | Compare fixed Train-only uniform and organ-aware candidate crops; Definitions: sha256_file, fixed_rng, uniform_origin, target_origin, patch_counts, main | 比较固定的训练集均匀与器官感知候选裁剪；定义：sha256_file, fixed_rng, uniform_origin, target_origin, patch_counts, main |
| `code/tools/audit_g3_existing_pipeline.py` | Audit current Train downsampling, normalization and whole-volume arrays; Definitions: sha256_file, class_counts, occupied_z_slices, software_versions, main | 审计当前训练集降采样、归一化与整卷张量；定义：sha256_file, class_counts, occupied_z_slices, software_versions, main |
| `code/tools/audit_g4_patient_sampling.py` | Dry-run patient-balanced model-ready Train patches on CPU; Definitions: sha256_file, main | 在 CPU 上预演患者均衡的训练模型输入 Patch；定义：sha256_file, main |
| `code/tools/audit_g5_2d_sampling.py` | CPU audit of patient-balanced uniform and organ-aware 2D slice exposure; Definitions: sha256_file, main | 在 CPU 审计按患者均衡的 2D 均匀及器官感知切片曝光；定义：sha256_file, main |
| `code/tools/audit_g5_aux_logits_validation.py` | Compare selected Improved 3D summed and main-only logits on Validation; Definitions: predictions, main | 在验证集比较选定 Improved 3D 的相加输出和单独主输出；定义：predictions, main |
| `code/tools/audit_g5_train_patch_geometry.py` | Reconstruct frozen 3D Train patch face and corner coverage; Definitions: digest, touches, main | 重建固定方案的 3D 训练 Patch 边缘及角落覆盖；定义：digest, touches, main |
| `code/tools/audit_g5_validation_body.py` | Measure fixed-checkpoint Validation Body boundaries and draw outliers; Definitions: draw, main | 测量固定权重的验证集 Body 边界并绘制异常案例；定义：draw, main |
| `code/tools/audit_g7_edge_sampling.py` | Replay frozen Train epochs and compare 9+1 edge sampling exposure on CPU; Definitions: sha, main | 在 CPU 重放固定训练轮次并比较 9+1 边缘采样曝光；定义：sha, main |
| `code/tools/audit_local_generalization.py` | Definitions: main | 定义：main |
| `code/tools/audit_markdown.py` | Definitions: classify, main | 定义：classify, main |
| `code/tools/audit_p2_inference_context.py` | Definitions: stats, xyz_logits, border_distance, components, measure, margin_comparison, draw, main | 定义：stats, xyz_logits, border_distance, components, measure, margin_comparison, draw, main |
| `code/tools/audit_p2b_prediction_mechanisms.py` | Definitions: dump, filter_figure, main | 定义：dump, filter_figure, main |
| `code/tools/audit_p2b_replay_numerics.py` | Definitions: replay_layer, inspect_layer, capture_layers, replay_layers, differences, main, null_context | 定义：replay_layer, inspect_layer, capture_layers, replay_layers, differences, main, null_context |
| `code/tools/audit_train_raw.py` | Train-only raw MRI/mask integrity audit; Definitions: train_inventory, sha256_file, audit_case, main | 仅对训练集原始 MRI/Mask 做完整性审计；定义：train_inventory, sha256_file, audit_case, main |
| `code/tools/audit_v4_augmentation.py` | Definitions: main | 定义：main |
| `code/tools/build_best_checkpoint_review.py` | Definitions: read, table, save, main | 定义：read, table, save, main |
| `code/tools/evaluate_g6_test.py` | Definitions: sha256, selected_model, boundary_summary, main | 定义：sha256, selected_model, boundary_summary, main |
| `code/tools/organize_result_artifacts.py` | Definitions: destination, main | 定义：destination, main |
| `code/tools/package_transfer.py` | Build and verify a source-only transfer ZIP; Definitions: main | 创建并校验仅含源码的传输包；定义：main |
| `code/tools/package_v4_paired.py` | Definitions: module-level code | 定义：module-level code |
| `code/tools/probe_g5_full_volume.py` | Measure full-volume Train inference geometry, time and A100 memory with random weights; Definitions: sha256_file, main | 用随机权重测量完整训练体积推理的几何、时间及 A100 显存；定义：sha256_file, main |
| `code/tools/probe_g5_patch_gpu.py` | Measure both 3D models' real Train patch gradients and A100 memory; Definitions: sha256_file, main | 测量两个 3D 模型真实训练 Patch 的梯度和 A100 显存；定义：sha256_file, main |
| `code/tools/probe_g5_training_smoke.py` | Check paired two-patient weighted Train batches, gradients and actual exposure without a checkpoint; Definitions: sha256_file, check_config, main | 不保存权重，检查双模型两患者加权训练批次、梯度和实际曝光；定义：sha256_file, check_config, main |
| `code/tools/review_g1_bladder_islands.py` | Locate and visualize two flagged Train Bladder components; Definitions: inspect_case, main | 定位并可视化两例训练集膀胱小连通区域；定义：inspect_case, main |
| `code/tools/review_g1_overlays.py` | Render raw Train MRI and ground-truth overlays for G1 review; Definitions: render_case, main | 绘制训练集原始 MRI 与真实标签叠加图供 G1 审查；定义：render_case, main |
| `code/tools/review_g1_train_contact_sheets.py` | Render two-slice Train patient sheets for G1 alignment screening; Definitions: render_patient, main | 绘制训练集患者双切面图供 G1 对齐筛查；定义：render_patient, main |
| `code/tools/review_g5_validation_failures.py` | Definitions: sha256, load_selected, selected_slice, render, main | 定义：sha256, load_selected, selected_slice, render, main |
| `code/tools/review_g6_body_boundaries.py` | Reproduce selected Test cases and draw Body boundary errors; Definitions: planes, draw, main | 复现选定测试案例并绘制 Body 边界错误；定义：planes, draw, main |
| `code/tools/review_g6_test_cases.py` | Reproduce selected held-out Test 3D predictions and draw descriptive overlays; Definitions: axial_slice, render, main | 复现选定独立测试样本的 3D 预测并绘制描述性叠加图；定义：axial_slice, render, main |
| `code/tools/review_v2_validation.py` | Definitions: sha, main | 定义：sha, main |
| `code/tools/submit_all_model_audit.sh` | UTF-8 text with 19 lines; 1,932 bytes | UTF-8 文本，19 行；1,932 字节 |
| `code/tools/submit_best_checkpoint_audit.sh` | UTF-8 text with 26 lines; 1,820 bytes | UTF-8 文本，26 行；1,820 字节 |
| `code/tools/submit_g5_2d.sh` | UTF-8 text with 77 lines; 3,122 bytes | UTF-8 文本，77 行；3,122 字节 |
| `code/tools/submit_g5_2d_extend_50.sh` | Verify and queue a frozen 2D continuation to epoch 50; UTF-8 text with 81 lines; 2,914 bytes | 核验并排队固定配置的 2D 模型续训至 50 轮；UTF-8 文本，81 行；2,914 字节 |
| `code/tools/submit_g5_40.sh` | Submit CPU recovery check and dependency-gated Standard/Improved 40-epoch A100 jobs; UTF-8 text with 95 lines; 3,818 bytes | 提交 CPU 恢复测试及依赖测试通过的双模型 40 轮 A100 作业；UTF-8 文本，95 行；3,818 字节 |
| `code/tools/submit_g5_extend_50.sh` | UTF-8 text with 109 lines; 4,866 bytes | UTF-8 文本，109 行；4,866 字节 |
| `code/tools/submit_g6_test.sh` | UTF-8 text with 35 lines; 1,550 bytes | UTF-8 文本，35 行；1,550 字节 |
| `code/tools/submit_g7_edge_audit.sh` | Submit smoke or smoke-gated full Train-only CPU edge audit; UTF-8 text with 66 lines; 2,607 bytes | 提交小规模或通过小规模门槛的完整训练集 CPU 边缘审计；UTF-8 文本，66 行；2,607 字节 |
| `code/tools/submit_p2_inference_context.sh` | UTF-8 text with 34 lines; 1,365 bytes | UTF-8 文本，34 行；1,365 字节 |
| `code/tools/submit_p2b_prediction_mechanisms.sh` | UTF-8 text with 32 lines; 1,320 bytes | UTF-8 文本，32 行；1,320 字节 |
| `code/tools/submit_v2_50.sh` | UTF-8 text with 34 lines; 2,156 bytes | UTF-8 文本，34 行；2,156 字节 |
| `code/tools/submit_v3_50.sh` | UTF-8 text with 34 lines; 2,140 bytes | UTF-8 文本，34 行；2,140 字节 |
| `code/tools/submit_v4_paired.sh` | UTF-8 text with 35 lines; 2,672 bytes | UTF-8 文本，35 行；2,672 字节 |
| `code/tools/summarize_all_class_review.py` | Verify selected records and draw six-class, training and failure figures; Definitions: save, paired_bars, validation_figures, test_figures_and_table, three_arm_validation_figures, main | 核验选定记录并绘制六类、训练及失误图；定义：save, paired_bars, validation_figures, test_figures_and_table, three_arm_validation_figures, main |
| `code/tools/summarize_all_model_audit.py` | Definitions: sha, read, write, verify, table, save, main | 定义：sha, read, write, verify, table, save, main |
| `code/tools/summarize_all_model_validation.py` | Definitions: main | 定义：main |
| `code/tools/summarize_best_checkpoint_audit.py` | Definitions: main | 定义：main |
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
| `code/tools/summarize_p2_inference_context.py` | Definitions: main | 定义：main |
| `code/tools/summarize_p2b_prediction_mechanisms.py` | Definitions: sha, checked, existing, replay | 定义：sha, checked, existing, replay |
| `code/tools/summarize_v2_results.py` | Definitions: read, metrics, patient_matrices, score, table, save, main | 定义：read, metrics, patient_matrices, score, table, save, main |
| `code/tools/summarize_v3_results.py` | Definitions: read, metrics, patient_matrices, score, table, main | 定义：read, metrics, patient_matrices, score, table, main |
| `code/tools/summarize_v4_pair.py` | Definitions: write_csv, main | 定义：write_csv, main |
| `code/tools/tiny_overfit_g5.py` | Check both 3D models can learn two fixed Train patches on GPU; Definitions: sha256_file, measure, main | 检查两个 3D 模型能否在 GPU 上学习两个固定训练 Patch；定义：sha256_file, measure, main |
| `code/tools/train_g5_2d.py` | Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main | 定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main |
| `code/tools/train_g5_cohort.py` | Train a frozen 3D arm with patient-balanced exposure, Validation selection and resumable checkpoints; Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main | 按固定方案训练 3D 模型，记录患者均衡曝光、验证集选择及可恢复权重；定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, main |
| `code/tools/train_v2_cohort.py` | Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main | 定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main |
| `code/tools/train_v3_cohort.py` | Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main | 定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main |
| `code/tools/train_v4_cohort.py` | Definitions: sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main | 定义：sha256_file, atomic_json, atomic_checkpoint, load_durable_state, exclusive_run_lock, main |
| `code/tools/update_project_map.py` | Generate and watch this bilingual inventory; Definitions: _files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main | 生成并持续更新本双语目录；定义：_files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main |
| `code/transfer/20261017_source.zip` | Generated source-only transfer package; ZIP archive with 66 members; 52,512 bytes | 生成的仅含源码传输包；ZIP 压缩包，含 66 个成员；52,512 字节 |
| `code/transfer/all_model_assessment_20261021.zip` | ZIP archive with 6 members; 12,984 bytes | ZIP 压缩包，含 6 个成员；12,984 字节 |
| `code/transfer/best_checkpoint_audit_current_20261020.zip` | ZIP archive with 7 members; 14,111 bytes | ZIP 压缩包，含 7 个成员；14,111 字节 |
| `code/transfer/v4_paired_spatial_20261020.zip` | ZIP archive with 9 members; 22,875 bytes | ZIP 压缩包，含 9 个成员；22,875 字节 |
| `DOCUMENTATION_INDEX.md` | Sections: HipMRI 文档导航; 1. 建议阅读顺序; 2. 分类与命名规则; 3. 当前、历史与固定快照; 4. 更新方式 | 章节标题：HipMRI 文档导航; 1. 建议阅读顺序; 2. 分类与命名规则; 3. 当前、历史与固定快照; 4. 更新方式 |
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
| `logs/validation/inference_context/p2_four_cases_20261020/B040_Week4_improved_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/B040_Week4_standard_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/checksums.json` | JSON fields: B040_Week4_improved_3d.json, B040_Week4_standard_3d.json, J026_Week4_improved_3d.json, J026_Week4_standard_3d.json, K008_Week0_improved_3d.json, K008_Week0_standard_3d.json, K018_Week4_improved_3d.json, K018_Week4_standard_3d.json, manifest.json | JSON 字段：B040_Week4_improved_3d.json, B040_Week4_standard_3d.json, J026_Week4_improved_3d.json, J026_Week4_standard_3d.json, K008_Week0_improved_3d.json, K008_Week0_standard_3d.json, K018_Week4_improved_3d.json, K018_Week4_standard_3d.json, manifest.json |
| `logs/validation/inference_context/p2_four_cases_20261020/J026_Week4_improved_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/J026_Week4_standard_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/K008_Week0_improved_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/K008_Week0_standard_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/K018_Week4_improved_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/K018_Week4_standard_3d.json` | JSON fields: case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference | JSON 字段：case, patient, model, shape_xyz, spacing_mm, affine, input_sha256, full_confusion_reproduced, protocols, changed_prediction_voxels, same_voxel_margins, shared_reference |
| `logs/validation/inference_context/p2_four_cases_20261020/manifest.json` | Binary or large file; 244,703 bytes | 二进制或大文件；244,703 字节 |
| `logs/validation/README.md` | Entry point or directory guide; Sections: Validation 诊断原始记录 | 项目入口或目录说明；章节标题：Validation 诊断原始记录 |
| `PROJECT_MAP.md` | Generated bilingual tree and file content index; Self-entry; regenerated from current tree and file structure | 自动生成的双语文件树与内容索引；本文件条目；根据当前文件树和文件结构重新生成 |
| `README.md` | Entry point or directory guide; Sections: HipMRI Segmentation Research; 当前进度; 阅读入口; GitHub中的内容; Artificial Intelligence Usage Disclosure | 项目入口或目录说明；章节标题：HipMRI Segmentation Research; 当前进度; 阅读入口; GitHub中的内容; Artificial Intelligence Usage Disclosure |
| `results/case_studies/comparisons/README.md` | Entry point or directory guide; Sections: 分割案例索引; Selected segmentation overlays | 项目入口或目录说明；章节标题：分割案例索引; Selected segmentation overlays |
| `results/case_studies/comparisons/test/body/g6_body_L011_Week5.png` | Binary or large file; 218,157 bytes | 二进制或大文件；218,157 字节 |
| `results/case_studies/comparisons/test/body/g6_body_M015_Week3.png` | Binary or large file; 266,760 bytes | 二进制或大文件；266,760 字节 |
| `results/case_studies/comparisons/test/body/g6_body_selected_cases.json` | JSON fields: kind, split, test_result_sha256, cases | JSON 字段：kind, split, test_result_sha256, cases |
| `results/case_studies/comparisons/test/body/g6_body_W041_Week0.png` | Binary or large file; 244,148 bytes | 二进制或大文件；244,148 字节 |
| `results/case_studies/comparisons/test/body/README.md` | Entry point or directory guide; Sections: Test Body 边界 | 项目入口或目录说明；章节标题：Test Body 边界 |
| `results/case_studies/comparisons/test/small_organs/g6_test_M015_Week7_bladder.png` | Binary or large file; 130,570 bytes | 二进制或大文件；130,570 字节 |
| `results/case_studies/comparisons/test/small_organs/g6_test_M020_Week1_prostate.png` | Binary or large file; 94,741 bytes | 二进制或大文件；94,741 字节 |
| `results/case_studies/comparisons/test/small_organs/g6_test_selected_cases.json` | JSON fields: kind, split, test_result_sha256, cases | JSON 字段：kind, split, test_result_sha256, cases |
| `results/case_studies/comparisons/test/small_organs/g6_test_W029_Week6_prostate.png` | Binary or large file; 83,023 bytes | 二进制或大文件；83,023 字节 |
| `results/case_studies/comparisons/test/small_organs/README.md` | Entry point or directory guide; Sections: Test 小器官案例 | 项目入口或目录说明；章节标题：Test 小器官案例 |
| `results/case_studies/comparisons/validation/body/g5_body_validation_B040_Week4.png` | Binary or large file; 268,835 bytes | 二进制或大文件；268,835 字节 |
| `results/case_studies/comparisons/validation/body/g5_body_validation_J026_Week4.png` | Binary or large file; 222,090 bytes | 二进制或大文件；222,090 字节 |
| `results/case_studies/comparisons/validation/body/g5_body_validation_K018_Week4.png` | Binary or large file; 249,121 bytes | 二进制或大文件；249,121 字节 |
| `results/case_studies/comparisons/validation/body/README.md` | Entry point or directory guide; Sections: Validation Body 边界 | 项目入口或目录说明；章节标题：Validation Body 边界 |
| `results/case_studies/comparisons/validation/small_organs/g5_validation_B040_Week1_prostate.png` | Binary or large file; 99,219 bytes | 二进制或大文件；99,219 字节 |
| `results/case_studies/comparisons/validation/small_organs/g5_validation_J026_Week1_rectum.png` | Binary or large file; 100,764 bytes | 二进制或大文件；100,764 字节 |
| `results/case_studies/comparisons/validation/small_organs/g5_validation_K018_Week3_rectum.png` | Binary or large file; 89,639 bytes | 二进制或大文件；89,639 字节 |
| `results/case_studies/comparisons/validation/small_organs/g5_validation_S022_Week6_prostate.png` | Binary or large file; 89,736 bytes | 二进制或大文件；89,736 字节 |
| `results/case_studies/comparisons/validation/small_organs/g5_validation_selected_cases.json` | JSON fields: kind, split, cases, sources | JSON 字段：kind, split, cases, sources |
| `results/case_studies/comparisons/validation/small_organs/README.md` | Entry point or directory guide; Sections: Validation 小器官案例 | 项目入口或目录说明；章节标题：Validation 小器官案例 |
| `results/data_analysis/boundary_sampling_audit/class_exposure.csv` | Table with 12 data rows; columns: ﻿policy, class, patch_hits, patch_hit_rate, patch_voxels, voxel_share | 表格含 12 条数据行；列：﻿policy, class, patch_hits, patch_hit_rate, patch_voxels, voxel_share |
| `results/data_analysis/boundary_sampling_audit/crop_origin_distribution.png` | Binary or large file; 214,581 bytes | 二进制或大文件；214,581 字节 |
| `results/data_analysis/boundary_sampling_audit/crop_origin_distribution.svg` | Binary or large file; 175,989 bytes | 二进制或大文件；175,989 字节 |
| `results/data_analysis/boundary_sampling_audit/patient_class_exposure.csv` | Table with 312 data rows; columns: ﻿policy, patient, class, patches, patch_hits, patch_voxels, voxel_share | 表格含 312 条数据行；列：﻿policy, patient, class, patches, patch_hits, patch_voxels, voxel_share |
| `results/data_analysis/boundary_sampling_audit/README.md` | Entry point or directory guide; Sections: Boundary Sampling Feasibility | 项目入口或目录说明；章节标题：Boundary Sampling Feasibility |
| `results/data_analysis/boundary_sampling_audit/sampling_exposure_and_geometry.png` | Binary or large file; 138,685 bytes | 二进制或大文件；138,685 字节 |
| `results/data_analysis/boundary_sampling_audit/sampling_exposure_and_geometry.svg` | UTF-8 text with 2630 lines; 88,539 bytes | UTF-8 文本，2630 行；88,539 字节 |
| `results/data_analysis/boundary_sampling_audit/verification.json` | JSON fields: kind, source_hashes, csv_rows, preserved_draw_pairs, baseline_epochs_reproduced, accepted_edge_draws, eligible_under_provisional_gates, inference_or_model_quality_established | JSON 字段：kind, source_hashes, csv_rows, preserved_draw_pairs, baseline_epochs_reproduced, accepted_edge_draws, eligible_under_provisional_gates, inference_or_model_quality_established |
| `results/data_analysis/label_component_review/g1_bladder_island_D031_Week7.png` | Binary or large file; 250,160 bytes | 二进制或大文件；250,160 字节 |
| `results/data_analysis/label_component_review/g1_bladder_island_N010_Week3.png` | Binary or large file; 254,248 bytes | 二进制或大文件；254,248 字节 |
| `results/data_analysis/label_component_review/README.md` | Entry point or directory guide; Sections: Bladder Label Component Review | 项目入口或目录说明；章节标题：Bladder Label Component Review |
| `results/data_analysis/legacy_training_statistics/HipMRI_Pretraining_Data_Analysis.previous-41201a5e.bak` | UTF-8 text with 106 lines; 14,485 bytes | UTF-8 文本，106 行；14,485 字节 |
| `results/data_analysis/legacy_training_statistics/HipMRI_Pretraining_Data_Analysis.previous-5dc166bd.bak` | UTF-8 text with 93 lines; 12,305 bytes | UTF-8 文本，93 行；12,305 字节 |
| `results/data_analysis/legacy_training_statistics/historical_patch_vs_full_volume.svg` | UTF-8 text with 49 lines; 4,617 bytes | UTF-8 文本，49 行；4,617 字节 |
| `results/data_analysis/legacy_training_statistics/historical_resampling_volume_ratio.svg` | UTF-8 text with 768 lines; 59,609 bytes | UTF-8 文本，768 行；59,609 字节 |
| `results/data_analysis/legacy_training_statistics/historical_train_case_profile.csv` | Table with 143 data rows; columns: case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels | 表格含 143 条数据行；列：case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels |
| `results/data_analysis/legacy_training_statistics/historical_train_statistics.json` | JSON fields: scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts | JSON 字段：scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts |
| `results/data_analysis/legacy_training_statistics/README.md` | Entry point or directory guide; Sections: Historical Train statistics | 项目入口或目录说明；章节标题：Historical Train statistics |
| `results/data_analysis/legacy_training_statistics/train_class_distribution.svg` | UTF-8 text with 35 lines; 3,679 bytes | UTF-8 文本，35 行；3,679 字节 |
| `results/data_analysis/legacy_training_statistics/train_foreground_distribution.svg` | UTF-8 text with 32 lines; 3,368 bytes | UTF-8 文本，32 行；3,368 字节 |
| `results/data_analysis/legacy_training_statistics/train_patient_examinations.svg` | UTF-8 text with 72 lines; 7,370 bytes | UTF-8 文本，72 行；7,370 字节 |
| `results/data_analysis/legacy_training_statistics/train_rare_organ_volumes.svg` | UTF-8 text with 320 lines; 25,203 bytes | UTF-8 文本，320 行；25,203 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_B006.png` | Binary or large file; 263,728 bytes | 二进制或大文件；263,728 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_B038.png` | Binary or large file; 134,754 bytes | 二进制或大文件；134,754 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_C032.png` | Binary or large file; 1,001,604 bytes | 二进制或大文件；1,001,604 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_D031.png` | Binary or large file; 979,042 bytes | 二进制或大文件；979,042 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_G021.png` | Binary or large file; 794,896 bytes | 二进制或大文件；794,896 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_H007.png` | Binary or large file; 108,239 bytes | 二进制或大文件；108,239 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_H017.png` | Binary or large file; 895,730 bytes | 二进制或大文件；895,730 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_J005.png` | Binary or large file; 568,074 bytes | 二进制或大文件；568,074 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_K019.png` | Binary or large file; 1,185,795 bytes | 二进制或大文件；1,185,795 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_K042.png` | Binary or large file; 128,697 bytes | 二进制或大文件；128,697 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_M004.png` | Binary or large file; 1,218,442 bytes | 二进制或大文件；1,218,442 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_M013.png` | Binary or large file; 989,753 bytes | 二进制或大文件；989,753 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_M023.png` | Binary or large file; 123,435 bytes | 二进制或大文件；123,435 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_M036.png` | Binary or large file; 123,922 bytes | 二进制或大文件；123,922 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_N010.png` | Binary or large file; 1,179,116 bytes | 二进制或大文件；1,179,116 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_O025.png` | Binary or large file; 1,050,411 bytes | 二进制或大文件；1,050,411 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_R016.png` | Binary or large file; 1,037,894 bytes | 二进制或大文件；1,037,894 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_R024.png` | Binary or large file; 1,151,048 bytes | 二进制或大文件；1,151,048 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_R039.png` | Binary or large file; 1,029,032 bytes | 二进制或大文件；1,029,032 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_S028.png` | Binary or large file; 135,808 bytes | 二进制或大文件；135,808 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_S033.png` | Binary or large file; 1,144,701 bytes | 二进制或大文件；1,144,701 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_S035.png` | Binary or large file; 107,138 bytes | 二进制或大文件；107,138 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_T009.png` | Binary or large file; 120,824 bytes | 二进制或大文件；120,824 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_T014.png` | Binary or large file; 996,731 bytes | 二进制或大文件；996,731 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_V027.png` | Binary or large file; 887,276 bytes | 二进制或大文件；887,276 字节 |
| `results/data_analysis/patient_alignment_screening/g1_train_patient_W012.png` | Binary or large file; 1,046,773 bytes | 二进制或大文件；1,046,773 字节 |
| `results/data_analysis/patient_alignment_screening/README.md` | Entry point or directory guide; Sections: Cohort-Wide MRI–Label Alignment | 项目入口或目录说明；章节标题：Cohort-Wide MRI–Label Alignment |
| `results/data_analysis/patient_balanced_sampling/class_exposure.csv` | Table with 12 data rows; columns: ﻿policy, class, patches, patch_hits, patch_hit_rate, patch_voxels, voxel_share | 表格含 12 条数据行；列：﻿policy, class, patches, patch_hits, patch_hit_rate, patch_voxels, voxel_share |
| `results/data_analysis/patient_balanced_sampling/g5_train_patch_geometry.json` | JSON fields: kind, scope, g1_sha256, config_sha256, run_manifest_sha256, definition, patch_xyz, epochs, counts, rates, by_patient | JSON 字段：kind, scope, g1_sha256, config_sha256, run_manifest_sha256, definition, patch_xyz, epochs, counts, rates, by_patient |
| `results/data_analysis/patient_balanced_sampling/patient_balanced_exposure.svg` | UTF-8 text with 1714 lines; 51,622 bytes | UTF-8 文本，1714 行；51,622 字节 |
| `results/data_analysis/patient_balanced_sampling/patient_exposure.csv` | Table with 52 data rows; columns: ﻿policy, patient, case, patches, bladder_hits, rectum_hits, prostate_hits, bladder_voxel_share, rectum_voxel_share, prostate_voxel_share | 表格含 52 条数据行；列：﻿policy, patient, case, patches, bladder_hits, rectum_hits, prostate_hits, bladder_voxel_share, rectum_voxel_share, prostate_voxel_share |
| `results/data_analysis/patient_balanced_sampling/README.md` | Entry point or directory guide; Sections: Patient-Balanced Sampling Exposure | 项目入口或目录说明；章节标题：Patient-Balanced Sampling Exposure |
| `results/data_analysis/preprocessing_validation/candidate_patch_class_exposure.csv` | Table with 12 data rows; columns: ﻿arm, class, patches, patch_hits, patch_hit_pct, patch_voxels, voxel_share_pct, complete_organ_patches | 表格含 12 条数据行；列：﻿arm, class, patches, patch_hits, patch_hit_pct, patch_voxels, voxel_share_pct, complete_organ_patches |
| `results/data_analysis/preprocessing_validation/candidate_patch_exposure.svg` | UTF-8 text with 1709 lines; 53,196 bytes | UTF-8 文本，1709 行；53,196 字节 |
| `results/data_analysis/preprocessing_validation/candidate_patch_patient_exposure.csv` | Table with 52 data rows; columns: ﻿patient, arm, volumes, patches, bladder_hit_pct, rectum_hit_pct, prostate_hit_pct | 表格含 52 条数据行；列：﻿patient, arm, volumes, patches, bladder_hit_pct, rectum_hit_pct, prostate_hit_pct |
| `results/data_analysis/preprocessing_validation/organ_physical_volume_ratio.svg` | UTF-8 text with 1439 lines; 44,212 bytes | UTF-8 文本，1439 行；44,212 字节 |
| `results/data_analysis/preprocessing_validation/organ_preservation_by_volume.csv` | Table with 429 data rows; columns: ﻿case, patient, organ, raw_voxels, downsampled_voxels, raw_axial_slices, downsampled_axial_slices, raw_physical_volume_mm3, downsampled_physical_volume_mm3, physical_volume_ratio | 表格含 429 条数据行；列：﻿case, patient, organ, raw_voxels, downsampled_voxels, raw_axial_slices, downsampled_axial_slices, raw_physical_volume_mm3, downsampled_physical_volume_mm3, physical_volume_ratio |
| `results/data_analysis/preprocessing_validation/README.md` | Entry point or directory guide; Sections: Preprocessing and Sampling Validation | 项目入口或目录说明；章节标题：Preprocessing and Sampling Validation |
| `results/data_analysis/README.md` | Entry point or directory guide; Sections: no headings | 项目入口或目录说明；章节标题：no headings |
| `results/data_analysis/representative_alignment_review/g1_overlay_B038_Week0.png` | Binary or large file; 477,710 bytes | 二进制或大文件；477,710 字节 |
| `results/data_analysis/representative_alignment_review/g1_overlay_D031_Week7.png` | Binary or large file; 393,632 bytes | 二进制或大文件；393,632 字节 |
| `results/data_analysis/representative_alignment_review/g1_overlay_H017_Week7.png` | Binary or large file; 365,557 bytes | 二进制或大文件；365,557 字节 |
| `results/data_analysis/representative_alignment_review/g1_overlay_K019_Week1.png` | Binary or large file; 471,124 bytes | 二进制或大文件；471,124 字节 |
| `results/data_analysis/representative_alignment_review/g1_overlay_N010_Week3.png` | Binary or large file; 453,764 bytes | 二进制或大文件；453,764 字节 |
| `results/data_analysis/representative_alignment_review/g1_overlay_S033_Week1.png` | Binary or large file; 443,971 bytes | 二进制或大文件；443,971 字节 |
| `results/data_analysis/representative_alignment_review/README.md` | Entry point or directory guide; Sections: Representative MRI–Label Alignment | 项目入口或目录说明；章节标题：Representative MRI–Label Alignment |
| `results/data_analysis/training_distribution/class_distribution.csv` | Table with 6 data rows; columns: ﻿class, voxel_count, share_pct, denominator_voxels, train_volumes | 表格含 6 条数据行；列：﻿class, voxel_count, share_pct, denominator_voxels, train_volumes |
| `results/data_analysis/training_distribution/class_frequency_log.svg` | UTF-8 text with 544 lines; 20,424 bytes | UTF-8 文本，544 行；20,424 字节 |
| `results/data_analysis/training_distribution/intensity_case_summary.svg` | UTF-8 text with 675 lines; 28,992 bytes | UTF-8 文本，675 行；28,992 字节 |
| `results/data_analysis/training_distribution/intensity_statistics.csv` | Table with 143 data rows; columns: ﻿case, patient, finite_min, finite_median, finite_max, zero_fraction, nonzero_mean, nonzero_std, nonzero_p0_5, nonzero_p1, nonzero_p50, nonzero_p99 | 表格含 143 条数据行；列：﻿case, patient, finite_min, finite_median, finite_max, zero_fraction, nonzero_mean, nonzero_std, nonzero_p0_5, nonzero_p1, nonzero_p50, nonzero_p99 |
| `results/data_analysis/training_distribution/longitudinal_organ_variation.csv` | Table with 26 data rows; columns: ﻿patient, volume_count, bladder_min_ml, bladder_max_ml, bladder_max_min_ratio, rectum_min_ml, rectum_max_ml, rectum_max_min_ratio, prostate_min_ml, prostate_max_ml, prostate_max_min_ratio | 表格含 26 条数据行；列：﻿patient, volume_count, bladder_min_ml, bladder_max_ml, bladder_max_min_ratio, rectum_min_ml, rectum_max_ml, rectum_max_min_ratio, prostate_min_ml, prostate_max_ml, prostate_max_min_ratio |
| `results/data_analysis/training_distribution/organ_bbox_centers.svg` | UTF-8 text with 1210 lines; 95,981 bytes | UTF-8 文本，1210 行；95,981 字节 |
| `results/data_analysis/training_distribution/organ_slices_per_volume.svg` | UTF-8 text with 679 lines; 23,727 bytes | UTF-8 文本，679 行；23,727 字节 |
| `results/data_analysis/training_distribution/patient_rare_organ_share.svg` | UTF-8 text with 657 lines; 34,529 bytes | UTF-8 文本，657 行；34,529 字节 |
| `results/data_analysis/training_distribution/per_patient_distribution.csv` | Table with 26 data rows; columns: ﻿patient, volume_count, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct, rectum_voxels | 表格含 26 条数据行；列：﻿patient, volume_count, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct, rectum_voxels |
| `results/data_analysis/training_distribution/per_slice_class_coverage.csv` | Binary or large file; 1,494,705 bytes | 二进制或大文件；1,494,705 字节 |
| `results/data_analysis/training_distribution/per_volume_distribution.csv` | Table with 143 data rows; columns: ﻿case, patient, visit, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct | 表格含 143 条数据行；列：﻿case, patient, visit, total_voxels, background_voxels, background_share_pct, body_voxels, body_share_pct, bones_voxels, bones_share_pct, bladder_voxels, bladder_share_pct |
| `results/data_analysis/training_distribution/per_volume_intensity_histogram.csv` | Binary or large file; 491,370 bytes | 二进制或大文件；491,370 字节 |
| `results/data_analysis/training_distribution/per_volume_slice_coverage.csv` | Table with 143 data rows; columns: ﻿case, patient, axial_slices, background_present_slices, background_slice_hit_pct, body_present_slices, body_slice_hit_pct, bones_present_slices, bones_slice_hit_pct, bladder_present_slices, bladder_slice_hit_pct, rectum_present_slices | 表格含 143 条数据行；列：﻿case, patient, axial_slices, background_present_slices, background_slice_hit_pct, body_present_slices, body_slice_hit_pct, bones_present_slices, bones_slice_hit_pct, bladder_present_slices, bladder_slice_hit_pct, rectum_present_slices |
| `results/data_analysis/training_distribution/pooled_positive_intensity_histogram.csv` | Table with 120 data rows; columns: ﻿lower_inclusive, upper_exclusive, positive_voxels | 表格含 120 条数据行；列：﻿lower_inclusive, upper_exclusive, positive_voxels |
| `results/data_analysis/training_distribution/pooled_positive_intensity_histogram.svg` | UTF-8 text with 298 lines; 13,306 bytes | UTF-8 文本，298 行；13,306 字节 |
| `results/data_analysis/training_distribution/raw_axial_slice_hit_rate.svg` | UTF-8 text with 262 lines; 12,836 bytes | UTF-8 文本，262 行；12,836 字节 |
| `results/data_analysis/training_distribution/README.md` | Entry point or directory guide; Sections: Training Data Distribution | 项目入口或目录说明；章节标题：Training Data Distribution |
| `results/data_analysis/training_distribution/slice_presence_summary.csv` | Table with 6 data rows; columns: ﻿class, present_slices, total_slices, slice_hit_pct, median_present_slices_per_volume, min_present_slices_per_volume, max_present_slices_per_volume | 表格含 6 条数据行；列：﻿class, present_slices, total_slices, slice_hit_pct, median_present_slices_per_volume, min_present_slices_per_volume, max_present_slices_per_volume |
| `results/data_analysis/training_distribution/spatial_statistics.csv` | Table with 143 data rows; columns: ﻿case, patient, shape_x, shape_y, shape_z, spacing_x_mm, spacing_y_mm, spacing_z_mm, voxel_volume_mm3, orientation, bladder_volume_ml, bladder_bbox_min_x_voxel | 表格含 143 条数据行；列：﻿case, patient, shape_x, shape_y, shape_z, spacing_x_mm, spacing_y_mm, spacing_z_mm, voxel_volume_mm3, orientation, bladder_volume_ml, bladder_bbox_min_x_voxel |
| `results/data_analysis/training_distribution/voxel_spacing.svg` | UTF-8 text with 372 lines; 32,876 bytes | UTF-8 文本，372 行；32,876 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/B040_Week4_head_margins.png` | Binary or large file; 288,330 bytes | 二进制或大文件；288,330 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/B040_Week4_improved_3d_body.png` | Binary or large file; 247,830 bytes | 二进制或大文件；247,830 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/B040_Week4_standard_3d_body.png` | Binary or large file; 204,666 bytes | 二进制或大文件；204,666 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/checksums.json` | JSON fields: B040_Week4_head_margins.png, B040_Week4_improved_3d_body.png, B040_Week4_standard_3d_body.png, J026_Week4_head_margins.png, J026_Week4_improved_3d_body.png, J026_Week4_standard_3d_body.png, K008_Week0_head_margins.png, K008_Week0_improved_3d_body.png, K008_Week0_standard_3d_body.png, K018_Week4_head_margins.png, K018_Week4_improved_3d_body.png, K018_Week4_standard_3d_body.png | JSON 字段：B040_Week4_head_margins.png, B040_Week4_improved_3d_body.png, B040_Week4_standard_3d_body.png, J026_Week4_head_margins.png, J026_Week4_improved_3d_body.png, J026_Week4_standard_3d_body.png, K008_Week0_head_margins.png, K008_Week0_improved_3d_body.png, K008_Week0_standard_3d_body.png, K018_Week4_head_margins.png, K018_Week4_improved_3d_body.png, K018_Week4_standard_3d_body.png |
| `results/diagnostics/inference_context/context_sensitivity_20261020/CONTEXT_SENSITIVITY_REVIEW_ZH.md` | Sections: 推理范围敏感性：固定权重诊断结果; Body 误检与小器官变化; 如何解释 | 章节标题：推理范围敏感性：固定权重诊断结果; Body 误检与小器官变化; 如何解释 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/J026_Week4_head_margins.png` | Binary or large file; 261,434 bytes | 二进制或大文件；261,434 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/J026_Week4_improved_3d_body.png` | Binary or large file; 197,241 bytes | 二进制或大文件；197,241 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/J026_Week4_standard_3d_body.png` | Binary or large file; 175,030 bytes | 二进制或大文件；175,030 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K008_Week0_head_margins.png` | Binary or large file; 299,816 bytes | 二进制或大文件；299,816 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K008_Week0_improved_3d_body.png` | Binary or large file; 213,450 bytes | 二进制或大文件；213,450 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K008_Week0_standard_3d_body.png` | Binary or large file; 200,852 bytes | 二进制或大文件；200,852 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K018_Week4_head_margins.png` | Binary or large file; 304,804 bytes | 二进制或大文件；304,804 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K018_Week4_improved_3d_body.png` | Binary or large file; 256,174 bytes | 二进制或大文件；256,174 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/K018_Week4_standard_3d_body.png` | Binary or large file; 161,339 bytes | 二进制或大文件；161,339 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/README.md` | Entry point or directory guide; Sections: P2 fixed-weight Validation diagnostic | 项目入口或目录说明；章节标题：P2 fixed-weight Validation diagnostic |
| `results/diagnostics/inference_context/context_sensitivity_20261020/six_class_context_comparison.png` | Binary or large file; 105,674 bytes | 二进制或大文件；105,674 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/six_class_context_comparison.svg` | UTF-8 text with 1834 lines; 58,935 bytes | UTF-8 文本，1834 行；58,935 字节 |
| `results/diagnostics/inference_context/context_sensitivity_20261020/six_class_metrics.csv` | Table with 96 data rows; columns: model, case, protocol, class, dice, iou, precision, recall | 表格含 96 条数据行；列：model, case, protocol, class, dice, iou, precision, recall |
| `results/diagnostics/inference_context/prediction_mechanisms_20261020/NUMERICAL_REPLAY_REVIEW_ZH.md` | Sections: 数值回放审查：正控失败与解释边界; 实际观察; 待检验的数值解释; 最小下一项：只做数值定位; 最终数值审计与 P2 关口结论 | 章节标题：数值回放审查：正控失败与解释边界; 实际观察; 待检验的数值解释; 最小下一项：只做数值定位; 最终数值审计与 P2 关口结论 |
| `results/diagnostics/inference_context/prediction_mechanisms_20261020/README.md` | Entry point or directory guide; Sections: 预测机制审查：远处误检的现有证据; A. 推理等价性; B–D. 同体素 logits、几何与辅助头; 根因证据排序; E. 下一项与后处理边界 | 项目入口或目录说明；章节标题：预测机制审查：远处误检的现有证据; A. 推理等价性; B–D. 同体素 logits、几何与辅助头; 根因证据排序; E. 下一项与后处理边界 |
| `results/diagnostics/inference_context/prediction_mechanisms_20261020/same_voxel_logit_evidence.png` | Binary or large file; 135,181 bytes | 二进制或大文件；135,181 字节 |
| `results/diagnostics/inference_context/prediction_mechanisms_20261020/same_voxel_logit_evidence.svg` | UTF-8 text with 2443 lines; 86,786 bytes | UTF-8 文本，2443 行；86,786 字节 |
| `results/diagnostics/inference_context/prediction_mechanisms_20261020/spatial_logit_evidence.csv` | Table with 16 data rows; columns: model, case, protocol, distant_background_to_body, body_hd95_mm, all_body_fp_edge_distance_median_mm, all_body_fp_to_gt_distance_p95_mm, pred_body_component_count_6, pred_body_components_without_gt_overlap, pred_body_voxels_in_components_without_gt_overlap, reference_voxels, margin_p05 | 表格含 16 条数据行；列：model, case, protocol, distant_background_to_body, body_hd95_mm, all_body_fp_edge_distance_median_mm, all_body_fp_to_gt_distance_p95_mm, pred_body_component_count_6, pred_body_components_without_gt_overlap, pred_body_voxels_in_components_without_gt_overlap, reference_voxels, margin_p05 |
| `results/diagnostics/inference_context/README.md` | Entry point or directory guide; Sections: no headings | 项目入口或目录说明；章节标题：no headings |
| `results/documentation_audit/audit_20261020.json` | JSON fields: markdown_count, local_link_scope, broken_local_links, encoding_flags, protected_count, approved_protected_retirements, publication_scope, intentionally_local_links, changed_protected_files, files | JSON 字段：markdown_count, local_link_scope, broken_local_links, encoding_flags, protected_count, approved_protected_retirements, publication_scope, intentionally_local_links, changed_protected_files, files |
| `results/documentation_audit/before_20261020.json` | JSON fields: before, protected | JSON 字段：before, protected |
| `results/documentation_audit/cleanup_validation_20261020.json` | JSON fields: python_ast_files_before_temporary_executor_removal, current_markdown_files, exact_duplicate_markdown_groups, original_suite_tests_passed, post_audit_update_targeted_tests_passed, active_v4_source_hashes_match, active_v4_zip_unchanged, unique_legacy_archive_unchanged, new_audit_package_matches_sources, packaging_excludes_nested_zip_and_refuses_overwrite, full_merged_body_line_counts_preserved, python_cache_directories_removed | JSON 字段：python_ast_files_before_temporary_executor_removal, current_markdown_files, exact_duplicate_markdown_groups, original_suite_tests_passed, post_audit_update_targeted_tests_passed, active_v4_source_hashes_match, active_v4_zip_unchanged, unique_legacy_archive_unchanged, new_audit_package_matches_sources, packaging_excludes_nested_zip_and_refuses_overwrite, full_merged_body_line_counts_preserved, python_cache_directories_removed |
| `results/documentation_audit/consolidation_20261020.json` | JSON fields: date, authorization, merged_sections, retired_files, removed_transfer_members, redirects, active_v4_checksums_preserved, preserved_legacy_archive_sha256, approved_protected_retirements, retired_derived_artifacts, derived_replacement_verification, image_review_merge | JSON 字段：date, authorization, merged_sections, retired_files, removed_transfer_members, redirects, active_v4_checksums_preserved, preserved_legacy_archive_sha256, approved_protected_retirements, retired_derived_artifacts, derived_replacement_verification, image_review_merge |
| `results/documentation_audit/naming_migration_20261020.json` | JSON fields: schema_version, moves, updated_consumers, protected_files_unchanged, active_training_paths_unchanged, statistical_links_switched_to_svg, temporary_executors_removed, statistical_figure_renames | JSON 字段：schema_version, moves, updated_consumers, protected_files_unchanged, active_training_paths_unchanged, statistical_links_switched_to_svg, temporary_executors_removed, statistical_figure_renames |
| `results/documentation_audit/publication_audit_20261020.json` | JSON fields: markdown_count, local_link_scope, broken_local_links, encoding_flags, protected_count, approved_protected_retirements, publication_scope, intentionally_local_links, changed_protected_files, files | JSON 字段：markdown_count, local_link_scope, broken_local_links, encoding_flags, protected_count, approved_protected_retirements, publication_scope, intentionally_local_links, changed_protected_files, files |
| `results/documentation_audit/README.md` | Entry point or directory guide; Sections: 项目去重、合并与完整性审查; 主题命名与发布更新; 当前清理结果 · 2026-10-20; 保留的独立内容; 合并后的完整报告 | 项目入口或目录说明；章节标题：项目去重、合并与完整性审查; 主题命名与发布更新; 当前清理结果 · 2026-10-20; 保留的独立内容; 合并后的完整报告 |
| `results/documentation_audit/relocations_20261020.json` | JSON fields: moves, link_repairs | JSON 字段：moves, link_repairs |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/J026_Week4_body_three_planes.png` | Binary or large file; 449,945 bytes | 二进制或大文件；449,945 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/J026_Week4_six_classes.png` | Binary or large file; 936,832 bytes | 二进制或大文件；936,832 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/K018_Week4_body_three_planes.png` | Binary or large file; 422,307 bytes | 二进制或大文件；422,307 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/K018_Week4_six_classes.png` | Binary or large file; 885,510 bytes | 二进制或大文件；885,510 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/local_verification.json` | JSON fields: record_checks_against_downloaded_remote_provenance, image_checksums, limitation | JSON 字段：record_checks_against_downloaded_remote_provenance, image_checksums, limitation |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/M030_Week0_body_three_planes.png` | Binary or large file; 393,301 bytes | 二进制或大文件；393,301 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/M030_Week0_six_classes.png` | Binary or large file; 1,006,039 bytes | 二进制或大文件；1,006,039 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/provenance.json` | JSON fields: job, script_sha256, checkpoints, cases | JSON 字段：job, script_sha256, checkpoints, cases |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/README.md` | Entry point or directory guide; Sections: 四例Validation图片审阅; 看图方法; 实际观察; S022_Week2：以完整混淆矩阵解释图片未覆盖的退步; v2 Validation对照 | 项目入口或目录说明；章节标题：四例Validation图片审阅; 看图方法; 实际观察; S022_Week2：以完整混淆矩阵解释图片未覆盖的退步; v2 Validation对照 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/S022_Week2_body_three_planes.png` | Binary or large file; 427,693 bytes | 二进制或大文件；427,693 字节 |
| `results/experiments/improved_3d_v2/case_studies/validation/review_45_50/S022_Week2_six_classes.png` | Binary or large file; 974,034 bytes | 二进制或大文件；974,034 字节 |
| `results/experiments/improved_3d_v2/figures/all_six_classes.png` | Binary or large file; 141,258 bytes | 二进制或大文件；141,258 字节 |
| `results/experiments/improved_3d_v2/figures/all_six_classes.svg` | UTF-8 text with 2797 lines; 86,823 bytes | UTF-8 文本，2797 行；86,823 字节 |
| `results/experiments/improved_3d_v2/figures/patient_comparison.png` | Binary or large file; 84,930 bytes | 二进制或大文件；84,930 字节 |
| `results/experiments/improved_3d_v2/figures/patient_comparison.svg` | UTF-8 text with 1374 lines; 42,857 bytes | UTF-8 文本，1374 行；42,857 字节 |
| `results/experiments/improved_3d_v2/figures/README.md` | Entry point or directory guide; Sections: v2分析图 | 项目入口或目录说明；章节标题：v2分析图 |
| `results/experiments/improved_3d_v2/figures/S022_Week2_error_counts.png` | Binary or large file; 76,832 bytes | 二进制或大文件；76,832 字节 |
| `results/experiments/improved_3d_v2/figures/S022_Week2_error_counts.svg` | UTF-8 text with 1588 lines; 49,944 bytes | UTF-8 文本，1588 行；49,944 字节 |
| `results/experiments/improved_3d_v2/figures/training_and_boundary_trajectory.png` | Binary or large file; 185,180 bytes | 二进制或大文件；185,180 字节 |
| `results/experiments/improved_3d_v2/figures/training_and_boundary_trajectory.svg` | UTF-8 text with 2210 lines; 69,951 bytes | UTF-8 文本，2210 行；69,951 字节 |
| `results/experiments/improved_3d_v2/provenance/local_source_checksums.json` | JSON fields: logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json, logs\training\baseline_3d\epoch_008.json, logs\training\baseline_3d\epoch_009.json, logs\training\baseline_3d\epoch_010.json, logs\training\baseline_3d\epoch_011.json, logs\training\baseline_3d\epoch_012.json | JSON 字段：logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json, logs\training\baseline_3d\epoch_008.json, logs\training\baseline_3d\epoch_009.json, logs\training\baseline_3d\epoch_010.json, logs\training\baseline_3d\epoch_011.json, logs\training\baseline_3d\epoch_012.json |
| `results/experiments/improved_3d_v2/README.md` | Entry point or directory guide; Sections: Improved 3D v2 独立实验区; 最新分析; Test边界; 最终归档与清理 | 项目入口或目录说明；章节标题：Improved 3D v2 独立实验区; 最新分析; Test边界; 最终归档与清理 |
| `results/experiments/improved_3d_v2/reports/RESEARCH_REPORT_ZH.md` | Sections: V2研究报告：设计、实验结果与局限; 原文：V2_Literature_Redesign_ZH.md; Improved 3D v2：论文复核、数据不均衡与重新设计; 原文：V2_Validation_Analysis_ZH.md; Improved 3D v2：Validation系统分析 | 章节标题：V2研究报告：设计、实验结果与局限; 原文：V2_Literature_Redesign_ZH.md; Improved 3D v2：论文复核、数据不均衡与重新设计; 原文：V2_Validation_Analysis_ZH.md; Improved 3D v2：Validation系统分析 |
| `results/experiments/improved_3d_v2/tables/analysis_summary.json` | JSON fields: summary, epoch45_to_50_patient_delta, verification | JSON 字段：summary, epoch45_to_50_patient_delta, verification |
| `results/experiments/improved_3d_v2/tables/body_case_comparison.csv` | Table with 152 data rows; columns: ﻿model, case, patient, hd95_mm, far_fp, background_voxels | 表格含 152 条数据行；列：﻿model, case, patient, hd95_mm, far_fp, background_voxels |
| `results/experiments/improved_3d_v2/tables/epoch45_to_50_patient_delta.csv` | Table with 6 data rows; columns: ﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 6 条数据行；列：﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/experiments/improved_3d_v2/tables/patient_comparison.csv` | Table with 24 data rows; columns: ﻿model, epoch, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 24 条数据行；列：﻿model, epoch, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/experiments/improved_3d_v2/tables/S022_Week2_error_counts.csv` | Table with 6 data rows; columns: ﻿epoch, case, target, true_positive, false_positive, false_negative, background_to_target, body_to_target | 表格含 6 条数据行；列：﻿epoch, case, target, true_positive, false_positive, false_negative, background_to_target, body_to_target |
| `results/experiments/improved_3d_v2/tables/six_class_comparison.csv` | Table with 24 data rows; columns: ﻿model, epoch, label, pooled_dice, pooled_iou, pooled_precision, pooled_recall, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall | 表格含 24 条数据行；列：﻿model, epoch, label, pooled_dice, pooled_iou, pooled_precision, pooled_recall, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall |
| `results/experiments/improved_3d_v2/tables/validation_trajectory.csv` | Table with 10 data rows; columns: ﻿epoch, score, body_hd95_median, far_fp | 表格含 10 条数据行；列：﻿epoch, score, body_hd95_median, far_fp |
| `results/experiments/improved_3d_v2/training_records/config_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `results/experiments/improved_3d_v2/training_records/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_005.json` | Binary or large file; 141,022 bytes | 二进制或大文件；141,022 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_010.json` | Binary or large file; 140,803 bytes | 二进制或大文件；140,803 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_015.json` | Binary or large file; 140,483 bytes | 二进制或大文件；140,483 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_020.json` | Binary or large file; 140,535 bytes | 二进制或大文件；140,535 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_025.json` | Binary or large file; 140,871 bytes | 二进制或大文件；140,871 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_030.json` | Binary or large file; 140,249 bytes | 二进制或大文件；140,249 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_035.json` | Binary or large file; 140,084 bytes | 二进制或大文件；140,084 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_040.json` | Binary or large file; 140,292 bytes | 二进制或大文件；140,292 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_045.json` | Binary or large file; 140,101 bytes | 二进制或大文件；140,101 字节 |
| `results/experiments/improved_3d_v2/training_records/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v2/training_records/epoch_050.json` | Binary or large file; 140,261 bytes | 二进制或大文件；140,261 字节 |
| `results/experiments/improved_3d_v2/training_records/manifest.json` | JSON fields: boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 | JSON 字段：boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 |
| `results/experiments/improved_3d_v3/provenance/local_source_checksums.json` | JSON fields: logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json, logs\training\baseline_3d\epoch_008.json, logs\training\baseline_3d\epoch_009.json, logs\training\baseline_3d\epoch_010.json, logs\training\baseline_3d\epoch_011.json, logs\training\baseline_3d\epoch_012.json | JSON 字段：logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json, logs\training\baseline_3d\epoch_008.json, logs\training\baseline_3d\epoch_009.json, logs\training\baseline_3d\epoch_010.json, logs\training\baseline_3d\epoch_011.json, logs\training\baseline_3d\epoch_012.json |
| `results/experiments/improved_3d_v3/README.md` | Entry point or directory guide; Sections: Improved 3D v3独立实验区 | 项目入口或目录说明；章节标题：Improved 3D v3独立实验区 |
| `results/experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md` | Sections: V3系统研究报告：设计、验证、泛化与方法审查; 原文：V3_Research_Plan_ZH.md; Improved 3D v3：依据现有证据的下一项实验规划; 原文：V3_Validation_Analysis_ZH.md; Improved 3D v3：50轮Validation系统分析 | 章节标题：V3系统研究报告：设计、验证、泛化与方法审查; 原文：V3_Research_Plan_ZH.md; Improved 3D v3：依据现有证据的下一项实验规划; 原文：V3_Validation_Analysis_ZH.md; Improved 3D v3：50轮Validation系统分析 |
| `results/experiments/improved_3d_v3/tables/analysis_summary.json` | JSON fields: summary, v3_minus_v1_patient_delta, verification, decision_gates | JSON 字段：summary, v3_minus_v1_patient_delta, verification, decision_gates |
| `results/experiments/improved_3d_v3/tables/body_case_comparison.csv` | Table with 152 data rows; columns: ﻿model, case, patient, hd95_mm, far_fp, background_voxels | 表格含 152 条数据行；列：﻿model, case, patient, hd95_mm, far_fp, background_voxels |
| `results/experiments/improved_3d_v3/tables/foreground_surfaces_v3.csv` | Table with 190 data rows; columns: ﻿case, patient, label, assd_mm, hd95_mm, status | 表格含 190 条数据行；列：﻿case, patient, label, assd_mm, hd95_mm, status |
| `results/experiments/improved_3d_v3/tables/patient_comparison.csv` | Table with 24 data rows; columns: ﻿model, epoch, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 24 条数据行；列：﻿model, epoch, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/experiments/improved_3d_v3/tables/six_class_comparison.csv` | Table with 24 data rows; columns: ﻿model, epoch, label, pooled_dice, pooled_iou, pooled_precision, pooled_recall, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall | 表格含 24 条数据行；列：﻿model, epoch, label, pooled_dice, pooled_iou, pooled_precision, pooled_recall, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall |
| `results/experiments/improved_3d_v3/tables/v3_minus_standard_patient_delta.csv` | Table with 6 data rows; columns: ﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 6 条数据行；列：﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/experiments/improved_3d_v3/tables/v3_minus_v1_patient_delta.csv` | Table with 6 data rows; columns: ﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 6 条数据行；列：﻿patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/experiments/improved_3d_v3/tables/validation_trajectory.csv` | Table with 10 data rows; columns: ﻿epoch, score, body_hd95_median, far_fp | 表格含 10 条数据行；列：﻿epoch, score, body_hd95_median, far_fp |
| `results/experiments/improved_3d_v3/training_records/config_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `results/experiments/improved_3d_v3/training_records/epoch_001.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_002.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_003.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_004.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_005.json` | Binary or large file; 120,922 bytes | 二进制或大文件；120,922 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_006.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_007.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_008.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_009.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_010.json` | Binary or large file; 120,507 bytes | 二进制或大文件；120,507 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_011.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_012.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_013.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_014.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_015.json` | Binary or large file; 120,212 bytes | 二进制或大文件；120,212 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_016.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_017.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_018.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_019.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_020.json` | Binary or large file; 120,215 bytes | 二进制或大文件；120,215 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_021.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_022.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_023.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_024.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_025.json` | Binary or large file; 120,098 bytes | 二进制或大文件；120,098 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_026.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_027.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_028.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_029.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_030.json` | Binary or large file; 119,975 bytes | 二进制或大文件；119,975 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_031.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_032.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_033.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_034.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_035.json` | Binary or large file; 120,112 bytes | 二进制或大文件；120,112 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_036.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_037.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_038.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_039.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_040.json` | Binary or large file; 120,051 bytes | 二进制或大文件；120,051 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_041.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_042.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_043.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_044.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_045.json` | Binary or large file; 119,944 bytes | 二进制或大文件；119,944 字节 |
| `results/experiments/improved_3d_v3/training_records/epoch_046.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_047.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_048.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_049.json` | JSON fields: best_epoch, best_score, epoch, job_id, selected_cases, train, validation | JSON 字段：best_epoch, best_score, epoch, job_id, selected_cases, train, validation |
| `results/experiments/improved_3d_v3/training_records/epoch_050.json` | Binary or large file; 119,973 bytes | 二进制或大文件；119,973 字节 |
| `results/experiments/improved_3d_v3/training_records/manifest.json` | JSON fields: boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 | JSON 字段：boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind, loss_sha256 |
| `results/experiments/improved_3d_v4/a0/training_records/config_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_001.json` | Binary or large file; 251,886 bytes | 二进制或大文件；251,886 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_002.json` | Binary or large file; 251,868 bytes | 二进制或大文件；251,868 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_003.json` | Binary or large file; 251,923 bytes | 二进制或大文件；251,923 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_004.json` | Binary or large file; 252,071 bytes | 二进制或大文件；252,071 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_005.json` | Binary or large file; 320,836 bytes | 二进制或大文件；320,836 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_006.json` | Binary or large file; 252,011 bytes | 二进制或大文件；252,011 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_007.json` | Binary or large file; 251,862 bytes | 二进制或大文件；251,862 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_008.json` | Binary or large file; 252,132 bytes | 二进制或大文件；252,132 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_009.json` | Binary or large file; 252,009 bytes | 二进制或大文件；252,009 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_010.json` | Binary or large file; 320,724 bytes | 二进制或大文件；320,724 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_011.json` | Binary or large file; 252,214 bytes | 二进制或大文件；252,214 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_012.json` | Binary or large file; 252,007 bytes | 二进制或大文件；252,007 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_013.json` | Binary or large file; 252,074 bytes | 二进制或大文件；252,074 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_014.json` | Binary or large file; 251,881 bytes | 二进制或大文件；251,881 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_015.json` | Binary or large file; 320,319 bytes | 二进制或大文件；320,319 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_016.json` | Binary or large file; 251,978 bytes | 二进制或大文件；251,978 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_017.json` | Binary or large file; 252,118 bytes | 二进制或大文件；252,118 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_018.json` | Binary or large file; 251,861 bytes | 二进制或大文件；251,861 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_019.json` | Binary or large file; 251,858 bytes | 二进制或大文件；251,858 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_020.json` | Binary or large file; 320,260 bytes | 二进制或大文件；320,260 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_021.json` | Binary or large file; 251,982 bytes | 二进制或大文件；251,982 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_022.json` | Binary or large file; 252,024 bytes | 二进制或大文件；252,024 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_023.json` | Binary or large file; 252,146 bytes | 二进制或大文件；252,146 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_024.json` | Binary or large file; 252,102 bytes | 二进制或大文件；252,102 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_025.json` | Binary or large file; 319,947 bytes | 二进制或大文件；319,947 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_026.json` | Binary or large file; 252,069 bytes | 二进制或大文件；252,069 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_027.json` | Binary or large file; 252,273 bytes | 二进制或大文件；252,273 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_028.json` | Binary or large file; 252,097 bytes | 二进制或大文件；252,097 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_029.json` | Binary or large file; 252,196 bytes | 二进制或大文件；252,196 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_030.json` | Binary or large file; 320,297 bytes | 二进制或大文件；320,297 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_031.json` | Binary or large file; 252,311 bytes | 二进制或大文件；252,311 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_032.json` | Binary or large file; 252,182 bytes | 二进制或大文件；252,182 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_033.json` | Binary or large file; 252,185 bytes | 二进制或大文件；252,185 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_034.json` | Binary or large file; 252,323 bytes | 二进制或大文件；252,323 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_035.json` | Binary or large file; 320,131 bytes | 二进制或大文件；320,131 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_036.json` | Binary or large file; 252,121 bytes | 二进制或大文件；252,121 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_037.json` | Binary or large file; 252,481 bytes | 二进制或大文件；252,481 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_038.json` | Binary or large file; 252,297 bytes | 二进制或大文件；252,297 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_039.json` | Binary or large file; 252,160 bytes | 二进制或大文件；252,160 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_040.json` | Binary or large file; 320,341 bytes | 二进制或大文件；320,341 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_041.json` | Binary or large file; 252,029 bytes | 二进制或大文件；252,029 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_042.json` | Binary or large file; 252,100 bytes | 二进制或大文件；252,100 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_043.json` | Binary or large file; 252,009 bytes | 二进制或大文件；252,009 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_044.json` | Binary or large file; 252,119 bytes | 二进制或大文件；252,119 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_045.json` | Binary or large file; 319,920 bytes | 二进制或大文件；319,920 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_046.json` | Binary or large file; 252,386 bytes | 二进制或大文件；252,386 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_047.json` | Binary or large file; 252,223 bytes | 二进制或大文件；252,223 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_048.json` | Binary or large file; 252,300 bytes | 二进制或大文件；252,300 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_049.json` | Binary or large file; 252,157 bytes | 二进制或大文件；252,157 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/epoch_050.json` | Binary or large file; 319,956 bytes | 二进制或大文件；319,956 字节 |
| `results/experiments/improved_3d_v4/a0/training_records/manifest.json` | JSON fields: augmentation_sha256, boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind | JSON 字段：augmentation_sha256, boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind |
| `results/experiments/improved_3d_v4/a1/training_records/config_frozen.json` | JSON fields: status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling | JSON 字段：status, purpose, split_sha256, train_cases, train_patients, validation_cases, validation_patients, models, model_defaults, seed, preprocessing, sampling |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_001.json` | Binary or large file; 256,444 bytes | 二进制或大文件；256,444 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_002.json` | Binary or large file; 256,445 bytes | 二进制或大文件；256,445 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_003.json` | Binary or large file; 256,674 bytes | 二进制或大文件；256,674 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_004.json` | Binary or large file; 256,651 bytes | 二进制或大文件；256,651 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_005.json` | Binary or large file; 325,373 bytes | 二进制或大文件；325,373 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_006.json` | Binary or large file; 257,104 bytes | 二进制或大文件；257,104 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_007.json` | Binary or large file; 257,302 bytes | 二进制或大文件；257,302 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_008.json` | Binary or large file; 257,561 bytes | 二进制或大文件；257,561 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_009.json` | Binary or large file; 257,060 bytes | 二进制或大文件；257,060 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_010.json` | Binary or large file; 326,278 bytes | 二进制或大文件；326,278 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_011.json` | Binary or large file; 256,958 bytes | 二进制或大文件；256,958 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_012.json` | Binary or large file; 257,405 bytes | 二进制或大文件；257,405 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_013.json` | Binary or large file; 257,466 bytes | 二进制或大文件；257,466 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_014.json` | Binary or large file; 256,929 bytes | 二进制或大文件；256,929 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_015.json` | Binary or large file; 325,225 bytes | 二进制或大文件；325,225 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_016.json` | Binary or large file; 256,960 bytes | 二进制或大文件；256,960 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_017.json` | Binary or large file; 256,635 bytes | 二进制或大文件；256,635 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_018.json` | Binary or large file; 256,667 bytes | 二进制或大文件；256,667 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_019.json` | Binary or large file; 256,900 bytes | 二进制或大文件；256,900 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_020.json` | Binary or large file; 324,921 bytes | 二进制或大文件；324,921 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_021.json` | Binary or large file; 257,975 bytes | 二进制或大文件；257,975 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_022.json` | Binary or large file; 257,226 bytes | 二进制或大文件；257,226 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_023.json` | Binary or large file; 257,148 bytes | 二进制或大文件；257,148 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_024.json` | Binary or large file; 256,458 bytes | 二进制或大文件；256,458 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_025.json` | Binary or large file; 325,068 bytes | 二进制或大文件；325,068 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_026.json` | Binary or large file; 257,689 bytes | 二进制或大文件；257,689 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_027.json` | Binary or large file; 257,344 bytes | 二进制或大文件；257,344 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_028.json` | Binary or large file; 257,279 bytes | 二进制或大文件；257,279 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_029.json` | Binary or large file; 257,875 bytes | 二进制或大文件；257,875 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_030.json` | Binary or large file; 325,005 bytes | 二进制或大文件；325,005 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_031.json` | Binary or large file; 258,156 bytes | 二进制或大文件；258,156 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_032.json` | Binary or large file; 256,616 bytes | 二进制或大文件；256,616 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_033.json` | Binary or large file; 256,941 bytes | 二进制或大文件；256,941 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_034.json` | Binary or large file; 257,207 bytes | 二进制或大文件；257,207 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_035.json` | Binary or large file; 325,126 bytes | 二进制或大文件；325,126 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_036.json` | Binary or large file; 257,410 bytes | 二进制或大文件；257,410 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_037.json` | Binary or large file; 258,078 bytes | 二进制或大文件；258,078 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_038.json` | Binary or large file; 257,680 bytes | 二进制或大文件；257,680 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_039.json` | Binary or large file; 257,172 bytes | 二进制或大文件；257,172 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_040.json` | Binary or large file; 324,949 bytes | 二进制或大文件；324,949 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_041.json` | Binary or large file; 256,446 bytes | 二进制或大文件；256,446 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_042.json` | Binary or large file; 257,486 bytes | 二进制或大文件；257,486 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_043.json` | Binary or large file; 257,085 bytes | 二进制或大文件；257,085 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_044.json` | Binary or large file; 258,174 bytes | 二进制或大文件；258,174 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_045.json` | Binary or large file; 325,467 bytes | 二进制或大文件；325,467 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_046.json` | Binary or large file; 257,776 bytes | 二进制或大文件；257,776 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_047.json` | Binary or large file; 257,103 bytes | 二进制或大文件；257,103 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_048.json` | Binary or large file; 257,249 bytes | 二进制或大文件；257,249 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_049.json` | Binary or large file; 257,537 bytes | 二进制或大文件；257,537 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/epoch_050.json` | Binary or large file; 325,217 bytes | 二进制或大文件；325,217 字节 |
| `results/experiments/improved_3d_v4/a1/training_records/manifest.json` | JSON fields: augmentation_sha256, boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind | JSON 字段：augmentation_sha256, boundaries_sha256, config_sha256, data_sha256, dataset_root, g1_sha256, gpu, inference_sha256, initial_state_sha256, initialization_evidence, job_id_initial, kind |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/body_boundary_distribution.png` | Binary or large file; 63,151 bytes | 二进制或大文件；63,151 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/body_boundary_distribution.svg` | UTF-8 text with 1863 lines; 56,274 bytes | UTF-8 文本，1863 行；56,274 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_base_apex.png` | Binary or large file; 194,073 bytes | 二进制或大文件；194,073 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_base_apex.svg` | UTF-8 text with 1699 lines; 57,641 bytes | UTF-8 文本，1699 行；57,641 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_boundary_tails.png` | Binary or large file; 76,023 bytes | 二进制或大文件；76,023 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_boundary_tails.svg` | UTF-8 text with 1905 lines; 60,294 bytes | UTF-8 文本，1905 行；60,294 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_oar_tradeoff.png` | Binary or large file; 129,656 bytes | 二进制或大文件；129,656 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_oar_tradeoff.svg` | UTF-8 text with 2179 lines; 75,879 bytes | UTF-8 文本，2179 行；75,879 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/resource_profile.png` | Binary or large file; 65,807 bytes | 二进制或大文件；65,807 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/resource_profile.svg` | UTF-8 text with 1859 lines; 56,251 bytes | UTF-8 文本，1859 行；56,251 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/six_class_validation_test.png` | Binary or large file; 404,073 bytes | 二进制或大文件；404,073 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/six_class_validation_test.svg` | Binary or large file; 180,229 bytes | 二进制或大文件；180,229 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/validation_test_generalization.png` | Binary or large file; 63,476 bytes | 二进制或大文件；63,476 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/validation_test_generalization.svg` | UTF-8 text with 1635 lines; 47,405 bytes | UTF-8 文本，1635 行；47,405 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/boundary_errors.csv` | Table with 14 data rows; columns: model, split, far_background_to_body_voxels, far_body_fp_per_100000_background_voxels, prostate_predictions_in_true_bladder, prostate_predictions_in_true_rectum, prostate_in_OAR_per_100000_predicted_prostate, missed_true_prostate_fraction, mean_missed_GT_positive_slices_per_volume, mean_predicted_slices_outside_GT_extent | 表格含 14 条数据行；列：model, split, far_background_to_body_voxels, far_body_fp_per_100000_background_voxels, prostate_predictions_in_true_bladder, prostate_predictions_in_true_rectum, prostate_in_OAR_per_100000_predicted_prostate, missed_true_prostate_fraction, mean_missed_GT_positive_slices_per_volume, mean_predicted_slices_outside_GT_extent |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/model_summary.csv` | Table with 14 data rows; columns: model, split, epoch, small_organ_dice, prostate_dice, foreground_macro_dice, six_class_macro_dice | 表格含 14 条数据行；列：model, split, epoch, small_organ_dice, prostate_dice, foreground_macro_dice, six_class_macro_dice |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/per_case_inference.csv` | Table with 476 data rows; columns: model, split, case, seconds_single_pass, peak_allocated_GiB | 表格含 476 条数据行；列：model, split, case, seconds_single_pass, peak_allocated_GiB |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/prostate_regions.csv` | Table with 42 data rows; columns: model, split, region, patient_equal_slice_mean_dice, available_volumes | 表格含 42 条数据行；列：model, split, region, patient_equal_slice_mean_dice, available_volumes |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/resources.csv` | Table with 7 data rows; columns: model, epoch, gpu, parameters, reference_median_seconds, reference_peak_allocated_GiB, reference_peak_reserved_GiB, test_median_seconds, test_p95_seconds, test_max_peak_allocated_GiB | 表格含 7 条数据行；列：model, epoch, gpu, parameters, reference_median_seconds, reference_peak_allocated_GiB, reference_peak_reserved_GiB, test_median_seconds, test_p95_seconds, test_max_peak_allocated_GiB |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/six_class_metrics.csv` | Table with 84 data rows; columns: model, split, class, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall, voxel_pooled_dice, voxel_pooled_iou, voxel_pooled_precision, voxel_pooled_recall | 表格含 84 条数据行；列：model, split, class, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall, voxel_pooled_dice, voxel_pooled_iou, voxel_pooled_precision, voxel_pooled_recall |
| `results/experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/surface_summary.csv` | Table with 70 data rows; columns: model, split, class, volume_median_hd95_mm, volume_p95_hd95_mm, volume_max_hd95_mm, patient_equal_mean_hd95_mm, patient_equal_mean_assd_mm, both_present, one_empty, both_empty | 表格含 70 条数据行；列：model, split, class, volume_median_hd95_mm, volume_p95_hd95_mm, volume_max_hd95_mm, patient_equal_mean_hd95_mm, patient_equal_mean_assd_mm, both_present, one_empty, both_empty |
| `results/experiments/improved_3d_v4/assessment_analysis/coverage.json` | JSON fields: verified, surface_status, prostate_region_observations, models, test_is_fresh_blind, one_seed_per_recipe, anatomical_autopsy_requires_manual_image_review, clinical_dose_or_safety_evidence | JSON 字段：verified, surface_status, prostate_region_observations, models, test_is_fresh_blind, one_seed_per_recipe, anatomical_autopsy_requires_manual_image_review, clinical_dose_or_safety_evidence |
| `results/experiments/improved_3d_v4/assessment_analysis/integrity_verification.json` | JSON fields: job, complete, verified_files, metrics_sha256, script_sha256, lock_sha256, models, model_volume_comparisons, validation_matrices_reproduced_locally, historical_test_matrices_reproduced_locally, exact_cross_split_file_duplicates, test_status | JSON 字段：job, complete, verified_files, metrics_sha256, script_sha256, lock_sha256, models, model_volume_comparisons, validation_matrices_reproduced_locally, historical_test_matrices_reproduced_locally, exact_cross_split_file_duplicates, test_status |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/distant_body_false_positives.png` | Binary or large file; 88,100 bytes | 二进制或大文件；88,100 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/distant_body_false_positives.svg` | UTF-8 text with 2083 lines; 63,559 bytes | UTF-8 文本，2083 行；63,559 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/patient_variability.png` | Binary or large file; 218,478 bytes | 二进制或大文件；218,478 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/patient_variability.svg` | UTF-8 text with 1846 lines; 62,201 bytes | UTF-8 文本，1846 行；62,201 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/prostate_continuity.png` | Binary or large file; 127,314 bytes | 二进制或大文件；127,314 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/figures/prostate_continuity.svg` | UTF-8 text with 2354 lines; 76,941 bytes | UTF-8 文本，2354 行；76,941 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/continuity.csv` | Binary or large file; 198,343 bytes | 二进制或大文件；198,343 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/paired_patient_sensitivity.csv` | Table with 18 data rows; columns: split, model, reference, mean_difference, patients_improved, patients_worse, leave_one_patient_out_min_difference, leave_one_patient_out_max_difference, difference_without_M015_descriptive_only | 表格含 18 条数据行；列：split, model, reference, mean_difference, patients_improved, patients_worse, leave_one_patient_out_min_difference, leave_one_patient_out_max_difference, difference_without_M015_descriptive_only |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/per_case_class_metrics.csv` | Binary or large file; 353,246 bytes | 二进制或大文件；353,246 字节 |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/per_case_regions.csv` | Table with 1428 data rows; columns: model, split, case, patient, region, dice | 表格含 1428 条数据行；列：model, split, case, patient, region, dice |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/per_patient.csv` | Table with 84 data rows; columns: model, split, patient, Background_dice, Body_dice, Bones_dice, Bladder_dice, Rectum_dice, Prostate_dice, small_organ_dice | 表格含 84 条数据行；列：model, split, patient, Background_dice, Body_dice, Bones_dice, Bladder_dice, Rectum_dice, Prostate_dice, small_organ_dice |
| `results/experiments/improved_3d_v4/assessment_analysis/supplementary_research/tables/surface_distances.csv` | Binary or large file; 198,845 bytes | 二进制或大文件；198,845 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/audit.json` | Binary or large file; 135,066 bytes | 二进制或大文件；135,066 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/checksums.json` | JSON fields: preview_06_R039_Week2.png, preview_02_O025_Week6.png, README.md, audit.json, preview_03_O025_Week6.png, preview_05_R039_Week2.png, preview_04_O025_Week6.png, preview_01_O025_Week6.png | JSON 字段：preview_06_R039_Week2.png, preview_02_O025_Week6.png, README.md, audit.json, preview_03_O025_Week6.png, preview_05_R039_Week2.png, preview_04_O025_Week6.png, preview_01_O025_Week6.png |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_01_O025_Week6.png` | Binary or large file; 47,441 bytes | 二进制或大文件；47,441 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_02_O025_Week6.png` | Binary or large file; 47,001 bytes | 二进制或大文件；47,001 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_03_O025_Week6.png` | Binary or large file; 51,977 bytes | 二进制或大文件；51,977 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_04_O025_Week6.png` | Binary or large file; 52,262 bytes | 二进制或大文件；52,262 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_05_R039_Week2.png` | Binary or large file; 50,437 bytes | 二进制或大文件；50,437 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/preview_06_R039_Week2.png` | Binary or large file; 49,937 bytes | 二进制或大文件；49,937 字节 |
| `results/experiments/improved_3d_v4/augmentation_audit_20261020/README.md` | Entry point or directory guide; Sections: V4 Train配对增强预览 | 项目入口或目录说明；章节标题：V4 Train配对增强预览 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/checksums.json` | JSON fields: input_provenance.json, README.md, metrics_partial.json, metrics.json, images/test/W029_Week6_Prostate.png, images/test/W029_Week6_Prostate_base_apex.png, images/test/M015_Week7_Bladder.png, images/test/M020_Week1_Prostate_base_apex.png, images/test/README.md, images/test/W029_Week6_Bones.png, images/test/M020_Week1_Prostate.png, images/test/W029_Week6_Body.png | JSON 字段：input_provenance.json, README.md, metrics_partial.json, metrics.json, images/test/W029_Week6_Prostate.png, images/test/W029_Week6_Prostate_base_apex.png, images/test/M015_Week7_Bladder.png, images/test/M020_Week1_Prostate_base_apex.png, images/test/README.md, images/test/W029_Week6_Bones.png, images/test/M020_Week1_Prostate.png, images/test/W029_Week6_Body.png |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/M015_Week7_Bladder.png` | Binary or large file; 319,382 bytes | 二进制或大文件；319,382 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/M020_Week1_Prostate.png` | Binary or large file; 240,944 bytes | 二进制或大文件；240,944 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/M020_Week1_Prostate_base_apex.png` | Binary or large file; 332,188 bytes | 二进制或大文件；332,188 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/README.md` | Entry point or directory guide; Sections: 固定best权重诊断图 | 项目入口或目录说明；章节标题：固定best权重诊断图 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/W029_Week6_Body.png` | Binary or large file; 522,140 bytes | 二进制或大文件；522,140 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/W029_Week6_Bones.png` | Binary or large file; 440,480 bytes | 二进制或大文件；440,480 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/W029_Week6_Prostate.png` | Binary or large file; 249,345 bytes | 二进制或大文件；249,345 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/test/W029_Week6_Prostate_base_apex.png` | Binary or large file; 344,564 bytes | 二进制或大文件；344,564 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Background.png` | Binary or large file; 413,867 bytes | 二进制或大文件；413,867 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Bladder.png` | Binary or large file; 259,937 bytes | 二进制或大文件；259,937 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Body.png` | Binary or large file; 505,366 bytes | 二进制或大文件；505,366 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Body_three_planes.png` | Binary or large file; 857,633 bytes | 二进制或大文件；857,633 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Bones.png` | Binary or large file; 431,797 bytes | 二进制或大文件；431,797 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Prostate.png` | Binary or large file; 233,093 bytes | 二进制或大文件；233,093 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Prostate_base_apex.png` | Binary or large file; 331,204 bytes | 二进制或大文件；331,204 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/B040_Week1_Rectum.png` | Binary or large file; 237,351 bytes | 二进制或大文件；237,351 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/J026_Week4_Body_three_planes.png` | Binary or large file; 756,421 bytes | 二进制或大文件；756,421 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/J026_Week4_Rectum.png` | Binary or large file; 236,291 bytes | 二进制或大文件；236,291 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K008_Week0_Bladder.png` | Binary or large file; 253,829 bytes | 二进制或大文件；253,829 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K008_Week0_Body_three_planes.png` | Binary or large file; 761,859 bytes | 二进制或大文件；761,859 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K008_Week0_Prostate.png` | Binary or large file; 241,403 bytes | 二进制或大文件；241,403 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K008_Week0_Prostate_base_apex.png` | Binary or large file; 345,403 bytes | 二进制或大文件；345,403 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K018_Week4_Background.png` | Binary or large file; 513,035 bytes | 二进制或大文件；513,035 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K018_Week4_Body.png` | Binary or large file; 507,334 bytes | 二进制或大文件；507,334 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/K018_Week4_Body_three_planes.png` | Binary or large file; 833,347 bytes | 二进制或大文件；833,347 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/M030_Week4_Body.png` | Binary or large file; 595,199 bytes | 二进制或大文件；595,199 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/M030_Week4_Body_three_planes.png` | Binary or large file; 791,274 bytes | 二进制或大文件；791,274 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/M030_Week4_Bones.png` | Binary or large file; 394,878 bytes | 二进制或大文件；394,878 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/README.md` | Entry point or directory guide; Sections: 固定best权重诊断图 | 项目入口或目录说明；章节标题：固定best权重诊断图 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/S022_Week2_Bladder.png` | Binary or large file; 282,046 bytes | 二进制或大文件；282,046 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/S022_Week2_Body_three_planes.png` | Binary or large file; 701,295 bytes | 二进制或大文件；701,295 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/S022_Week2_Prostate.png` | Binary or large file; 254,753 bytes | 二进制或大文件；254,753 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/images/validation/S022_Week2_Prostate_base_apex.png` | Binary or large file; 359,628 bytes | 二进制或大文件；359,628 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/input_provenance.json` | JSON fields: hashes, exact_cross_split_mri_duplicates, note | JSON 字段：hashes, exact_cross_split_mri_duplicates, note |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics.json` | Binary or large file; 3,611,934 bytes | 二进制或大文件；3,611,934 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics_partial.json` | Binary or large file; 3,611,914 bytes | 二进制或大文件；3,611,914 字节 |
| `results/experiments/improved_3d_v4/comprehensive_audit_20261021/README.md` | Entry point or directory guide; Sections: 最佳checkpoint泛化审计 | 项目入口或目录说明；章节标题：最佳checkpoint泛化审计 |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/selection.csv` | Table with 7 data rows; columns: model, epoch, patient_equal_small_organ_dice | 表格含 7 条数据行；列：model, epoch, patient_equal_small_organ_dice |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/seven_model_selection.png` | Binary or large file; 65,772 bytes | 二进制或大文件；65,772 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/seven_model_selection.svg` | UTF-8 text with 1410 lines; 41,443 bytes | UTF-8 文本，1410 行；41,443 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/seven_model_six_class_dice.png` | Binary or large file; 140,928 bytes | 二进制或大文件；140,928 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/seven_model_six_class_dice.svg` | UTF-8 text with 1900 lines; 71,380 bytes | UTF-8 文本，1900 行；71,380 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/six_class_validation.csv` | Table with 42 data rows; columns: model, epoch, class, dice, iou, precision, recall | 表格含 42 条数据行；列：model, epoch, class, dice, iou, precision, recall |
| `results/experiments/improved_3d_v4/paired_analysis/all_models/source_checksums.json` | JSON fields: logs\training\baseline_2d\epoch_001.json, logs\training\baseline_2d\epoch_002.json, logs\training\baseline_2d\epoch_003.json, logs\training\baseline_2d\epoch_004.json, logs\training\baseline_2d\epoch_005.json, logs\training\baseline_2d\epoch_006.json, logs\training\baseline_2d\epoch_007.json, logs\training\baseline_2d\epoch_008.json, logs\training\baseline_2d\epoch_009.json, logs\training\baseline_2d\epoch_010.json, logs\training\baseline_2d\epoch_011.json, logs\training\baseline_2d\epoch_012.json | JSON 字段：logs\training\baseline_2d\epoch_001.json, logs\training\baseline_2d\epoch_002.json, logs\training\baseline_2d\epoch_003.json, logs\training\baseline_2d\epoch_004.json, logs\training\baseline_2d\epoch_005.json, logs\training\baseline_2d\epoch_006.json, logs\training\baseline_2d\epoch_007.json, logs\training\baseline_2d\epoch_008.json, logs\training\baseline_2d\epoch_009.json, logs\training\baseline_2d\epoch_010.json, logs\training\baseline_2d\epoch_011.json, logs\training\baseline_2d\epoch_012.json |
| `results/experiments/improved_3d_v4/paired_analysis/learning_curves.csv` | Table with 100 data rows; columns: arm, epoch, train_loss, validation_score | 表格含 100 条数据行；列：arm, epoch, train_loss, validation_score |
| `results/experiments/improved_3d_v4/paired_analysis/learning_curves.png` | Binary or large file; 116,954 bytes | 二进制或大文件；116,954 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/learning_curves.svg` | UTF-8 text with 1878 lines; 56,781 bytes | UTF-8 文本，1878 行；56,781 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/patient_metrics.csv` | Table with 12 data rows; columns: arm, patient, Background_dice, Body_dice, Bones_dice, Bladder_dice, Rectum_dice, Prostate_dice, Background_iou, Body_iou, Bones_iou, Bladder_iou | 表格含 12 条数据行；列：arm, patient, Background_dice, Body_dice, Bones_dice, Bladder_dice, Rectum_dice, Prostate_dice, Background_iou, Body_iou, Bones_iou, Bladder_iou |
| `results/experiments/improved_3d_v4/paired_analysis/README.md` | Entry point or directory guide; Sections: V4 配对实验分析; 完成记录与选模; 解释与限制; 已生成与下一步; 七模型统一复核与课程覆盖（2026-10-21） | 项目入口或目录说明；章节标题：V4 配对实验分析; 完成记录与选模; 解释与限制; 已生成与下一步; 七模型统一复核与课程覆盖（2026-10-21） |
| `results/experiments/improved_3d_v4/paired_analysis/record_checksums.json` | JSON fields: results\experiments\improved_3d_v4\a0\training_records\config_frozen.json, results\experiments\improved_3d_v4\a0\training_records\epoch_001.json, results\experiments\improved_3d_v4\a0\training_records\epoch_002.json, results\experiments\improved_3d_v4\a0\training_records\epoch_003.json, results\experiments\improved_3d_v4\a0\training_records\epoch_004.json, results\experiments\improved_3d_v4\a0\training_records\epoch_005.json, results\experiments\improved_3d_v4\a0\training_records\epoch_006.json, results\experiments\improved_3d_v4\a0\training_records\epoch_007.json, results\experiments\improved_3d_v4\a0\training_records\epoch_008.json, results\experiments\improved_3d_v4\a0\training_records\epoch_009.json, results\experiments\improved_3d_v4\a0\training_records\epoch_010.json, results\experiments\improved_3d_v4\a0\training_records\epoch_011.json | JSON 字段：results\experiments\improved_3d_v4\a0\training_records\config_frozen.json, results\experiments\improved_3d_v4\a0\training_records\epoch_001.json, results\experiments\improved_3d_v4\a0\training_records\epoch_002.json, results\experiments\improved_3d_v4\a0\training_records\epoch_003.json, results\experiments\improved_3d_v4\a0\training_records\epoch_004.json, results\experiments\improved_3d_v4\a0\training_records\epoch_005.json, results\experiments\improved_3d_v4\a0\training_records\epoch_006.json, results\experiments\improved_3d_v4\a0\training_records\epoch_007.json, results\experiments\improved_3d_v4\a0\training_records\epoch_008.json, results\experiments\improved_3d_v4\a0\training_records\epoch_009.json, results\experiments\improved_3d_v4\a0\training_records\epoch_010.json, results\experiments\improved_3d_v4\a0\training_records\epoch_011.json |
| `results/experiments/improved_3d_v4/paired_analysis/six_class_dice.png` | Binary or large file; 48,660 bytes | 二进制或大文件；48,660 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/six_class_dice.svg` | UTF-8 text with 1321 lines; 36,309 bytes | UTF-8 文本，1321 行；36,309 字节 |
| `results/experiments/improved_3d_v4/paired_analysis/six_class_metrics.csv` | Table with 12 data rows; columns: arm, class, dice, iou, precision, recall, volume_median_hd95_mm | 表格含 12 条数据行；列：arm, class, dice, iou, precision, recall, volume_median_hd95_mm |
| `results/experiments/improved_3d_v4/paired_analysis/summary.json` | JSON fields: a0, a1 | JSON 字段：a0, a1 |
| `results/experiments/improved_3d_v4/README.md` | Entry point or directory guide; Sections: V4 A0/A1配对训练; 执行身份与最近状态; 本区内容 | 项目入口或目录说明；章节标题：V4 A0/A1配对训练; 执行身份与最近状态; 本区内容 |
| `results/experiments/improved_3d_v4/reports/OPTIMIZATION_REVIEW_ZH.md` | Sections: 常用优化方法补充审查与 v4 训练方案建议; 1. 当前真正遗漏了什么; 2. 常用方法清单：收益、风险与优先级; 3. 三篇参考论文给出的直接提醒; 4. 修订 v4：先优化训练配方，再考虑加网络模块 | 章节标题：常用优化方法补充审查与 v4 训练方案建议; 1. 当前真正遗漏了什么; 2. 常用方法清单：收益、风险与优先级; 3. 三篇参考论文给出的直接提醒; 4. 修订 v4：先优化训练配方，再考虑加网络模块 |
| `results/experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md` | Sections: 七模型最佳权重统一评估与 v4 配对实验结论; 1. 结论与模型选择; 2. 数据来源、完整性与评估规则; 3. 七模型总体表现; 4. A1改善什么，又退步什么 | 章节标题：七模型最佳权重统一评估与 v4 配对实验结论; 1. 结论与模型选择; 2. 数据来源、完整性与评估规则; 3. 七模型总体表现; 4. A1改善什么，又退步什么 |
| `results/figures/baseline_2d/g5_2d_validation_all_classes.png` | Binary or large file; 58,445 bytes | 二进制或大文件；58,445 字节 |
| `results/figures/baseline_2d/g5_2d_validation_all_classes.svg` | UTF-8 text with 1331 lines; 38,132 bytes | UTF-8 文本，1331 行；38,132 字节 |
| `results/figures/baseline_2d/README.md` | Entry point or directory guide; Sections: Supplementary 2D Validation figure | 项目入口或目录说明；章节标题：Supplementary 2D Validation figure |
| `results/figures/comparisons/README.md` | Entry point or directory guide; Sections: 模型对比图索引 | 项目入口或目录说明；章节标题：模型对比图索引 |
| `results/figures/comparisons/test/g6_test_all_classes.png` | Binary or large file; 70,228 bytes | 二进制或大文件；70,228 字节 |
| `results/figures/comparisons/test/g6_test_all_classes.svg` | UTF-8 text with 1505 lines; 44,031 bytes | UTF-8 文本，1505 行；44,031 字节 |
| `results/figures/comparisons/test/g6_test_failure_patterns.png` | Binary or large file; 128,918 bytes | 二进制或大文件；128,918 字节 |
| `results/figures/comparisons/test/g6_test_failure_patterns.svg` | UTF-8 text with 2063 lines; 69,179 bytes | UTF-8 文本，2063 行；69,179 字节 |
| `results/figures/comparisons/test/g6_test_hd95_all_foreground.png` | Binary or large file; 64,275 bytes | 二进制或大文件；64,275 字节 |
| `results/figures/comparisons/test/g6_test_hd95_all_foreground.svg` | UTF-8 text with 1507 lines; 42,799 bytes | UTF-8 文本，1507 行；42,799 字节 |
| `results/figures/comparisons/test/README.md` | Entry point or directory guide; Sections: 固定 Test 结果 | 项目入口或目录说明；章节标题：固定 Test 结果 |
| `results/figures/comparisons/training/g5_three_arm_training_trajectories.png` | Binary or large file; 182,705 bytes | 二进制或大文件；182,705 字节 |
| `results/figures/comparisons/training/g5_three_arm_training_trajectories.svg` | UTF-8 text with 1875 lines; 60,281 bytes | UTF-8 文本，1875 行；60,281 字节 |
| `results/figures/comparisons/training/README.md` | Entry point or directory guide; Sections: 三模型训练过程 | 项目入口或目录说明；章节标题：三模型训练过程 |
| `results/figures/comparisons/validation/g5_3d_validation_all_classes.png` | Binary or large file; 68,380 bytes | 二进制或大文件；68,380 字节 |
| `results/figures/comparisons/validation/g5_3d_validation_all_classes.svg` | UTF-8 text with 1503 lines; 44,330 bytes | UTF-8 文本，1503 行；44,330 字节 |
| `results/figures/comparisons/validation/g5_three_arm_validation_all_classes.png` | Binary or large file; 56,788 bytes | 二进制或大文件；56,788 字节 |
| `results/figures/comparisons/validation/g5_three_arm_validation_all_classes.svg` | UTF-8 text with 1415 lines; 40,455 bytes | UTF-8 文本，1415 行；40,455 字节 |
| `results/figures/comparisons/validation/g5_validation_body_boundaries.png` | Binary or large file; 100,677 bytes | 二进制或大文件；100,677 字节 |
| `results/figures/comparisons/validation/g5_validation_body_boundaries.svg` | UTF-8 text with 1975 lines; 61,632 bytes | UTF-8 文本，1975 行；61,632 字节 |
| `results/figures/comparisons/validation/g5_validation_patient_class_dice.png` | Binary or large file; 172,263 bytes | 二进制或大文件；172,263 字节 |
| `results/figures/comparisons/validation/g5_validation_patient_class_dice.svg` | UTF-8 text with 2229 lines; 83,604 bytes | UTF-8 文本，2229 行；83,604 字节 |
| `results/figures/comparisons/validation/README.md` | Entry point or directory guide; Sections: Validation 六类指标与边界诊断 | 项目入口或目录说明；章节标题：Validation 六类指标与边界诊断 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/body_boundary_by_case.png` | Binary or large file; 234,566 bytes | 二进制或大文件；234,566 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/body_boundary_by_case.svg` | UTF-8 text with 2076 lines; 73,868 bytes | UTF-8 文本，2076 行；73,868 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/learning_curves_and_lr.png` | Binary or large file; 276,344 bytes | 二进制或大文件；276,344 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/learning_curves_and_lr.svg` | Binary or large file; 105,335 bytes | 二进制或大文件；105,335 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/six_class_dice_iou.png` | Binary or large file; 74,775 bytes | 二进制或大文件；74,775 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/figures/six_class_dice_iou.svg` | UTF-8 text with 1934 lines; 55,510 bytes | UTF-8 文本，1934 行；55,510 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/README.md` | Entry point or directory guide; Sections: 课程要求相关指标与图表 | 项目入口或目录说明；章节标题：课程要求相关指标与图表 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/source_review/assessment_page_14.png` | Binary or large file; 169,921 bytes | 二进制或大文件；169,921 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/source_review/assessment_page_5.png` | Binary or large file; 185,940 bytes | 二进制或大文件；185,940 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/source_review/assessment_page_6.png` | Binary or large file; 168,163 bytes | 二进制或大文件；168,163 字节 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/tables/body_boundary_available.csv` | Table with 190 data rows; columns: ﻿model, case, patient, body_hd95_mm, far_fp, availability | 表格含 190 条数据行；列：﻿model, case, patient, body_hd95_mm, far_fp, availability |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/tables/resources_measured_and_pending.csv` | Table with 5 data rows; columns: ﻿model, parameters, measured_train_peak_allocated_GiB, summed_epoch_train_and_checkpoint_seconds, matched_inference_peak_vram, matched_inference_latency, warning | 表格含 5 条数据行；列：﻿model, parameters, measured_train_peak_allocated_GiB, summed_epoch_train_and_checkpoint_seconds, matched_inference_peak_vram, matched_inference_latency, warning |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/tables/selected_checkpoints.json` | JSON fields: 2D, Standard, v1, v2, v3 | JSON 字段：2D, Standard, v1, v2, v3 |
| `results/model_evaluation/best_checkpoint_comparison/assessment_required/tables/six_class_best_metrics.csv` | Table with 30 data rows; columns: ﻿model, selected_epoch, label, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall, pooled_dice, pooled_iou, pooled_precision, pooled_recall | 表格含 30 条数据行；列：﻿model, selected_epoch, label, patient_equal_dice, patient_equal_iou, patient_equal_precision, patient_equal_recall, pooled_dice, pooled_iou, pooled_precision, pooled_recall |
| `results/model_evaluation/best_checkpoint_comparison/README.md` | Entry point or directory guide; Sections: 最佳模型分析入口; 统一选模; 分区 | 项目入口或目录说明；章节标题：最佳模型分析入口; 统一选模; 分区 |
| `results/model_evaluation/best_checkpoint_comparison/source_checksums.json` | JSON fields: logs\training\baseline_2d\epoch_001.json, logs\training\baseline_2d\epoch_002.json, logs\training\baseline_2d\epoch_003.json, logs\training\baseline_2d\epoch_004.json, logs\training\baseline_2d\epoch_005.json, logs\training\baseline_2d\epoch_006.json, logs\training\baseline_2d\epoch_007.json, logs\training\baseline_2d\epoch_008.json, logs\training\baseline_2d\epoch_009.json, logs\training\baseline_2d\epoch_010.json, logs\training\baseline_2d\epoch_011.json, logs\training\baseline_2d\epoch_012.json | JSON 字段：logs\training\baseline_2d\epoch_001.json, logs\training\baseline_2d\epoch_002.json, logs\training\baseline_2d\epoch_003.json, logs\training\baseline_2d\epoch_004.json, logs\training\baseline_2d\epoch_005.json, logs\training\baseline_2d\epoch_006.json, logs\training\baseline_2d\epoch_007.json, logs\training\baseline_2d\epoch_008.json, logs\training\baseline_2d\epoch_009.json, logs\training\baseline_2d\epoch_010.json, logs\training\baseline_2d\epoch_011.json, logs\training\baseline_2d\epoch_012.json |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/distant_body_fp.png` | Binary or large file; 199,648 bytes | 二进制或大文件；199,648 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/distant_body_fp.svg` | UTF-8 text with 1450 lines; 52,919 bytes | UTF-8 文本，1450 行；52,919 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/normalized_confusion.png` | Binary or large file; 234,074 bytes | 二进制或大文件；234,074 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/normalized_confusion.svg` | Binary or large file; 178,578 bytes | 二进制或大文件；178,578 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/patient_small_organ_dice.png` | Binary or large file; 80,576 bytes | 二进制或大文件；80,576 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/patient_small_organ_dice.svg` | UTF-8 text with 1335 lines; 40,426 bytes | UTF-8 文本，1335 行；40,426 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/precision_recall.png` | Binary or large file; 77,614 bytes | 二进制或大文件；77,614 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/precision_recall.svg` | UTF-8 text with 1968 lines; 56,645 bytes | UTF-8 文本，1968 行；56,645 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/volume_bias.png` | Binary or large file; 65,344 bytes | 二进制或大文件；65,344 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/figures/volume_bias.svg` | UTF-8 text with 1615 lines; 45,207 bytes | UTF-8 文本，1615 行；45,207 字节 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/README.md` | Entry point or directory guide; Sections: 补充研究指标 | 项目入口或目录说明；章节标题：补充研究指标 |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/tables/per_patient_best.csv` | Table with 30 data rows; columns: ﻿model, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate | 表格含 30 条数据行；列：﻿model, patient, small_organ_dice, Background, Body, Bones, Bladder, Rectum, Prostate |
| `results/model_evaluation/best_checkpoint_comparison/supplementary_research/tables/relative_volume_bias.csv` | Table with 180 data rows; columns: ﻿model, patient, label, relative_volume_bias | 表格含 180 条数据行；列：﻿model, patient, label, relative_volume_bias |
| `results/model_evaluation/generalization_diagnostics/figures/patient_influence.png` | Binary or large file; 82,673 bytes | 二进制或大文件；82,673 字节 |
| `results/model_evaluation/generalization_diagnostics/figures/patient_influence.svg` | UTF-8 text with 1787 lines; 55,059 bytes | UTF-8 文本，1787 行；55,059 字节 |
| `results/model_evaluation/generalization_diagnostics/provenance/local_source_checksums.json` | JSON fields: code\config\patient_split.json, logs\training\baseline_3d\epoch_050.json, logs\training\improved_3d\epoch_050.json, results\experiments\improved_3d_v2\training_records\epoch_045.json, results\experiments\improved_3d_v3\training_records\epoch_050.json, logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json | JSON 字段：code\config\patient_split.json, logs\training\baseline_3d\epoch_050.json, logs\training\improved_3d\epoch_050.json, results\experiments\improved_3d_v2\training_records\epoch_045.json, results\experiments\improved_3d_v3\training_records\epoch_050.json, logs\training\baseline_3d\epoch_001.json, logs\training\baseline_3d\epoch_002.json, logs\training\baseline_3d\epoch_003.json, logs\training\baseline_3d\epoch_004.json, logs\training\baseline_3d\epoch_005.json, logs\training\baseline_3d\epoch_006.json, logs\training\baseline_3d\epoch_007.json |
| `results/model_evaluation/generalization_diagnostics/README.md` | Entry point or directory guide; Sections: 泛化与失败机制复核 | 项目入口或目录说明；章节标题：泛化与失败机制复核 |
| `results/model_evaluation/generalization_diagnostics/tables/audit.json` | JSON fields: patient_id_split_disjoint, five_completed_runs, cross_split_image_duplicate_check, validation_v3_minus_standard, historical_test_v1_minus_standard, v3_test_status | JSON 字段：patient_id_split_disjoint, five_completed_runs, cross_split_image_duplicate_check, validation_v3_minus_standard, historical_test_v1_minus_standard, v3_test_status |
| `results/model_evaluation/generalization_diagnostics/tables/historical_test_patient_influence.csv` | Table with 6 data rows; columns: ﻿patient, standard, v1, v1_minus_standard | 表格含 6 条数据行；列：﻿patient, standard, v1, v1_minus_standard |
| `results/model_evaluation/generalization_diagnostics/tables/validation_error_sources.csv` | Table with 16 data rows; columns: ﻿model, target, tp, fp, fn, fp_from_background, fp_from_body | 表格含 16 条数据行；列：﻿model, target, tp, fp, fn, fp_from_background, fp_from_body |
| `results/README.md` | Entry point or directory guide; Sections: 结果与分析产物 | 项目入口或目录说明；章节标题：结果与分析产物 |
| `results/reports/DATA_AND_METHODS_REVIEW_ZH.md` | Sections: 前期数据与独立研究审查：完整阶段记录; 原文：HipMRI_Pretraining_Data_Analysis.md; HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 原文：HipMRI_P1_Independent_Review_20261020_ZH.md; P1 独立复审：证据、未决机制与下一项实验 | 章节标题：前期数据与独立研究审查：完整阶段记录; 原文：HipMRI_Pretraining_Data_Analysis.md; HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 原文：HipMRI_P1_Independent_Review_20261020_ZH.md; P1 独立复审：证据、未决机制与下一项实验 |
| `results/reports/README.md` | Entry point or directory guide; Sections: 研究报告与治理材料 | 项目入口或目录说明；章节标题：研究报告与治理材料 |
| `results/reports/RESEARCH_GOVERNANCE_ZH.md` | Sections: 研究治理与实验设计指令; P1：HipMRI 研究与课程交付独立审查; 1. 课程约束已核对，交付仍有明确缺口; 2. 源码复核带来的主要新判断; 3. G7 已完成什么，尚未证明什么 | 章节标题：研究治理与实验设计指令; P1：HipMRI 研究与课程交付独立审查; 1. 课程约束已核对，交付仍有明确缺口; 2. 源码复核带来的主要新判断; 3. G7 已完成什么，尚未证明什么 |
| `results/reports/RESEARCH_REPORT.md` | Sections: HipMRI Research Report; Dataset Characteristics; Problems Identified; Interventions; Experimental Results | 章节标题：HipMRI Research Report; Dataset Characteristics; Problems Identified; Interventions; Experimental Results |
| `results/STORAGE_GUIDE.md` | Sections: 文件存储规则 | 章节标题：文件存储规则 |
| `results/tables/comparisons/g5_validation_aux_logits.json` | JSON fields: kind, split, interpretation, checkpoint, body_audit_sha256, cases, summary | JSON 字段：kind, split, interpretation, checkpoint, body_audit_sha256, cases, summary |
| `results/tables/comparisons/g5_validation_body_audit.json` | JSON fields: kind, split, selection, cases, sources, pictured_cases | JSON 字段：kind, split, selection, cases, sources, pictured_cases |
| `results/tables/comparisons/g5_validation_case_class_metrics.csv` | Table with 456 data rows; columns: split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou | 表格含 456 条数据行；列：split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou |
| `results/tables/comparisons/g6_test_3d.json` | Binary or large file; 220,443 bytes | 二进制或大文件；220,443 字节 |
| `results/tables/comparisons/g6_test_case_class_metrics.csv` | Table with 360 data rows; columns: split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou | 表格含 360 条数据行；列：split, model, patient, case, class, gt_voxels, predicted_voxels, tp, fp, fn, dice, iou |
| `results/tables/comparisons/README.md` | Entry point or directory guide; Sections: Held-out Test comparison | 项目入口或目录说明；章节标题：Held-out Test comparison |

Generated by `code/tools/update_project_map.py`; edit the source files, then regenerate this index. / 由 `code/tools/update_project_map.py` 生成；修改源文件后重新生成本目录。
