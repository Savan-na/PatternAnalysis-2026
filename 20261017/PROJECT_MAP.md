# Project File Map / 项目文件总览

Updated / 更新时间: **2026-10-18 10:37:27 AEST**

**English:** This generated index lists every directory and file under `20261017`, including local logs and generated artifacts. The content column summarizes structure (headings, code definitions, fields or file type) rather than copying another document's prose. Run `python code/tools/update_project_map.py --watch` to keep it current while the watcher is active.

**中文：** 本自动目录列出 `20261017` 下的全部目录和文件，包括本地日志及生成产物。内容栏概述章节、代码定义、字段或文件类型，不重复抄录其他文档。运行 `python code/tools/update_project_map.py --watch`，并保持进程运行，即可持续更新。

Directories / 目录: **42** · Files / 文件: **53**

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
│   │   └── splits.py
│   ├── tests/
│   │   ├── test_boundaries.py
│   │   ├── test_data.py
│   │   ├── test_metrics.py
│   │   ├── test_models_and_losses.py
│   │   ├── test_raw_audit.py
│   │   └── test_splits.py
│   ├── tools/
│   │   ├── audit_train_raw.py
│   │   ├── package_transfer.py
│   │   └── update_project_map.py
│   ├── transfer/
│   │   └── 20261017_source.zip
│   ├── README.md
│   └── requirements.txt
├── logs/
│   ├── audits/
│   ├── internal/
│   │   ├── map_watcher_stderr.txt
│   │   ├── map_watcher_stdout.txt
│   │   └── project_map_watcher.pid
│   ├── profiles/
│   ├── training/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   └── improved_3d/
│   ├── validation/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   └── improved_3d/
│   └── README.md
├── results/
│   ├── case_studies/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   ├── comparisons/
│   │   └── improved_3d/
│   ├── data_analysis/
│   │   ├── HipMRI_Pretraining_Data_Analysis.md
│   │   ├── HipMRI_Pretraining_Data_Analysis.previous-41201a5e.bak
│   │   ├── HipMRI_Pretraining_Data_Analysis.previous-5dc166bd.bak
│   │   ├── historical_patch_vs_full_volume.svg
│   │   ├── historical_resampling_volume_ratio.svg
│   │   ├── historical_train_case_profile.csv
│   │   ├── historical_train_statistics.json
│   │   ├── train_class_distribution.svg
│   │   ├── train_foreground_distribution.svg
│   │   ├── train_patient_examinations.svg
│   │   └── train_rare_organ_volumes.svg
│   ├── figures/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   ├── comparisons/
│   │   └── improved_3d/
│   ├── reports/
│   ├── tables/
│   │   ├── baseline_2d/
│   │   ├── baseline_3d/
│   │   ├── comparisons/
│   │   └── improved_3d/
│   └── README.md
├── .gitignore
├── PROJECT_MAP.md
└── README.md
```

## File Contents / 文件内容概览

| File / 文件 | English content overview | 中文内容概览 |
|---|---|---|
| `.gitignore` | Git rules for local artifacts; UTF-8 text with 19 lines; 327 bytes | 本地运行产物的 Git 规则；UTF-8 文本，19 行；327 字节 |
| `code/config/arms/baseline_2d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/baseline_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/arms/improved_3d/arm.json` | Reserved model-arm identity and training state; JSON fields: arm_id, role, model_import, training_status, training_config_status | 预留的模型分组标识及训练状态；JSON 字段：arm_id, role, model_import, training_status, training_config_status |
| `code/config/patient_split.json` | Frozen patient groups and expected counts; JSON fields: provenance, train, validation, test, expected_volumes | 固定患者分组与预期数量；JSON 字段：provenance, train, validation, test, expected_volumes |
| `code/docs/EXECUTION.md` | Evidence and execution gates; Sections: Execution and evidence checklist; Evidence identity; G1: raw integrity gate; Later gates; Outputs still required | 证据与执行关口；章节标题：Execution and evidence checklist; Evidence identity; G1: raw integrity gate; Later gates; Outputs still required |
| `code/docs/RANGPUR_COMMANDS.md` | Separate Windows and Rangpur commands; Sections: Rangpur G1 commands and terminal boundaries; 1. Read-only location check; 2. Pending confirmed paths | 分开的 Windows 与 Rangpur 命令；章节标题：Rangpur G1 commands and terminal boundaries; 1. Read-only location check; 2. Pending confirmed paths |
| `code/docs/SOURCE_DECISIONS.md` | Paper sources and implementation deviations; Sections: Paper and assignment decision register | 论文来源与实现差异；章节标题：Paper and assignment decision register |
| `code/docs/STUDY_PROTOCOL.md` | Research hypotheses, endpoints, and freeze rules; Sections: HipMRI research protocol — data-first revision; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints | 研究假设、终点指标与冻结规则；章节标题：HipMRI research protocol — data-first revision; Research question and evidence order; Cohort and information boundaries; Controlled experiments and endpoints |
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
| `code/hipmri_rewrite/splits.py` | Frozen patient identity and MRI/mask file pairing; Definitions: Case, read_split, discover | 固定患者身份及 MRI/mask 文件配对；定义：Case, read_split, discover |
| `code/README.md` | Entry point or directory guide; Sections: Code and Methods | 项目入口或目录说明；章节标题：Code and Methods |
| `code/requirements.txt` | Python dependency ranges; UTF-8 text with 4 lines; 60 bytes | Python 依赖版本范围；UTF-8 文本，4 行；60 字节 |
| `code/tests/test_boundaries.py` | Physical direction and empty-surface tests; Definitions: BoundaryTests, SurfaceTests | 物理方向及空表面测试；定义：BoundaryTests, SurfaceTests |
| `code/tests/test_data.py` | NIfTI alignment, downsampling and label tests; Definitions: DataContractTests, DownsamplingGeometryTests, NiftiPairTests | NIfTI 对齐、降采样与标签测试；定义：DataContractTests, DownsamplingGeometryTests, NiftiPairTests |
| `code/tests/test_metrics.py` | Class-confusion and overlap metric tests; Definitions: OverlapTests | 类别混淆与重叠指标测试；定义：OverlapTests |
| `code/tests/test_models_and_losses.py` | Shape, gradient, supervision and inference tests; Definitions: NetworkTests | 形状、梯度、深监督与推理测试；定义：NetworkTests |
| `code/tests/test_raw_audit.py` | Train inventory and file-hash audit tests; Definitions: RawAuditInventoryTests | 训练集文件盘点与文件哈希审计测试；定义：RawAuditInventoryTests |
| `code/tests/test_splits.py` | Patient isolation and file pairing tests; Definitions: PatientSplitTests | 患者隔离与文件配对测试；定义：PatientSplitTests |
| `code/tools/audit_train_raw.py` | Train-only raw MRI/mask integrity audit; Definitions: train_inventory, sha256_file, audit_case, main | 仅对训练集原始 MRI/Mask 做完整性审计；定义：train_inventory, sha256_file, audit_case, main |
| `code/tools/package_transfer.py` | Build and verify a source-only transfer ZIP; Definitions: main | 创建并校验仅含源码的传输包；定义：main |
| `code/tools/update_project_map.py` | Generate and watch this bilingual inventory; Definitions: _files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main | 生成并持续更新本双语目录；定义：_files_and_dirs, _tree, _read_text, _content, _escape, render, refresh, _fingerprint, main |
| `code/transfer/20261017_source.zip` | Generated source-only transfer package; ZIP archive with 64 members; 49,355 bytes | 生成的仅含源码传输包；ZIP 压缩包，含 64 个成员；49,355 字节 |
| `logs/internal/map_watcher_stderr.txt` | UTF-8 text with 0 lines; 0 bytes | UTF-8 文本，0 行；0 字节 |
| `logs/internal/map_watcher_stdout.txt` | UTF-8 text with 0 lines; 0 bytes | UTF-8 文本，0 行；0 字节 |
| `logs/internal/project_map_watcher.pid` | UTF-8 text with 1 lines; 7 bytes | UTF-8 文本，1 行；7 字节 |
| `logs/README.md` | Entry point or directory guide; Sections: Experiment Logs | 项目入口或目录说明；章节标题：Experiment Logs |
| `PROJECT_MAP.md` | Generated bilingual tree and file content index; Self-entry; regenerated from current tree and file structure | 自动生成的双语文件树与内容索引；本文件条目；根据当前文件树和文件结构重新生成 |
| `README.md` | Entry point or directory guide; Sections: HipMRI Data-Driven Model Development; Where things belong | 项目入口或目录说明；章节标题：HipMRI Data-Driven Model Development; Where things belong |
| `results/data_analysis/HipMRI_Pretraining_Data_Analysis.md` | Sections: HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 1. Independent verification of the available record; 2. What the raw label statistics show; 3. Unusual cases to inspect first; 4. What remains unknown | 章节标题：HipMRI Raw Train Data: Statistical Review and Improvement Priorities; 1. Independent verification of the available record; 2. What the raw label statistics show; 3. Unusual cases to inspect first; 4. What remains unknown |
| `results/data_analysis/HipMRI_Pretraining_Data_Analysis.previous-41201a5e.bak` | UTF-8 text with 106 lines; 14,485 bytes | UTF-8 文本，106 行；14,485 字节 |
| `results/data_analysis/HipMRI_Pretraining_Data_Analysis.previous-5dc166bd.bak` | UTF-8 text with 93 lines; 12,305 bytes | UTF-8 文本，93 行；12,305 字节 |
| `results/data_analysis/historical_patch_vs_full_volume.svg` | UTF-8 text with 49 lines; 4,617 bytes | UTF-8 文本，49 行；4,617 字节 |
| `results/data_analysis/historical_resampling_volume_ratio.svg` | UTF-8 text with 768 lines; 59,609 bytes | UTF-8 文本，768 行；59,609 字节 |
| `results/data_analysis/historical_train_case_profile.csv` | Table with 143 data rows; columns: case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels | 表格含 143 条数据行；列：case, patient_id, week, shape_x, shape_y, shape_z, raw_voxel_volume_mm3, mask_header_unit, raw_background_voxels, raw_body_voxels, raw_bones_voxels, raw_bladder_voxels |
| `results/data_analysis/historical_train_statistics.json` | JSON fields: scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts | JSON 字段：scope, source_jsonl_sha256, frozen_split_sha256, records, denominators, class_rows, patients, patient_exam_count_quantiles, source_shape_counts, resampled_shape_counts, source_voxel_volume_mm3_quantiles, mask_header_unit_counts |
| `results/data_analysis/train_class_distribution.svg` | UTF-8 text with 35 lines; 3,679 bytes | UTF-8 文本，35 行；3,679 字节 |
| `results/data_analysis/train_foreground_distribution.svg` | UTF-8 text with 32 lines; 3,368 bytes | UTF-8 文本，32 行；3,368 字节 |
| `results/data_analysis/train_patient_examinations.svg` | UTF-8 text with 72 lines; 7,370 bytes | UTF-8 文本，72 行；7,370 字节 |
| `results/data_analysis/train_rare_organ_volumes.svg` | UTF-8 text with 320 lines; 25,203 bytes | UTF-8 文本，320 行；25,203 字节 |
| `results/README.md` | Entry point or directory guide; Sections: Results and Report Artifacts | 项目入口或目录说明；章节标题：Results and Report Artifacts |

Generated by `code/tools/update_project_map.py`; edit the source files, then regenerate this index. / 由 `code/tools/update_project_map.py` 生成；修改源文件后重新生成本目录。
