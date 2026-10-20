# Train数据质量与分布分析

本目录按研究问题命名。每组图片保留就近README、来源和解释，读者无需了解旧阶段编号。

| 目录 | 问题与用途 |
|---|---|
| [representative_alignment_review](representative_alignment_review/README.md) | 六例代表性MRI与标签配对筛查 |
| [label_component_review](label_component_review/README.md) | 两例Bladder微小连通组件是否属于标签异常 |
| [patient_alignment_screening](patient_alignment_screening/README.md) | 全26患者、143卷的系统性配准/标注视觉筛查 |
| [training_distribution](training_distribution/README.md) | 六类体素、重复扫描、强度、空间与切片分布 |
| [preprocessing_validation](preprocessing_validation/README.md) | 降采样的器官保持与候选裁剪曝光 |
| [patient_balanced_sampling](patient_balanced_sampling/README.md) | 模型实际接收的患者和类别曝光、patch边界覆盖 |
| [boundary_sampling_audit](boundary_sampling_audit/README.md) | 边界采样的CPU可行性与曝光代价；未训练模型 |
| [legacy_training_statistics](legacy_training_statistics/README.md) | 唯一旧统计证据，与当前预处理结果区分 |

综合论证见[研究报告](../reports/RESEARCH_REPORT.md)。原始机器记录继续保留历史run ID；[路径登记](../../code/config/artifact_paths.json)可追溯旧目录，当前展示与输出使用上表主题名称。
