# 最佳模型分析入口

**当前七模型最新结果已完成，见[七模型综合报告](../../experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)和[课程指标图表](../../experiments/improved_3d_v4/assessment_analysis/assessment_required/)。** 本目录保留五模型阶段的Validation快照及方法历史，不再作为最新完整Test/资源结果入口。[方法与课程复核报告](../../experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md#methods)保留学习率、三篇论文及loss的原始审查。

## 统一选模

每模型按其50轮内已记录Validation的小器官患者等权Dice选best，平分取早。当前2D50、Standard50、v1-50、v2-45、v3-50。每5轮验证，共10个候选；不按Test或训练loss选轮次，不称逐一验证过全部50轮权重。

## 分区

- [课程要求相关](assessment_required/README.md)：逐类Dice/IoU、loss/指标、资源及关键边界证据。
- [补充研究指标](supplementary_research/README.md)：precision/recall、错类流向、患者差异、体积偏差、远处误报等。

本目录图表是五模型Validation best历史快照。独立七模型任务已完成描述性Test与同条件边界/资源评估，最新结果在上方链接的v4专区，不回写覆盖这份历史快照。

复现：项目根目录 `python -B code/tools/build_best_checkpoint_review.py`。源JSON不修改，哈希保存在`source_checksums.json`；更具体的选模SHA在assessment_required/tables/selected_checkpoints.json。所有CSV包含精确数字，PNG便于查看、SVG便于报告排版。
