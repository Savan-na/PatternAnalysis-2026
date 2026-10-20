# 最佳模型分析入口

本目录是当前跨模型分析的最新入口。[方法与课程复核报告](../../experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md#methods)解释学习率、三篇论文、loss及尚缺证据。

## 统一选模

每模型按其50轮内已记录Validation的小器官患者等权Dice选best，平分取早。当前2D50、Standard50、v1-50、v2-45、v3-50。每5轮验证，共10个候选；不按Test或训练loss选轮次，不称逐一验证过全部50轮权重。

## 分区

- [课程要求相关](assessment_required/README.md)：逐类Dice/IoU、loss/指标、资源及关键边界证据。
- [补充研究指标](supplementary_research/README.md)：precision/recall、错类流向、患者差异、体积偏差、远处误报等。

当前图表是Validation best结果。新五模型描述性Test与同条件边界/资源评估尚待远端任务，不能用Validation图代称Test结果。既有根目录figures和历史报告保留为旧阶段记录，主要比较优先引用本入口。

复现：项目根目录 `python -B code/tools/build_best_checkpoint_review.py`。源JSON不修改，哈希保存在`source_checksums.json`；更具体的选模SHA在assessment_required/tables/selected_checkpoints.json。所有CSV包含精确数字，PNG便于查看、SVG便于报告排版。
