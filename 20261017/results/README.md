# 结果与分析产物

本目录保存由实验记录解释得到的结果。每组新结果应标注来源run、checkpoint、数据split、指标定义和缺失项；指标、图像和结论不能脱离原记录。

| 内容 | 入口 |
|---|---|
| 当前已完成模型统一best比较 | [v3/analysis](model_evaluation/best_checkpoint_comparison/README.md) |
| 版本专属设计、结果和证据 | [v2](experiments/improved_3d_v2/README.md)、[v3](experiments/improved_3d_v3/README.md)、[v4](experiments/improved_3d_v4/README.md) |
| 原始Train数据问题、预处理与采样 | [data_analysis](data_analysis/README.md) |
| 固定权重机制/推理诊断 | [diagnostics](diagnostics/inference_context/README.md) |
| 历史G5/G6统计图与原始评估 | [figures/comparisons](figures/comparisons/README.md)、[tables/comparisons](tables/comparisons/README.md) |
| 真实MRI、GT、预测和错误对照 | [case_studies](case_studies/comparisons/README.md)；v2/v4案例在各自实验区 |
| 完整研究与治理报告 | [reports](reports/README.md) |
| 清理来源与完整性检查 | [documentation_audit](documentation_audit/README.md) |

历史图仍保留其独立问题：pooled指标、Test错误案例、v2末期退步、P2机制诊断等不等于重复版本。已被五模型best完全覆盖的旧v3概览图已删除。PNG与SVG用途不同，前者便于查看，后者用于出版排版；数值CSV是精确分析依据，不视为重复文件。

数据位置与输出约定见[STORAGE_GUIDE](STORAGE_GUIDE.md)。项目训练进度只从[项目入口](../README.md)及实验入口维护。
