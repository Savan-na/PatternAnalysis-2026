# Improved 3D v2 独立实验区

## 最新分析

[v2文献复核与设计](reports/RESEARCH_REPORT_ZH.md#design)现已与本版本结果报告同区保存；这是训练前方案，不是实验结果。

[中文系统分析](reports/RESEARCH_REPORT_ZH.md#results) · [六类图](figures/all_six_classes.svg) · [逐患者图](figures/patient_comparison.svg) · [训练/边界轨迹](figures/training_and_boundary_trajectory.svg) · [八张病例图审阅](case_studies/validation/review_45_50/README.md)。已下载52个JSON并完成本地一致性复算、8张图逐张审阅及关键epoch/manifest来源核对；v2最佳第45轮未达到预设成功标准。全量远端文件SHA清单仍待归档前补齐。

本目录保存独立深监督 v2 的分析及下载记录，不与历史2D、Standard 3D或Improved v1结果混存。跨版本图表放在本区，引用历史来源，不移动或覆盖历史结果。

| 路径 | 内容 |
|---|---|
| training_records/ | 远端完整epoch JSON、manifest及冻结配置副本 |
| provenance/ | 文件校验、Slurm日志/状态、代码配置版本、来源及实验索引 |
| tables/ | 六类、逐患者、逐轮及边界数值表 |
| figures/ | 训练轨迹和跨版本比较图 |
| case_studies/validation/ | 固定病例的MRI、GT、预测、FP/FN对照；每组图片附README |
| reports/ | v2中文系统分析、结论与归档建议 |

来源：Rangpur `logs/training/improved_3d_v2_ds_20261020/`，训练作业677856，正常完成50轮，用时01:56:54。JSON复算best为第45轮0.8362，第50轮0.7774，与日志一致。训练记录、统计图和Validation MRI对照均已在本区；绘图作业678349，四例均复现四组权重预测。

分析分别报告第45轮Validation最佳与固定第50轮，不用训练loss替代质量指标。比较包括六类Dice/IoU/precision/recall、Body和小器官边界、逐患者差异、远处误报、训练稳定性及资源成本。历史模型按各自既定选模规则引用，明确不同轮次与协议。

## Test边界

后续模型、参数及分支选择只依据Train/Validation。已查看Test只可作为历史描述，不能为v2选择轮次、阈值或修改方案。当前不自动提交新Test评估或绘图；若制作固定v2的Test描述，单列子目录并标明其已被查看、非新盲测，保持原Test记录。

## 最终归档与清理

当前不删除任何历史权重、日志、图片或原始数据。最终模型依Validation完整证据选择，不能预定v2胜出。未入选版本先归档，再核验，再依据具体清单清理冗余。

至少保留：冻结配置、代码版本/源码哈希、split哈希、作业与环境记录、epoch指标、选模依据、原始评估JSON、来源校验、关键图表和报告。指标可以证明记录一致性，但删除全部权重会失去精确复现预测的能力；建议归档每版本选中权重及其哈希，删除多余中间权重和可再生缓存前列出影响。归档位置应在本活动分析区之外，现阶段不创建空归档树。
