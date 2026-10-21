# 全部模型性能对比图表

本目录汇总当前10个模型的最佳Validation记录、已有7模型Test审计和相关统计图。新图表独立存放，不覆盖历史实验结果。重建脚本：[build_model_comparison_gallery.py](../../../code/tools/build_model_comparison_gallery.py)。

## 先看这三组图

1. [所有模型主要 Dice 对比](figures/all_models_primary_dice.png)：Validation覆盖10模型；已有Test覆盖7模型。v5-C/L/S的Test位置标记“No Test”，不是0分。
2. [六类 Validation Dice](figures/validation_dice_six_classes.png)：包含Background、Body、Bones、Bladder、Rectum、Prostate。
3. [Validation HD95极端值](figures/validation_hd95_max_mm.png)：检查高平均Dice掩盖的异常边界错误。应与[中位数](figures/validation_hd95_median_mm.png)和[P95](figures/validation_hd95_p95_mm.png)一起看。

## 模型与最佳轮次

| 图表名称 | 含义 | 最佳Validation轮次 | Validation小器官Dice | 既有Test小器官Dice |
|---|---|---:|---:|---:|
| 2D | 补充二维基线 | 50 | 0.77569 | 0.70822 |
| Standard | 标准3D U-Net | 50 | 0.85022 | 0.78510 |
| v1 | 初始残差/相加深监督Improved | 50 | 0.84200 | 0.80778 |
| v2 | 独立深监督、主头推理 | 45 | 0.83620 | 0.79590 |
| v3 | v1基础上改用GN1 | 50 | **0.85636** | 0.81448 |
| A0 | v4优化器/学习率配方对照 | 45 | 0.84500 | 0.78365 |
| A1 | A0加配对3D空间增强 | 45 | 0.82976 | 0.78012 |
| v5-C | 原始v3同期重跑对照 | 50 | 0.85424 | 未评估 |
| v5-L | C基础上压缩小器官CE权重 | 45 | 0.84256 | 未评估 |
| v5-S | C基础上仅使用余弦学习率 | 45 | 0.84649 | 未评估 |

选模只根据原先规定的Validation患者均衡三个小器官Dice，每5轮测量一次，共10个候选，同分保留更早轮次。不是强制对比第50轮，也不是根据Test选择轮次。图中e45/e50表示所选轮次。

## 完整图表索引

共21组图，每组同时生成PNG和SVG。PNG便于本机查看，SVG便于放大、排版与导出。

| 分析内容 | Validation：10模型 | 已有Test：7模型 |
|---|---|---|
| 六类Dice | [PNG](figures/validation_dice_six_classes.png) / [SVG](figures/validation_dice_six_classes.svg) | [PNG](figures/test_dice_six_classes.png) / [SVG](figures/test_dice_six_classes.svg) |
| 六类IoU | [PNG](figures/validation_iou_six_classes.png) / [SVG](figures/validation_iou_six_classes.svg) | [PNG](figures/test_iou_six_classes.png) / [SVG](figures/test_iou_six_classes.svg) |
| 六类precision | [PNG](figures/validation_precision_six_classes.png) / [SVG](figures/validation_precision_six_classes.svg) | [PNG](figures/test_precision_six_classes.png) / [SVG](figures/test_precision_six_classes.svg) |
| 六类recall | [PNG](figures/validation_recall_six_classes.png) / [SVG](figures/validation_recall_six_classes.svg) | [PNG](figures/test_recall_six_classes.png) / [SVG](figures/test_recall_six_classes.svg) |
| 五前景类HD95中位数 | [PNG](figures/validation_hd95_median_mm.png) | [PNG](figures/test_hd95_median_mm.png) |
| 五前景类HD95跨volume P95 | [PNG](figures/validation_hd95_p95_mm.png) | [PNG](figures/test_hd95_p95_mm.png) |
| 五前景类HD95最大值 | [PNG](figures/validation_hd95_max_mm.png) | [PNG](figures/test_hd95_max_mm.png) |
| 五前景类ASSD中位数 | [PNG](figures/validation_assd_median_mm.png) | [PNG](figures/test_assd_median_mm.png) |
| 逐患者小器官Dice | [PNG](figures/validation_patient_small_organ_dice.png) | [PNG](figures/test_patient_small_organ_dice.png) |

另有[10模型训练/Validation曲线](figures/all_models_training_trajectories.png)、[主要Dice汇总](figures/all_models_primary_dice.png)、[已有标准化资源基准](figures/available_resource_benchmarks.png)。边界图和患者图的同名SVG也均保存在figures中。

## 指标如何读

- Dice/IoU/precision/recall：每名患者先合并其所有visit的混淆矩阵，分别计算各类指标，再对6名患者等权平均。不是所有体素混在一起，也不是每次扫描等权。小器官主要指标平均Bladder、Rectum、Prostate；全前景宏平均另含Body和Bones；六类宏平均又包含Background，三者不可混称。
- Validation为6患者38卷，Test为另一组6患者30卷。逐患者图用于显示平均数掩盖的个体差异；重复visit不等于新增独立患者。
- HD95/ASSD：单位mm，越低越好。本目录为各volume统计的中位数、P95或最大值，不是患者等权；图上明确标出汇总方式。HD95图的P95是“38/30个volume各自HD95的第95百分位”，不是另算一种表面HD95。边界热图使用对数颜色刻度，格内数字为原始mm。
- Background不计算表面指标；五个前景类全部覆盖。本次记录中的这些表面比较均为GT与预测同时存在，脚本会拒绝将空掩码异常默默当作正常距离。
- 训练总loss因目标权重、监督和采样不同不能作为跨模型质量排名。资源图只有已有7模型同一参考病例的基准；v5未做同协议推理基准，因此不补造延迟或显存数值。GPU负载等因素仍可影响耗时。

## 证据与解释边界

每个模型读取全部50份epoch记录重新选择best；从所选Validation混淆计数重新计算指标。旧7模型的表面数据来自固定checkpoint统一审计，并先核对与选模记录的混淆矩阵完全一致。v5来自其原始完整Validation记录，三组52份JSON的聚合SHA已与Rangpur输出一致。输入文件和选模轮次校验值保存在[source_checksums.json](source_checksums.json)。

本目录只读取已有记录，不训练、不重新执行Test推理、不改变历史结果。既有Test已被多轮查看，只能描述当前队列表现，不能称为新的盲测或独立外部泛化验证。v5没有Test成绩，不能据此宣称在Test优于其他模型。

大部分是单种子、固定小患者集实验；2D与3D配方也不完全相同。图表不附伪精确的显著性或不适当的独立扫描置信区间。历史v3与v5-C配方相同但结果略有差异，提示不可将每个小数差异全部解释为单个设计因素的因果效果。

## 可下载统计表

- [模型汇总](tables/model_summary.csv)：主要小器官Dice、五前景宏平均、六类宏平均和best轮次。
- [六类完整指标](tables/six_class_metrics.csv)：Dice、IoU、precision、recall。
- [边界指标](tables/surface_metrics.csv)：HD95中位数/P95/最大值、ASSD中位数。
- [逐患者指标](tables/patient_scores.csv)、[50轮训练曲线](tables/training_curves.csv)、[已有资源基准](tables/available_resource_benchmarks.csv)。

CSV使用UTF-8 BOM，便于本机Excel读取。统计图不替代真实MRI误分审阅，v5病例图和分析见[Validation报告](../../experiments/improved_3d_v5/reports/VALIDATION_SCREEN_ZH.md)。
