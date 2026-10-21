# 模型对比：去除Background，重点分析Prostate

主报告：[作业指标与全部模型简明分析](ASSESSMENT_FOREGROUND_PROSTATE_ZH.md)。

本目录只保留4张核心PNG：

1. [五前景Dice与前列腺50%加权Dice](figures/01_foreground_and_prostate_priority.png)
2. [前列腺Validation/Test Dice](figures/02_prostate_validation_test.png)
3. [五前景逐类Dice/IoU](figures/03_foreground_dice_iou.png)
4. [前列腺precision、recall与边界风险](figures/04_prostate_precision_recall_boundaries.png)

精确数据：[汇总及加权结果](tables/model_summary.csv)、[五前景完整指标](tables/foreground_metrics.csv)、[表面指标](tables/surface_metrics.csv)、[资源基准](tables/available_resource_benchmarks.csv)。逐患者与训练曲线CSV仍保留，原始实验记录位于各模型实验区。

复算入口：`python -B 20261017/code/tools/build_model_comparison_gallery.py`，先验证原始记录，再调用[紧凑报告绘图代码](../../../code/tools/summarize_prostate_priority.py)。输入校验值和明确的分析权重见[source_checksums.json](source_checksums.json)。不训练、不重新执行Test、不改变任何checkpoint。
