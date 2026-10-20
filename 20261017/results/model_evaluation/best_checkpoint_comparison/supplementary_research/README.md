# 补充研究指标

这些指标用于解释模型行为，并非全部都是课程硬性指定的评分指标。

| 图表/指标 | 为什么有用 | 解释限制 |
|---|---|---|
| figures/precision_recall | 区分假阳性增加与漏检，观察器官取舍 | 需与Dice和逐患者结果一起看 |
| figures/normalized_confusion | 显示GT类流向哪些预测类 | GT-row标准化不是precision，低占比FP也可能量很大 |
| figures/patient_small_organ_dice | 防止平均值掩盖困难患者 | 六患者不能代表总体人群 |
| figures/volume_bias | 显示系统性体积高估/低估 | FP和FN会抵消；体积接近不代表位置正确 |
| figures/distant_body_fp | 揭示HD95可能忽略的远处小分量 | 数量要结合背景分母、患者等权及空间图 |
| 患者bootstrap/leave-one-out | 检查患者依赖 | [已有审查](../../generalization_diagnostics/README.md)；反复Validation选择未被校正 |
| ASSD、组件、缺片、面积过渡 | 分别看平均边界误差、碎片和不连续 | 待统一远端审计；连贯也可能位置完全错误 |

概率校准、ECE、NSD等尚无所需记录/统一定义，本轮不填不存在的数值。继续按具体研究问题增加指标，不以指标数量代替可信证据。
