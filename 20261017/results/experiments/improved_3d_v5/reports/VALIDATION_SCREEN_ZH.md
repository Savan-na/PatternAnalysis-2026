# v5 三分支 Validation 分析：C、类别权重 L、学习率 S

## 1. 结论

三组均完成50轮。各自按固定 Validation 规则选择最佳轮次，**本轮不推荐用 L 或 S 替换 v3/C**。L 的膀胱表现更好，但直肠、前列腺和 Body 错误明显增加；S 的前列腺有小幅收益，但直肠及其他指标回退。没有全面改善，也没有证据证明泛化能力增强。

| 分支 | 最佳轮次 | 患者均衡小器官 Dice | 相对 C | 优于 C 的患者数 |
|---|---:|---:|---:|---:|
| C：原始 v3 同期对照 | 50 | **0.854239** | — | — |
| L：小器官 CE 权重平方根压缩 | 45 | 0.842564 | −0.011675 | 2/6 |
| S：Adam 下每次更新余弦衰减 | 45 | 0.846494 | −0.007745 | 2/6 |

历史 v3 的最佳 Validation 为0.856356，C低0.002118。配方相同也未达到完全数值重现；这是解释小幅差异的重要背景，不能把每个差异完全归因于干预。GPU随机内核、运行环境或训练随机轨迹的具体贡献尚未证明。

## 2. 数据与核对范围

本地三组各52份JSON：50轮记录、冻结配置和manifest，共156份。原始记录不修改。分析脚本为 [summarize_v5_screen.py](../../../../code/tools/summarize_v5_screen.py)。

已核对：配置和训练源码SHA与冻结版本一致；三组实际50轮病例选择和类别曝光记录完全一致；每轮130次更新；全部19,500次更新的实际学习率符合配方；每5轮一次完整38个volume、6名患者的Validation；全部30次Validation的患者混淆矩阵重算得分与记录一致；各组best轮次与最终记录一致。

学习率校验允许极小浮点容差（相对1e−13、绝对1e−18），用于Windows/Linux余弦函数最后几位差异，不允许改动训练曲线。

**状态限制：**本地结构、源码和数值核对通过；远端156份JSON的聚合SHA对照待完成。尚未重新载入三组best权重生成真实分割图，也未做独立推理复现。此报告依据记录中的Validation混淆计数与物理表面距离，不从截图估计性能。本轮没有新增Test结果。

## 3. 六类覆盖

下表是每名患者先合并其所有visit混淆计数，再计算指标，最后对六名患者等权平均。不能与体素池化指标混用。

| 类别 | C Dice | L Dice | S Dice | 观察 |
|---|---:|---:|---:|---|
| Background | 0.99658 | 0.99442 | 0.99610 | C较好 |
| Body | 0.98041 | 0.97368 | 0.97740 | 两项干预都回退 |
| Bones | 0.88751 | 0.86646 | 0.87202 | 两项干预都回退 |
| Bladder | 0.90824 | **0.92916** | 0.90778 | L有明确的本次运行收益 |
| Rectum | **0.82869** | 0.78341 | 0.80339 | L下降最大 |
| Prostate | 0.82579 | 0.81513 | **0.82831** | S只提高0.00252，不能称显著提升 |

| 小器官 | C precision / recall | L precision / recall | S precision / recall |
|---|---:|---:|---:|
| Bladder | 0.87857 / 0.94532 | **0.91249 / 0.94797** | 0.87664 / 0.94546 |
| Rectum | **0.78137** / 0.89041 | 0.69352 / 0.91042 | 0.72402 / 0.91036 |
| Prostate | **0.82867** / 0.83078 | 0.77758 / 0.86331 | 0.79964 / **0.86534** |

L并没有普遍实现“权重下降→precision上升”的预期：只有Bladder符合，Rectum/Prostate反而precision下降、recall上升。weighted CE 的加权均值分母、Dice耦合、优化轨迹和不同最佳轮次都可能参与，**这些属于待区分机制，不是已证明原因**。因此不宜沿该方向继续盲目加大压缩。

## 4. 边界和远处误报

表面指标是38个volume的HD95中位数/最大值，单位毫米，越低越好；不是患者等权值，也不是所有表面点池化值。完整P95和ASSD见CSV。

| 前景类 | C HD95 中位/最大 | L HD95 中位/最大 | S HD95 中位/最大 |
|---|---:|---:|---:|
| Body | **4.50 / 9.50** | 5.22 / 43.67 | 4.86 / 13.44 |
| Bones | 4.75 / 18.23 | 7.25 / 48.26 | 6.72 / **13.63** |
| Bladder | 3.97 / 88.68 | **3.36 / 66.12** | 5.00 / 76.65 |
| Rectum | **6.24 / 22.10** | 13.65 / 175.48 | 9.42 / 35.77 |
| Prostate | 4.58 / **20.16** | 4.58 / 112.51 | 4.58 / 23.82 |

GT Background中距真实Body至少20mm、被预测为Body的体素总数：**C 2,229；L 104,524；S 7,272**。三组是同一Validation几何与病例，可直接比较计数；这不是患者等权率，也不是推理时使用GT过滤预测。

L 的前列腺和直肠异常尾部明显恶化；S 的前列腺HD95跨volume P95为7.04mm，优于C的7.76mm，但最大值反而较高。整体改善必须兼顾尾部和各器官，不能只选一个有利统计量。

## 5. 患者差异与公平性

L只在K018、S022超过C；S只在K008、S022超过C。样本单位是6名患者，38次扫描不能当38个独立患者。十个候选checkpoint中各自选best是预定比较，但不同best轮次和两项筛选仍增加解释限制；同轮次轨迹也应一起看。

三组训练加checkpoint记录耗时约3.08/3.00/3.03小时，另有Validation耗时；不包括全部排队、初始化和所有系统开销。日志记录的训练峰值allocated显存均约0.285GiB，不等于进程总显存或Slurm资源利用率。本轮没有独立标准化推理延迟基准。

## 6. 下一步

1. 完成远端聚合SHA核对，锁定这次记录；保留全部权重，不因分数下降丢弃证据。
2. 从38例Validation的逐类Dice/HD95尾部和分支差异确定少量共同病例，对C50、L45、S45生成同病例、同切片的六类图与FP/FN图；必须先重现原混淆计数。尤其查看L直肠/前列腺异常、L膀胱改善、S前列腺收益以及Body误报。此步骤尚未完成。
3. 当前保留历史v3作为既有候选，C作为新同期对照；L、S均不晋级为最终模型。没有必要为了寻找Test上偶然的胜出而立即再次遍历Test。
4. 不把本次结果推广为“学习率衰减无用”或“类别加权无用”。能否改用不同衰减时机、强度增强或目标采样，须结合真实图像和Train曝光证据另立单因素假设；本轮没有足够证据直接冻结下一套复杂配方。

## 7. 可复查输出

- [六类 Dice/IoU/precision/recall 图](../comparisons/validation_analysis/figures/six_class_metrics.png)
- [训练、Validation与学习率曲线](../comparisons/validation_analysis/figures/training_and_selection.png)
- [患者差异和边界尾部图](../comparisons/validation_analysis/figures/patients_and_boundary_tails.png)
- [汇总表](../comparisons/validation_analysis/tables/summary.csv)、[六类指标](../comparisons/validation_analysis/tables/six_class_patient_equal.csv)、[表面指标](../comparisons/validation_analysis/tables/surface_metrics.csv)
- [逐病例六类指标](../comparisons/validation_analysis/tables/case_metrics.csv)、[逐患者指标](../comparisons/validation_analysis/tables/patient_scores.csv)、[全部训练曲线](../comparisons/validation_analysis/tables/training_curves.csv)

PNG、原始训练JSON及病例图保留在本地研究目录；Git中发布分析代码、统计CSV、SVG与本报告。病例图尚未生成，不将统计图冒充实际MRI分割可视化。
