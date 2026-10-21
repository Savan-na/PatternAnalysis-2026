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

**校验与复现范围：**本地结构、源码和数值核对通过；用户提供的远端三组聚合SHA与本地完全一致，覆盖156份JSON，校验回执见分析目录的 `remote_record_hashes.json`。绘图作业682157成功完成，重新载入C50/L45/S45，在四个共同Validation病例复现全部12个原始混淆矩阵，生成9张PNG，下载后逐张SHA校验通过并完成目视审阅。复现范围不是全部38卷，也未独立重算表面距离；表面指标仍来自原始完整Validation记录。指标不从截图估计。该阶段未使用Test选模；后续固定checkpoint的Test补齐结果见第9节。

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

1. 远端聚合SHA核对已完成，三组记录一致；保留全部权重，不因分数下降丢弃证据。
2. 四个共同病例图已生成和审阅，覆盖L直肠/前列腺异常、L膀胱改善、S前列腺收益以及Body误报，详见下节。暂不增加无明确问题的GPU诊断。
3. 当前保留历史v3作为既有候选，C作为新同期对照；L、S均不晋级为最终模型。没有必要为了寻找Test上偶然的胜出而立即再次遍历Test。
4. 不把本次结果推广为“学习率衰减无用”或“类别加权无用”。能否改用不同衰减时机、强度增强或目标采样，须结合真实图像和Train曝光证据另立单因素假设；本轮没有足够证据直接冻结下一套复杂配方。

## 7. 可复查输出

- [六类 Dice/IoU/precision/recall 图](../comparisons/validation_analysis/figures/six_class_metrics.png)
- [训练、Validation与学习率曲线](../comparisons/validation_analysis/figures/training_and_selection.png)
- [患者差异和边界尾部图](../comparisons/validation_analysis/figures/patients_and_boundary_tails.png)
- [汇总表](../comparisons/validation_analysis/tables/summary.csv)、[六类指标](../comparisons/validation_analysis/tables/six_class_patient_equal.csv)、[表面指标](../comparisons/validation_analysis/tables/surface_metrics.csv)
- [逐病例六类指标](../comparisons/validation_analysis/tables/case_metrics.csv)、[逐患者指标](../comparisons/validation_analysis/tables/patient_scores.csv)、[全部训练曲线](../comparisons/validation_analysis/tables/training_curves.csv)

PNG、原始训练JSON及病例图保留在本地研究目录；Git中发布分析代码、统计CSV、SVG、病例复现校验回执与本报告。

## 8. 已完成的真实图像审阅

四个病例按干预差异和异常尾部选择，属于有目的的案例解释，不代表总体发生率。每图固定相同病例/切片；青线GT，橙线预测，红色FP，蓝色FN。图中标示的是NIfTI网格XY/XZ/YZ及实际spacing比例，未把网格轴直接命名为经独立验证的解剖方向。六类图标题Dice来自完整三维volume，不是所展示切片的Dice。

| 病例/类别 | C Dice / HD95 mm | L Dice / HD95 mm | S Dice / HD95 mm | 目视审阅与数值支持的判断 |
|---|---:|---:|---:|---|
| S022_Week6 Bladder | 0.594 / 88.68 | **0.925 / 3.36** | 0.826 / 76.65 | 三平面图中C、S存在与主目标分离的红色误报；L明显减少这些可见误报。L的precision从C的0.431提高到0.890，recall基本保留 |
| S022_Week6 Rectum | 0.874 / 3.36 | 0.785 / 175.48 | **0.877 / 3.36** | L在展示平面上局部轮廓也有偏差，但这些中心平面没有充分显示导致175mm尾部的远处错误；不能用中心切片看似接近来否定全volume错误 |
| J026_Week6 Prostate | **0.810 / 4.75** | 0.646 / 112.51 | 0.761 / 6.76 | L目标附近的橙线外扩，precision从0.702降至0.486、recall均约0.96。当前中心平面不足以定位112mm误差源，尚不能断言具体连通结构或成因 |
| K018_Week4 Body | **0.976 / 4.58** | 0.938 / 40.71 | 0.974 / 6.24 | 共用远处FP位置平面显示L在真实Body轮廓之外形成大片红色区域。远处Body误报C/L/S分别313/19,913/710体素，与完整记录一致 |
| K018_Week2 Prostate | 0.782 / 7.41 | **0.894 / 3.36** | 0.882 / 3.36 | C有较明显蓝色漏分；L、S更完整覆盖GT。C recall为0.672，L/S为0.921/0.872，同时precision由0.934降至0.869/0.893，体现覆盖与误报取舍 |

**从图像得到的新增认识：**C也会出现严重的类别特异性远处误报，不能将“主要选模指标最高”解释为所有类别和病例都稳健。L在同一S022扫描中改善Bladder却损害Rectum，说明一个全局权重调整不能被简化为所有小器官共同受益。S在特定病例修复漏分，但不足以抵消全患者的回退。

**尚未证明的机制：**上述图像确认预测错误的存在和形态，不证明这些误报由特定归一化统计、训练边界曝光或某一输出头造成。中心平面未显示的极端尾部也不能凭空解释为连通组件；本次没有进行新的全volume连通组件测量或临床专家判读。当前证据足以决定L/S不替换v3，不必继续扩大本轮诊断。

建议将下一轮优先级从继续整体压缩类别权重，转向明确的类别特异误报/漏分问题；任何新的采样、loss或后处理规则须先用Train/Validation证据定义，并与同期C配方匹配比较。目前仅提出方向，尚未冻结或启动新训练。

本地可直接打开：

- [S022膀胱成功与失败对照](../case_studies/validation/best_cls_review/S022_Week6_Bladder_three_planes.png)
- [K018 Body远处误报对照](../case_studies/validation/best_cls_review/K018_Week4_Body_three_planes.png)
- [K018前列腺漏分改善对照](../case_studies/validation/best_cls_review/K018_Week2_Prostate_three_planes.png)
- [J026六类对照](../case_studies/validation/best_cls_review/J026_Week6_six_classes.png)
- [病例指标明细](../comparisons/validation_analysis/tables/selected_case_evidence.csv)；[图像与复现校验回执](../comparisons/validation_analysis/visual_review_verification.json)。

## 9. 固定checkpoint的Test补齐（后续描述性评估）

作业682322正常完成（0:0，6分钟）。C50/L45/S45全部沿用已固定的Validation最佳权重，额外复现K008_Week0的Validation混淆矩阵后，在既有6患者30个Test volume进行评估。原始结果的SHA-256与服务器一致，输入文件哈希与旧七模型Test审计一致。此阶段没有重新训练、挑选轮次或调参。

| 模型 | Validation主要Dice | 既有Test主要Dice | Test前列腺Dice | Test远处Body FP | Test Body HD95中位数 mm |
|---|---:|---:|---:|---:|---:|
| 历史v3 | 0.85636 | 0.81448 | 0.74850 | 81,145 | 4.50 |
| C50 | 0.85424 | 0.80400 | **0.76327** | 33,341 | **3.36** |
| L45 | 0.84256 | 0.78578 | 0.73615 | 39,115 | 5.64 |
| S45 | 0.84649 | 0.77884 | 0.72810 | **5,919** | 4.70 |

这次Test描述没有推翻Validation判断：L/S的整体Dice仍低于同期C。S显著减少此队列的远处Body误报，但牺牲小器官表现，不能据此认定全面更好。C前列腺Dice高于历史v3，但整体Dice较低，进一步说明单一平均或单个类别都不能代表所有性能。

历史Standard的Test主要Dice为0.78510；L仅高约0.00068，不应解读为稳定提升。历史v3与C配方相同，Test主要Dice相差约0.01047，这种轨迹差异更应促使我们控制重复种子与患者不确定性，而非依据既有Test反复优化配方。

Test已被多次查看，只能作为描述性结果，不是外部验证或新盲测。后续优化依旧基于Train/Validation；当前不启动新模型。全部十模型最新图表见[统一对比图库](../../../reports/model_performance_comparison/README.md)。
