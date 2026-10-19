# HipMRI 模型复盘与下一轮实验决策

**日期：2026 年 10 月 19 日。** 本文是下一轮开发的中文决策备忘录。完整研究过程、方法及逐关证据保存在[正式研究报告](HipMRI_Research_Report.md)；这里集中回答一个问题：当前 Improved 3D U-Net 是否已经明显优于 Standard 3D U-Net，以及下一步怎样检验可行的改进。

## 1. 结论先明确

你的观察是对的：**现有 Improved 不能称为全面或明显胜过 Standard。** 两者均用 26 名 Train 患者训练 50 轮，以相同的 6 名 Validation 患者选择第 50 轮权重；只有这两个 3D 模型参加了既定的一次性 6 名患者 Test 评估。Improved 在部分小器官和 Bones 上获益，但在 Background、Body 边界及部分患者上明显退步。它使用约 **236.7 万**参数，Standard 约 **141.3 万**；平均每轮训练时间约 **145 秒**对 **104 秒**。额外复杂度尚未换来稳健的整体优势。

| 预先确定的患者等权小器官 Dice | Standard 3D | Improved 3D | Improved − Standard |
| --- | ---: | ---: | ---: |
| Validation，6 名患者 | 0.8502 | 0.8420 | **−0.0082** |
| 一次性 Test，6 名患者 | 0.7851 | 0.8078 | **+0.0227** |

这个分数先在每名患者内部合并所有周次的混淆矩阵，计算 Bladder、Rectum、Prostate Dice，再给六名患者相同权重。Validation 上 Improved 仅赢 **2/6** 名患者；Test 上赢 **3/6** 名。Test 的平均增益高度集中在 `M015`：该患者增加约 **0.1695**，其余五名患者的均值反而从 Standard 的 **0.8444** 降到 Improved 的 **0.8378**。后者只是敏感性描述，不能替代原定 Test 指标，更不能用来挑选新模型。六名独立患者太少，不支持“稳定领先”的结论。

## 2. 六类别与边界：它究竟赢在哪里、输在哪里

下表是**汇总体素 Dice**，与上面的患者等权选模分数定义不同；两种视角应一起看。数值来自两组第 50 轮[Validation 记录](../../logs/training/baseline_3d/epoch_050.json)及[一次性 Test JSON](../tables/comparisons/g6_test_3d.json)。[六类别 Validation 图](../figures/comparisons/g5_3d_validation_all_classes.png)与[六类别 Test 图](../figures/comparisons/g6_test_all_classes.png)均覆盖 Background、Body、Bones 和三个小器官。

| 类别 | Validation Standard → Improved | Test Standard → Improved | 判断 |
| --- | ---: | ---: | --- |
| Background | 0.9969 → 0.9636 | 0.9875 → 0.9619 | 明显退步 |
| Body | 0.9769 → 0.9274 | 0.9591 → 0.9312 | 明显退步 |
| Bones | 0.8596 → 0.8800 | 0.8569 → 0.8662 | 改善 |
| Bladder | 0.9436 → 0.9509 | 0.5192 → 0.7316 | 改善，Test 受 M015 影响大 |
| Rectum | 0.8038 → 0.7835 | 0.8063 → 0.8218 | 两组方向相反 |
| Prostate | 0.8179 → 0.7878 | 0.7180 → 0.7530 | 两组方向相反 |

**Body 错误不是少数几张难例造成的。** 两组固定第 50 轮权重在全部 38 个 Validation volume 上重新推理，均逐例复现原有混淆矩阵。Improved 的 Body HD95 中位数为 **76.61 mm**，Standard 为 **4.75 mm**；Improved 在 **38/38** 例均更差。真实 Background 被预测为 Body 的体素数为 **1,675,691 对 59,766**，其中距离真实 Body 至少 20 mm 的分别为 **1,358,516 对 1,007**。[六患者边界图](../figures/comparisons/g5_validation_body_boundaries.png)和[三例 MRI 对照图说明](../case_studies/comparisons/README.md)展示了这种远处误报。一次性 Test 也呈现相同方向：Body HD95 中位数约 **51.81 对 4.86 mm**。

Improved 有真实的局部价值，例如 Test 的 Bladder 和 Bones，以及部分患者的小器官召回；只是这些收益无法抵消严重的 Body/Background 退步。不能只挑三类器官或只看 Test 平均值宣称胜利。

## 3. 为什么没有达到预期：证据、解释与尚未证明之处

1. **训练目标与“全面胜过”目标没有对齐。** 两组使用相同的加权交叉熵与前景 Soft Dice；权重依据 Train patch 曝光设定，模型选择分数只平均三个小器官。它没有对 Body HD95、远处 Background→Body 误报设置约束。选择第 50 轮并无程序错误，但“选模分数较高”不等同于“六类及边界都更好”。相同损失下 Improved 的训练 loss 略低，也不能证明它泛化更好。
2. **不均衡仍存在，但“模型没见过小器官”不成立。** 当前 143 个 Train 原始 volume 的 Prostate、Rectum 体素占比分别约 **0.108%**、**0.145%**；完整 50 轮 3D 训练中，Prostate 出现在 **12,490/13,000** 个 patch、Rectum 出现在 **11,935/13,000** 个 patch，然而体素份额只有 **0.646%**、**0.752%**。两组 3D 模型收到完全相同的病例安排和类别曝光，因此这一共同数据问题不足以单独解释 Improved 特有的 Body 失误。[Train 类别统计](../data_analysis/g2_train_summary/class_distribution.csv)与[空间覆盖记录](../data_analysis/g4_patient_sampling/g5_train_patch_geometry.json)保留了分母和来源。
3. **体积边缘的训练覆盖稀少，值得检验，但尚非已证原因。** 在 13,000 个真实训练 crop 中，仅 **6 个（0.046%）**触及体积最外侧 2 体素的角落。大部分 patch 有 Background，并不表示见过足够多的整卷边缘。每个 patch 自身仍有卷积 padding，不能把“体积边缘覆盖少”简化成“模型没学过 padding”。Standard 接受相同 crop 却较少远处误报，说明还存在模型结构与采样的交互。
4. **Improved 的结构因素尚未分离。** 它同时增加残差块、InstanceNorm、Dropout、定位路径和主输出与两个辅助输出的相加。只用现有两臂结果，无法断言哪一项是根因。对固定权重进行只读诊断时，单独主输出的小器官分数仅 **0.2414**，Body HD95 **108.96 mm**，均劣于原相加输出的 **0.8420**、**76.61 mm**。主输出原本是作为总和的一部分训练的；这反对直接在推理时删去辅助输出，却不等于完成了重新训练的结构消融。[诊断 JSON](../tables/comparisons/g5_validation_aux_logits.json)记录了全部 38 例。
5. **患者数量小、单次随机种子造成结论不稳。** Validation 与 Test 各仅六名患者。对每组六个患者的 Improved−Standard 分数差，分别枚举有放回抽取六人的全部 **6⁶** 种组合，并取均值分布的 2.5% 与 97.5% 分位数：Validation 约 **[−0.0186, +0.0034]**，Test 约 **[−0.0170, +0.0852]**，都跨过零。这只是描述性重抽样区间，不是形式化显著性证明。`M015` 的 Test Bladder 异常说明单个患者能够强烈改变平均结果。

补充的 2D 模型已训练至 50 轮，Validation 小器官分数 **0.7757**，低于两组 3D；它的 Prostate 精确率 **0.7377** 高于 Standard 的 **0.7250** 与 Improved 的 **0.6745**，但召回率较低。2D 同时改变了空间上下文、采样、通道数和权重，不能把它的差异直接归因于“2D 方法更好”，也不能直接搬用其器官感知 sampler。它提示我们在新实验中必须同时检查**召回与误报**。

## 4. 下一轮：先验证采样假设，再谈新模型胜利

**实验 A：仅改 Improved 3D 的 Train patch 位置。** 保留患者划分、每轮病例、MRI/Mask 预处理、网络结构、损失、类别权重、优化器、随机种子、每轮 260 个 patch、50 轮预算及原验证选模方法。每名患者每轮保留 9 个原均匀 patch，把第 10 个改为针对体积外缘的 Background/Body 负样本；采样只使用 Train MRI/Mask，固定生成规则并记录来源。边缘 patch 的具体可用位置须先在 Rangpur CPU 上盘点，避免产生几乎只有零强度的无效样本。

**先做不训练的 143-volume、50-轮采样预演。** 与原 13,000 个 patch 对照，记录六类命中率与体素份额、每患者覆盖、外侧 2/8 体素面的覆盖、每 patch 有效 MRI 强度以及体素/仿射对齐。初步进入训练的门槛：外侧 8 体素角落覆盖达到至少 **5%**，三个小器官的命中率和体素份额均不低于原方案的 **90%**；这些是事先设定的工程门槛，不是统计定律。若达不到，修改候选 sampler 并再次预演，不消耗 GPU 训练。

**再做单因素 A100 对照。** 仅在通过预演后，以相同 50 轮和保存/续训规则从头训练 Improved。与现有 Improved 权重比较，主终点仍是患者等权小器官 Dice；必须同时报告六类 Dice、每患者差值、Body HD95、≥20 mm 的 Background→Body 误报、精确率/召回率及训练成本。第一道工程门槛是小器官分数不低于当前 **0.8420**，Body HD95 中位数从 **76.61 mm** 降到 **10 mm 以下**，远处误报至少减少 **90%**；若只修 Body 却损伤小器官，应认定为权衡而非整体改善。

**若想声称 Improved 架构本身明显优于 Standard，必须再让 Standard 使用同一新 sampler 训练。** 否则只能说“Improved 新训练配方优于旧 Standard 配方”，无法把收益归因于网络结构。建议至少比较多个随机种子，最终用按患者分组的交叉验证检验稳定性。一个严格的“明显胜出”目标可以预先定义为：患者等权小器官 Dice 比同条件 Standard 高至少 **0.03**，至少 **5/6** 名 Validation 患者获益，任一类别汇总 Dice 降幅不超过 **0.01**，Body HD95 不高于同条件 Standard；这些是研究目标，当前模型没有达到，也不保证下一轮达到。

若实验 A 提供足够边缘样本仍无法改善 Improved 的 Body 错误，下一项**独立**实验才是重新训练去掉辅助相加路径的 Improved 结构消融；不要同时改采样与网络。已有 Test 病例被查看过，后续不能反复用这 30 个 volume 选方案或把它们称为新模型的全新盲测。需要新的未见患者队列，才能对最终选定模型作新的独立 Test 结论。

## 5. 当前状态

本报告是**实验决策**，不是新实验结果。实验 A 已实现为独立的[候选采样模块](../../code/hipmri_rewrite/sampling_edges.py)、[CPU 预演程序](../../code/tools/audit_g7_edge_sampling.py)和[候选配置](../../code/config/g7_edge_sampling_candidate.json)，35 项本地测试通过。程序先用两名患者、两轮检查真实数据流程，再重放完整 50 轮；每个原均匀采样轮次必须复现训练记录中的类别曝光。完整输出还要满足第 4 节的覆盖与小器官门槛，才会标记为可进入训练。Rangpur 真实数据预演、新模型训练、多种子比较和新独立 Test 均尚未执行。操作入口在[远端命令说明](../../code/docs/RANGPUR_COMMANDS.md)。
