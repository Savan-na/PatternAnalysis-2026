# P2：远处误检与预测机制——现有证据

本页主体仅复算作业 677414 的四例固定权重记录。随后 P2B 作业 677509 在首例数值正控失败后停止，尚无合格的统计回放、六分类置信度与过滤结果；见 [失败记录与数值定位方案](P2B_numerical_failure_ZH.md)。没有新增训练。

## A. 推理等价性

原 G5 两模型共用配对降采样、整卷非零 Z-score、XYZ→DHW、高端补零、argmax 和反向裁剪；原流程没有滑窗/overlap/blending。两模型输出均与输入 padded grid 同大小，Improved 先将辅助头上采样再相加。P2 八组原整卷混淆矩阵已复现。没有找到已证实的共同错轴、错网格或漏裁剪缺陷；相同适配器不排除网络对上下文和 padding 的不同响应。

## B–D. 同体素 logits、几何与辅助头

以下 margin 在同一组 Improved 原整卷远处误检体素上计算，不能把图中的百分位范围当作置信区间。

| 病例 | Improved margin 中位数：整卷→滑窗 | Standard：整卷→滑窗 | Improved 整卷 winner margin 的 P95 |
|---|---|---|---|
| K018_Week4 | 1.277→-1.730 | -8.011→-8.192 | 4.235 |
| B040_Week4 | 0.848→-2.129 | -8.003→-8.393 | 2.889 |
| J026_Week4 | 1.406→-4.005 | -9.043→-9.033 | 4.430 |
| K008_Week0 | 0.584→-1.697 | -7.870→-8.303 | 2.188 |

错误中存在明显大于微小决策边界偏差的 margin，不能把全部误检解释成接近零的数值抖动。但现有记录没有全部六类概率或逐体素 logits，不能据此报告 softmax≥0.9 的比例或校准置信度。

| 模型 / 病例 | 整卷全部 Body FP 到体积边缘的中位距离（mm） | 整卷不接触 GT Body 的预测 Body 连通域数 |
|---|---:|---:|
| improved_3d / K018_Week4 | 20.156 | 364 |
| improved_3d / B040_Week4 | 16.797 | 343 |
| improved_3d / J026_Week4 | 31.200 | 226 |
| improved_3d / K008_Week0 | 21.840 | 238 |
| standard_3d / K018_Week4 | 60.469 | 8 |
| standard_3d / B040_Week4 | 53.040 | 13 |
| standard_3d / J026_Week4 | 43.680 | 10 |
| standard_3d / K008_Week0 | 37.440 | 4 |

此距离来自全部 Body FP，不限于 Background 或≥20 mm 子集；旧指标以外层体素中心为边缘，不能和新 P2B 的体素面距离直接混合。GT 不接触连通域是评估描述，不能用来定义推理时过滤规则。

三个 head 只对 logits 总和计算 loss；梯度传到所有 head 不代表每个 head 已独立学成分割器。quarter 在不同病例既有抵消也有助长 Body 偏差的表现。主头单独推理不是重训消融，禁止据此直接删除辅助监督。

## 根因证据排序

1. **高把握的现象**：当前权重与推理上下文存在交互；Improved 远处误检降低，但部分器官及 K018 HD95 退步，不采用滑窗替换。
2. **机制有证据，归因未闭合**：多头共适应与归一化范围均可能参与。现有整卷/窗口对照同时改变统计、感受野和边缘条件，不能单独证明 InstanceNorm 是原因。
3. **候选相关性**：G7 增加有内容角落曝光，尚无分割收益或新增曝光与真实错误的直接覆盖证据；loss 取舍和单 seed 也未排除。

## E. 下一项与后处理边界

已准备一次固定四例 P2B：相同窗口输入下仅回放该病例整卷 InstanceNorm 统计；先检查整卷自回放数值等价，保留参数/缓冲区哈希。并测六分类 softmax、物理空间分布和连通域范围。
另做一次无 GT 过滤：仅删除体积<100 mm³且小于最大预测 Body 连通域1%的六邻域分量，改为最优非 Body 类。它仍可能删除真组织，仅是预先固定的诊断，不是临床安全策略或默认模型。
当前不把 G7 视为最佳下一项训练的已定结论。先用统计回放区分 H4；按 P1 预设机制判据决定是否支持归一化因素，再决定单因素训练，禁止扫描窗口或过滤阈值。

## 查看与复现

[同体素 margin 图](same_voxel_logit_evidence.png) · [空间/logit 明细](spatial_logit_evidence.csv)
代表性 MRI：[K018 Body](../p2_four_cases_20261020/K018_Week4_improved_3d_body.png)、[B040 Body](../p2_four_cases_20261020/B040_Week4_improved_3d_body.png)、[J026 head](../p2_four_cases_20261020/J026_Week4_head_margins.png)。原图只展示已记录的病例，不从截图估算指标。

复现：`python -B code/tools/summarize_p2b_prediction_mechanisms.py --phase existing`；为防覆盖，已有输出目录时拒绝重建。
原始 manifest SHA-256：`a21a38f6bd2c312768c923138da25190ce4346b202dea9a1a61563c6d7acad68`。
