# P2B：远处误检机制诊断

## 状态与研究问题

现有记录的 [空间/logit 证据](../../results/diagnostics/inference_context/p2_mechanism_evidence_20261020/README.md) 已生成，原始记录和图像没有修改。新 P2B 程序及五项本地测试已准备；**真实统计回放、六分类置信度和过滤诊断尚未运行**。

本阶段不训练模型。先区分“推理范围敏感”与“归一化统计来源的作用”，再考虑 G7 采样或其他单因素训练，不预设哪种一定胜出。

## 已核对的推理与 Deep Supervision

两模型原 G5 均使用相同 MRI/标签加载、factor-two 配对物理网格、整卷非零 Z-score、XYZ→DHW、高端补零至 16 倍数、整卷前向、logits argmax、去 padding、还原 XYZ。没有原生滑窗、重叠或 blending；不能把新增滑窗的拼接行为当成原错误机制。

Improved 原生 main/half/quarter head 分别位于解码器全、半、四分之一空间分辨率；后两者经 trilinear、`align_corners=False` 上采样至 main 的形状，再逐类相加。训练 patch 为 DHW 32×64×64，因此原生辅助输出通常是 16×32×32、8×16×16；整卷的各级尺寸随输入改变。

loss 只作用于总和。在上采样后的 head logits 上，链式法则给出相同的总和梯度；原生低分辨率 head 还通过插值的反向算子接收梯度。这不等于各 head 有独立损失，也不保证独立主头可用。历史总梯度记录不足以比较各 head 的训练梯度大小。本阶段分析 head 的贡献，不把 main-only 当重训消融。

## 固定设计与判别标准

- 同 P2 四例 Validation，两组原第 50 轮权重；禁止 Test 读取与参数扫描。
- 所有原整卷及原生滑窗混淆矩阵必须复现 P2，输入文件哈希及 checkpoint 身份必须一致。
- Improved 原整卷一次前向捕获全部 14 个 InstanceNorm 的输入逐通道均值、**有偏方差**。在相同滑窗上，仅用这些统计量回放归一化，affine、epsilon、输入、head 融合和窗口融合保持不变。不启用未估计的 running statistics。
- 数值正控：整卷自身统计回放，所有 head 的最大 logit 差≤0.001，padded grid 的 argmax 不一致比例≤0.00001；不通过就保存失败记录并停止，不能解释机制。五项本地测试包括真实网络回放及点式网络上固定窗口的统计隔离检验。
- 对每例计算 `R=(回放远处误检−原生滑窗远处误检)/(原整卷远处误检−原生滑窗远处误检)`。至少三例 R≥0.5 且同一参考误检体素的 margin 中位数比原生滑窗更接近整卷，才支持统计来源有实质贡献。此条件不是显著性检验、临床阈值或最终选模规则。
- 回放可能引入分布外激活。即使阳性，也不能证明 InstanceNorm 普遍不适合分割；阴性不能排除其他上下文、padding 或输出融合效应。

## 新测量与后处理

1. 六类 softmax、Body−Background 和 Body−最大其他类 margin；正确 Background、原远处误检、当前远处误检和真实 Body 区域分别报告。两模型共享 Improved 原远处误检参考区域。
2. softmax≥0.9、winner margin 0–0.25 与>2 的计数仅作预设描述分箱，不用来调预测阈值；softmax 未做校准。数值近似造成的 argmax 近零 tie 单独报告，不误称推理缺陷。
3. 错误到 GT Body 的毫米距离、到原始 FOV 外面的距离、归一化 XYZ、世界坐标边界框、六邻域分量数/体积/范围。拒绝 shear 网格，源体积偶数尺寸保证当前 factor-two 网格保持原 FOV。
4. 滑窗测量到所有覆盖窗口的最近边面的距离及窗口—体素对的边缘比例；这些不是原 G5 的“拼接缝”。比较空间集中性时要考虑窗口覆盖本身的几何基率，不能只用靠近窗口边缘的计数证明 artifact。
   原始记录另存所有真实 Background 的对应空间分箱及窗口覆盖分母，并统计当前 Validation 错误与“外侧8体素、有非零 MRI、距 Body≥20 mm”区域的重叠。后者只检验 G7 候选定义的相关性，不用 Validation 坐标指导 Train 裁剪，也不等于已经证明采样收益。
5. 一项独立过滤诊断：仅移除 **<100 mm³且<最大预测 Body 分量1%** 的六邻域分量，重新分到最高 logit 的非 Body 类。函数不接收 GT。事后检查删掉多少真 Body、错 Body，以及六类和 HD95 的变化；不把它自动设为默认，不声称临床安全。

原 P2 的边缘距离以体素中心计，新 P2B 明确使用 FOV 外侧体素面。二者有定义差异，保持历史值，不能无说明合并。

## 输出与执行

| 内容 | 新路径 |
|---|---|
| 原始病例记录、统计向量、完整 manifest、校验和 | `logs/validation/inference_context/p2b_prediction_mechanisms_20261020/` |
| 回放/原生滑窗的 Body 与 head 图、过滤 FP/FN 图 | `results/diagnostics/inference_context/p2b_prediction_mechanisms_20261020/` |
| 当前已有记录的分析图与报告 | `results/diagnostics/inference_context/p2_mechanism_evidence_20261020/` |

传输包只含六个新增文件：模块、测试、配置、测量器、汇总器、提交脚本。一次 SCP 上传后，先核对包 SHA，检查六个固定相对文件名并拒绝覆盖，再解压。原冻结文件与旧 P2 helper 不在包中。

包名：`code/transfer/p2b_prediction_mechanisms_20261020.zip`；SHA-256：`cb761c04ccc2767abd77efe0882e1dfdfba3ec4621bbeb7fb92a45b42223ca6d`。

分别执行：

```bash
bash -n "$HOME/20261018_g1_stage/20261017/code/tools/submit_p2b_prediction_mechanisms.sh"
```

```bash
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_p2b_prediction_mechanisms.sh"
```

只提交一次十分钟 A100-test 作业；启动时再运行五项测试。输出已存在或同名作业在排队/运行时会拒绝再次提交。日志为 `logs/profiles/p2b_mechanisms_<jobid>.out`。只有 manifest 完整八组且 `complete=true` 才算完成；失败时先看日志和部分记录，不换阈值绕过正控。

下载两个结果目录、验证校验和后，在本机 `20261017` 根目录运行：

```bash
python -B code/tools/summarize_p2b_prediction_mechanisms.py --phase replay
```

它生成中文 P2B 简报和区域置信度 CSV。需要在有 NumPy/Matplotlib 的环境运行。后续结论依据实际结果更新，当前不能宣称统计回放或后处理已经有效。
