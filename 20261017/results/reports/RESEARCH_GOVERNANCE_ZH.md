# 研究治理与实验设计指令

以下保留用户原始指令与早期审查材料。旧阶段编号属于来源内容，当前组织与执行以项目入口和冻结协议为准。

# P1：HipMRI 研究与课程交付独立审查

日期：2026 年 10 月 20 日。按新提供的 Master Prompt/P1 执行；本轮核对既有
源码、配置、日志、G7 CSV 和课程原文，没有训练新模型或用 Test 选择新方案。
既有性能复盘见[研究报告](RESEARCH_REPORT.md)，本页只记录
本次审查新增的要求映射、竞争解释和路线修正。

## 1. 课程约束已核对，交付仍有明确缺口

依据本地[课程 PDF](../../../../COMP3710_Report_2026_draft_v2.2.pdf)，封面标题为
Version 2.0 Final Assessment Specification (Draft)，文件名为 draft_v2.2；SHA-256
为 `0e05eab1e3f7597424a235e84db778106875ffa4419f0702c803ee2b052954ea`。
已核对正文第 2、5–6、11–16 页，并查看第 5–6、12–15 页渲染图。
本地两份副本内容相同；尚无可核验的更新公告、具体截止日或导师 check-off 记录。
附件对草案的转述不能替代这些待确认信息。

| 原文要求 | 当前证据 | 剩余事项 |
| --- | --- | --- |
| 第 6 页：Hard 3D Improved U-Net，降采样 3D HipMRI、体积残差单元、Deep Supervision | 当前模型含残差块及三个输出头相加，梯度传播到全部头 | 最终候选继续保留残差与有意义的 Deep Supervision；结构变体须说明监督方式 |
| 第 5 页：同一评估划分的基线比较、逐类 Dice/IoU、GPU 显存及推理耗时 | Standard/Improved 同方案 50 轮、同 Validation/Test；已有六类指标 | 2D 尚无同口径完整边界比较；需统一资源测量方式 |
| 第 5、14 页：3–5 例失败剖析，关注 base/apex、界面与切片过渡 | 已有多平面 Body、Rectum、Prostate 图，3D Test 的 base/apex 指标 | 补充 Validation 上三模型同病例的 base/apex、相邻器官界面和连续性证据 |
| 第 11–12 页：一页 feasibility review、导师中期 check-off 必过 | 已有数据审计与可行性资源证据 | 活动项目未找到正式一页 review；实际 check-off 完成情况未知 |
| 第 12–13 页：recognition/ 下含 modules.py、dataset.py、train.py、predict.py、README.md | 当前 Git 根目录只有 20261017 等研究目录；有效实现分散于 code/ 包和 tools/ | 活动树缺五文件形式的提交入口和 recognition/ 目录；最后用薄入口打包复用现有实现，现阶段不搬动训练代码 |
| 第 12 页：模型组件限 PyTorch/TF 内部依赖 | 当前三个网络及共用块仅依赖 PyTorch 与类别常量 | 最终 modules.py 继续遵守；数据/可视化工具的 NumPy 不等于模型依赖违规 |
| 第 14–15 页：技术 README、精确依赖、AI Disclosure、PR、Turnitin PDF | 已有渐进 Git 提交、正式报告及环境记录，分支为 topic-recognition | 项目入口 README 尚非完整提交报告，缺正式 AI Disclosure；PR/Turnitin 完成状态未核实 |
| 第 13、16 页：不提交数据和模型权重 | git ls-files 未列出 NIfTI、pt/pth 或传输 ZIP | 最终提交时继续检查；现有源记录和可视化要给出来源 |

第 6 页写了 Hard 模型 Test Dice ≥0.7，但没有在该表中完整规定六类、前景、患者
或汇总体素的聚合方式。不能直接用现有“三小器官患者等权分数”证明这个目标的
准确含义。原文强调研究深度与工程权衡，没有要求 Improved 必须全面击败基线。

## 2. 源码复核带来的主要新判断

**当前没有 sliding-window 推理。** `inference.py` 将降采样整卷转成 DHW，仅在
高端补齐到 16 的倍数，一次前向后去掉 padding 并恢复 XYZ；两组 3D 共用此路径。
因此“patch 拼接缝造成现有整卷误报”不符合当前实现。配对重采样和预测轴还原有
测试覆盖，38 个 Validation 的原始混淆矩阵也已逐例复现；目前未发现已证实的
错轴、不同标签网格或忘记去 padding 的缺陷。

**训练输入与验证输入的空间范围不同。** 训练是 64×64×32 XYZ patch，验证通常
是 128×128×64 整卷（个别 Z 更大）。Improved 的 14 层 `InstanceNorm3d` 没有保存运行
统计量，eval 仍按当前输入、逐通道空间范围计算统计；Standard 的 GroupNorm(1,C)
也依赖当前输入，但在通道与空间上的聚合方式不同。两者还有下采样、上采样、
残差和输出融合差异。这构成一个比“角落曝光少就是根因”更具体的**推理上下文
与归一化交互假设**，尚不能据此认定 InstanceNorm 有 bug 或必然导致误报。

**相加输出不是三个独立训练好的分割器。** 当前只对主头与两个辅助 logits 的
总和计算 CE/Dice，三个头通过总和获得梯度，没有各自独立损失。主头单独推理
变差，说明它不适合直接作为现有权重的替代输出；既不能证明辅助头无害，也不能
把单头结果当作重新训练的消融。需要按 Body−Background 的 logit margin 分解
三头贡献，观察同一区域是微小偏差、强错误置信度还是头之间的抵消。

**训练与选模指标存在已知取舍。** 共同目标为加权 CE 与前景 Dice 的混合，选模只
看三个小器官患者等权 Dice，未约束 Body HD95。它能容忍边界退步，不是选模代码
计算错误。2D 与 3D 还改变了采样、通道宽度和权重，不能把全部差异归因于维度。
同患者周次先合并再评分的实现合理；独立样本量仍是六名患者。现有 HD95 是双向
表面距离合并后的 95 分位数，报告应保留此定义及降采样分辨率限制。

**计算成本数据需更谨慎。** 一次性 G6 程序先将两个模型同时载入 GPU，再分组
推理；峰值显存包含另一模型的常驻参数，且没有专门的统一 warm-up/重复计时段。
已有时间与显存是该次程序的真实测量，但不能当作严格隔离的三模型基准。后续
共同资源比较应逐模型加载、统一 warm-up、同步与重复次数，并分别报告预处理、
推理和 checkpoint 写入时间；无须为此重新训练。

## 3. G7 已完成什么，尚未证明什么

全量 CPU 作业 `676249` 用时 4 分 46 秒。两个下载文件与远端 SHA 一致；本机
从 CSV 重新核对 26,000 行、11,700 对保留 crop、全部 50 轮原训练曝光和六类总量。
全部 1,300 次替换均找到合格角落，三个预设门槛通过，详见
[G7 图表及表格](../data_analysis/boundary_sampling_audit/README.md)。

| 测量 | 原均匀采样 → 9+1 候选 |
| --- | --- |
| 外侧 8 体素角落覆盖 | 3.25% → 12.95% |
| 外侧 2 体素角落覆盖 | 0.046% → 0.262% |
| Bladder/Rectum/Prostate 体素曝光保留率 | 95.20% / 94.91% / 95.80% |
| 三个小器官命中率保留率 | 98.12% / 97.71% / 99.49% |
| 替换 patch 非零 MRI 份额，最小/中位数 | 46.17% / 85.38% |
| 每替换 patch 的外侧远处非零背景体素，最小/中位数 | 440 / 8,969.5 |

原 5% 非零阈值相当宽松，但实际 crop 没有靠接近空白样本通过检查。最差单患者
小器官体素保留率也约 92.6%，没有被总体均值掩盖的严重曝光崩塌。不过这种
“hard negative”由标签、距离、非零强度定义，尚未证明恰好覆盖模型的真实错误，
也不等同于紧邻器官界面的负样本。位置多样性和曝光增加不能替代模型性能证据。
`eligible_for_training=true` 的含义仅是通过这个候选的工程门槛。

## 4. 竞争假设与优先级

这里按预期信息收益、计算成本和课程价值排序，不代表排第一的机制已经得到因果证明。

| 优先级 | 假设及现有支持 | 反证或限制 | 最小有效检验 |
| --- | --- | --- | --- |
| 1 | H4：训练 patch 与整卷推理的上下文/归一化/边缘交互；源码明确存在输入范围变化 | Standard 同样改变范围且表现较好，不能单独归罪于 padding 或 InstanceNorm | 固定两模型权重，对相同 Validation 病例做整卷与固定训练大小滑窗对照 |
| 2 | H1：体积角落曝光不足；G7 已证实可增加有内容负样本并保住器官曝光 | 两模型原曝光相同；远处错误不一定都在角落 | 若低成本推理诊断未找到更强解释，再做 Improved 原采样/9+1 的单因素训练 |
| 3 | H2：辅助融合/主头共适应；主头单独推理很差 | 未训练独立主头，现有结果不能归因某个 head | 同一区域的三个 head margin、总和 margin 和类别变化；有依据后做保留 Deep Supervision 的训练消融 |
| 4 | H3：权重和 Dice 带来召回/精确率取舍 | 相同目标的 Standard Body 很好，不能解释全部架构差距 | 在共同推理/采样控制建立后，再单独改 loss |
| 持续限制 | H5：患者差异与随机种子；只有一次种子且六名 Validation 患者 | 影响结论稳定性，但不是 38 例 Body 空间误报的已证机制 | 对有价值的最终比较增加种子或按患者交叉验证，而非先盲目扩大所有训练 |

## 5. 对原路线的具体修正

下一项先做 **P2 固定权重诊断**：在既有三例高 Body HD95、不同 Validation 患者
`K018_Week4`、`B040_Week4`、`J026_Week4` 外，加入较轻错误患者 `K008_Week0`
作对照。整卷结果必须先复现 G5；滑窗固定 64×64×32 XYZ、50% overlap
及与 GT 无关的融合规则。对比两模型在相同体素上的 Body/Background margin、
错误到原卷边界/GT Body 的物理距离及分离连通域，并分析 Improved 三头 margin。
GT 只用于事后评估，不能生成推理时 ROI。预计先用一个短 A100 作业，不做训练；
这四例只能定位机制，有明确变化后再扩至全部 38 例，不能从四例选最终模型。

若固定推理上下文能够大幅减少 Improved 远处误报且保住小器官，而 Standard 变化
较小，将增强 H4；它仍同时改变上下文、padding 和归一化范围，不是对 InstanceNorm
的单独因果证明。若效果很弱或造成损伤，则削弱“简单匹配推理即可解决”的解释，
转向已通过 CPU 审计的 G7 单因素训练，而不是继续试滑窗参数。若推理协议改变，
两臂都要得到新的独立标识；保留全部原指标，不能把新旧推理混为采样收益。

**撤回“去掉所有辅助监督作为最终 Improved”这一后备提议。** 后续如需要结构消融，
优先比较“辅助头各有训练损失，主分辨率 head 输出推理 logits”与现有相加方案，
保留体积残差和 Deep Supervision。完全删除 Deep Supervision 的网络至多是明确
标记的机制消融，不应直接当作课程 Hard 最终候选。

G7 方案仍保留为可训练候选，而非必做路线。本轮不因为 sampler 已实现就直接
开启 50 轮训练。阶段 P6 的 base/apex、三维连续性及同条件成本分析应与有限模型
优化并行；P7 需优先补齐提交入口、feasibility 与 AI Disclosure。已有 Test 用于
历史描述，下一轮选择只依据 Train/Validation，不能再称为新的盲测。

## 6. P2 落地状态（2026 年 10 月 20 日）

英文 [feasibility review](<../../../feasibility review.md>) 已放在仓库最外层。
[P2 执行说明](../../code/docs/INFERENCE_CONTEXT_DIAGNOSTIC_ZH.md) 对应本节前述四例、固定权重和固定滑窗方案。
P2 作业 `677414` 已完成，八组整卷混淆矩阵精确复现，原始记录与图片已导回并验证。
[P2 中文结果](../diagnostics/inference_context/context_sensitivity_20261020/CONTEXT_SENSITIVITY_REVIEW_ZH.md)：Improved 四例远处 Body 误检合计下降 80.41%，但 K018 膀胱 Dice 下降约 0.149、Body HD95 升高，K008 前列腺也受损。因此不把滑窗直接替换为新协议，不扫描参数或扩展全部 38 例；按本报告原定关口转向 G7 采样单因素实验。H4 的敏感性得到支持，尚未证明 InstanceNorm 单独导致错误。

## 规划和codex指令
# COMP3710 HipMRI — 模型优化与 Final Assessment 系统性 Prompt Series

基于课程官方 Assessment Specification（Draft v2.2）、2026 年 10 月 19 日 HipMRI 实验结果及当前模型失效分析

我建议将下一阶段组织成 1 个 Master Prompt + 7 个阶段性 Prompts，交给 Codex 或其他具备代码执行能力的 AI，逐阶段分析、实施和验证。

最重要的是：这些 Prompts 不应强迫另一个 AI 按照我们现有的假设执行，而应该给予它独立分析、质疑实验设计、提出替代方案和修改实现的权力。

但是，AI 可以改变研究方法，不能随意改变课程选题、伪造实验结果、破坏原始数据或利用 Test 集选择模型。

## 一、先确认我们真正要完成的课程任务

根据你上传的 COMP3710 Report — Version 2.0 Final Assessment Specification (Draft)：

| 项目                 | 已选择的方向                                                   |
| ------------------ | -------------------------------------------------------- |
| Assessment         | COMP3710 Final Assessment，40 marks                       |
| Research Domain    | Medical Image Segmentation                               |
| Dataset            | HipMRI Study                                             |
| Difficulty         | Hard                                                     |
| Final Architecture | 3D Improved U-Net                                        |
| 核心机制               | 3D Volumetric Processing、Residual Units、Deep Supervision |
| 主要研究问题             | Spatial Context vs. Clinical Boundary Utility            |
| 实验比较               | 2D U-Net、Standard 3D U-Net、Improved 3D U-Net             |
| 实际计算环境             | UQ Rangpur A100                                          |
| 最终目标               | 工作模型、受控比较、失效分析、工程建议和正式 GitHub 交付                         |

课程文件第 5–6 页特别强调：研究不能只看全局 Dice，还需要分析 Prostate base/apex、与 Bladder/Rectum 相邻的危险边界、3D slice-transition errors，并比较空间上下文带来的收益和 GPU 计算成本。

第 12–15 页规定了五个核心交付文件、README 研究分析、渐进式 Git commits、GitHub Pull Request 和 AI 使用披露。

因此，我们应该同时推进两个目标：

研究目标： 找到当前 Improved 3D U-Net 的主要失效机制，通过可验证的实验改善分割精度、边界可靠性及泛化。

课程目标： 用真实证据回答老师的 Engineering Dilemma，完成符合 Hard Difficulty 的工作原型和完整交付，而不是为了追求单个最高 Dice 耗尽时间。

## 二、整个 Prompt 系列的执行结构

Master Prompt — Research Governance

课程约束、实验边界、AI 自主决策权

P1 — Independent Research Review

独立审查已有结论，重新排序问题与假设

P2 — Root-Cause Diagnostics

Logits、Body False Positives、Inference、Deep Supervision

P3 — Sampling Audit & Redesign

Train Patch、Hard Negatives、Spatial Coverage

P4 — Controlled Training Experiment

A100 训练、Validation、资源记录、结果比较

P5 — Evidence-Based Next Experiment

采样匹配的 Baseline / Deep Supervision 消融 / Loss 优化

P6 — Clinical Boundary & Generalisation Study

Base/Apex、三维连续性、难例、统计与成本

P7 — Final Assessment Delivery

README、Reproducibility、Git、PR、最终结论

建议按照编号依次使用。每一阶段都允许 AI 对原始方案提出修改，但重要研究假设应先经过证据检验，不能在同一实验中不加区分地同时改变采样、损失、模型架构和训练策略。

下一部分是可以直接复制给 Codex 的英文 Prompts。

## 三、Master Prompt：给另一个 AI 的长期研究规则

使用方式：在新的 Codex 对话中首先发送一次，之后再执行 P1–P7。不需要每次重复。

# COMP3710 HipMRI — Research Governance and Autonomous Engineering Instructions

## Role

Act as an independent senior machine-learning research engineer, experimental scientist, and code reviewer.

Your task is to improve and critically evaluate an existing HipMRI 3D medical segmentation project for the UQ COMP3710 Final Assessment.

You are not required to agree with the previous AI's diagnosis, proposed experiments, implementation details, or priorities.

**You are explicitly authorised to challenge the existing plan, identify overlooked causes, propose better alternatives, and modify non-frozen research code when justified by evidence.**

However, preserve scientific validity, reproducibility, academic integrity, and the official assessment requirements.

## Source hierarchy

Read and distinguish:

1. The supplied COMP3710 Final Assessment Specification, Version 2.0 Draft, especially Sections 2.2, 3, 4, 5, and 6.
2. The latest official course announcements or clarified requirements, if available.
3. The existing HipMRI research report, experiment records, source code, configurations, and model checkpoints.
4. The October 19, 2026 research decision memorandum and supplied visual case studies.
5. Prior AI recommendations, which are hypotheses rather than authoritative conclusions.

If the draft assessment conflicts with a confirmed newer official requirement, identify the conflict.

Do not invent missing source content or experimental evidence.

## Fixed project scope

- Domain: Medical image segmentation for prostate radiotherapy planning.
- Dataset: HipMRI Study.
- Final difficulty: Hard.
- Required final architecture family: 3D Improved U-Net with volumetric residual units and deep supervision, unless a different Hard solution receives appropriate course approval.
- Framework: PyTorch.
- Existing patient-level split: 26 Train / 6 Validation / 6 Test patients.
- Existing model families: 2D U-Net, Standard 3D U-Net, and Improved 3D U-Net.
- Primary compute environment: UQ Rangpur A100 via Slurm.

Maintain the course's central engineering dilemma:

**Does advanced spatial context improve clinically important anatomical boundaries and slice-to-slice consistency sufficiently to justify the extra computational cost?**

Do not reduce the project to maximising a single Dice score.

## Scientific constraints

Preserve the existing patient split, raw data, historical checkpoints, original experiment records, and previously reported results.

Never use Test labels, Test-derived metrics, or previously inspected Test examples to select architectures, thresholds, checkpoints, hyperparameters, or experimental branches.

The existing Test evaluation has already been inspected. Future model selection must rely on Train and Validation. Any new claim of independent final generalisation requires a genuinely unseen patient cohort.

Preserve clear distinctions among:

- Observed evidence
- Plausible interpretation
- Unverified hypothesis
- Proposed intervention
- Experimentally supported conclusion

Control major experimental factors whenever causal attribution is the objective.

Do not claim clinical safety, clinical deployment readiness, or radiation-treatment benefit based solely on segmentation metrics.

## Freedom to improve the research plan

You may independently:

- Reprioritise research hypotheses.
- Design better diagnostic experiments.
- Revise candidate patch-sampling strategies.
- Propose alternative loss formulations.
- Investigate normalization, padding, sliding-window inference, augmentation, and deep-supervision mechanisms.
- Design controlled structural ablations.
- Improve evaluation metrics and visualisations.
- Recommend stopping an unproductive experiment.
- Reject a proposed change when the evidence does not support it.

You do not need to preserve previous recommendations merely because they were documented.

Any major modification must have an explicit research rationale, an identified control, an expected observable effect, and a falsification criterion.

## Execution discipline

Avoid unnecessary refactoring, repeated provenance audits, infrastructure expansion, and documentation churn.

Prefer the smallest experiment capable of distinguishing competing explanations.

Prioritise:

1. Existing evidence and low-cost diagnosis.
2. CPU-based checks or sampling simulation.
3. Controlled GPU experiments.
4. Independent interpretation.
5. Final assessment delivery.

A prompt explicitly authorising a training experiment permits that experiment within its stated scope. Do not silently launch unrelated GPU experiments or overwrite existing runs.

Preserve all historical experiment results and record new work under a clearly distinguishable experiment identity.

For each completed stage, report:

- What you inspected or changed.
- Why you chose that approach.
- Which alternative hypotheses were considered.
- Tests and experiments actually executed.
- Quantitative evidence.
- What the results establish and do not establish.
- The next recommended action.
- Any justified deviation from the original plan.

## Final principle

Optimise for **scientific credibility, useful model improvements, reproducible experiments, and COMP3710 assessment quality**.

A well-supported negative result is preferable to an unsupported claim of improvement.

Do not assume the Improved model is superior merely because it has more parameters or a more complex architecture.

## P1 — 独立复盘与研究方向审查

Independent Research Review · 不启动训练，优先读取已有证据

为什么第一步不是直接改代码？

因为虽然我们现在倾向于认为 Edge Sampling 不足与 Deep Supervision 有关，但此前还没有通过受控实验确定这两个原因。我们应该先让另一个 AI 自主检查研究链条，判断有没有更基础的实现问题。

# P1 — Independent Review of the HipMRI Segmentation Research

Read and apply the previously provided Research Governance instructions.

## Objective

Conduct an independent, evidence-driven review of the current HipMRI segmentation project before proposing or training another model.

Your role is to identify which problems are established, which mechanisms remain unproven, and which next experiment has the highest expected research value.

## Current context

The project has:

- 26 Train, 6 Validation, and 6 previously evaluated Test patients.
- 143 Train, 38 Validation, and 30 Test MRI volumes.
- A completed 50-epoch Standard 3D U-Net.
- A completed 50-epoch Improved 3D U-Net.
- A supplementary 50-epoch 2D U-Net.

The selected Standard and Improved 3D checkpoints are both from epoch 50.

Reported patient-equal small-organ Dice:

| Evaluation    | Standard 3D | Improved 3D |
| ------------- | ----------- | ----------- |
| Validation    | 0.8502      | 0.8420      |
| Existing Test | 0.7851      | 0.8078      |

Important observed failure:

- Validation Body HD95 median: 4.75 mm Standard vs 76.61 mm Improved.
- Improved has worse Body HD95 in all 38 Validation volumes.
- Distant Background-to-Body false positives are dramatically more frequent in Improved.
- Some smaller organs benefit from increased recall, but this can be accompanied by reduced precision.
- The Test average is substantially influenced by patient M015.

Treat these as reported observations to verify against available records, not as explanations of their causes.

## Tasks

### 1. Review assessment alignment

Read the official segmentation project specification and identify the required evidence for:

- Hard Difficulty implementation.
- 2D vs 3D spatial context.
- Boundary precision and anatomical error analysis.
- Model benchmarking.
- GPU memory and runtime.
- Documentation and reproducibility.

Distinguish mandatory assessment requirements from optional scientific extensions.

### 2. Review existing experiments

Inspect available code, training configurations, evaluation logic, experiment logs, and research reports.

Check for weaknesses in:

- Data preprocessing and spatial alignment.
- Training/Validation sampling.
- Loss construction and class weighting.
- Output-head fusion.
- Checkpoint selection.
- Sliding-window inference.
- Evaluation aggregation.
- Repeated-patient dependence.
- Computational fairness.

Do not assume that a lower training loss implies better segmentation quality.

### 3. Challenge the current explanation

Critically assess these hypotheses:

H1: Insufficient training coverage of full-volume boundaries causes distant Body false positives.

H2: Summation of deep-supervision logits contributes to overprediction.

H3: Class weighting or the loss formulation creates an unfavourable precision-recall trade-off.

H4: Padding, normalization, preprocessing, or inference produces systematic spatial artifacts.

H5: Some performance differences are driven by small sample size, patient variability, or optimization randomness.

You may introduce additional hypotheses if there is concrete justification.

### 4. Propose the next research strategy

Rank hypotheses by:

- Strength of existing evidence.
- Potential impact on the assessment's clinical-boundary dilemma.
- Ability to distinguish causes experimentally.
- Expected computation cost.
- Risk of damaging existing segmentation performance.

Recommend a minimal, coherent experiment sequence.

You are allowed to revise or reject the previously proposed sequence if you provide a stronger technical argument.

## Restrictions

This stage is primarily read-only.

Do not train models, change historical checkpoints, restructure the project, or inspect Test data for tuning.

## Deliverable

Produce a concise research review containing:

1. Assessment requirement mapping.
2. Current model performance and limitations.
3. Root-cause hypothesis ranking.
4. Confounding factors or possible implementation errors.
5. Recommended next experiment and its falsification criteria.
6. Proposed changes to the existing research plan.

Clearly state which conclusions are supported and which remain speculative.

## P2 — 深入研究 Body False Positives 的根本原因

Mechanism Diagnostics · 重点使用现有 Checkpoints，不进行新模型训练

这一阶段是当前研究最值得补充的工作。

它不预先认定采样就是错误来源，而是要求 AI 区分：

- 网络学到的错误类别偏好；
- 辅助输出融合造成的类别变化；
- 推理或空间处理机制产生的人工误差。

# P2 — Investigate Distant False Positives and Model Prediction Mechanisms

Apply the Master Prompt and incorporate any justified corrections from P1.

## Research question

Why does the existing Improved 3D U-Net produce substantially more distant Background-to-Body false positives than the Standard 3D U-Net?

Determine whether the problem is associated with data exposure, model output behaviour, inference implementation, or their interaction.

Do not assume the cause in advance.

## Available evidence

The existing Validation results show:

- Body Dice: Standard 0.9769, Improved 0.9274.
- Median Body HD95: Standard 4.75 mm, Improved 76.61 mm.
- Background-to-Body predictions at least 20 mm from the true Body: 1,007 Standard and 1,358,516 Improved.
- The Improved model has worse Body HD95 in 38/38 Validation volumes.

Existing visual cases include B040_Week4, J026_Week4, K018_Week4, and other documented patients.

Use the available original evaluation records rather than estimating metrics from screenshots.

## Investigation

### A. Verify inference equivalence

Inspect whether both 3D models use equivalent:

- Volume preprocessing.
- Spatial orientation and resampling.
- Sliding-window strategy.
- Patch overlap and blending.
- Padding and cropping.
- Output resizing.
- Softmax/argmax implementation.
- Reconstruction to evaluation space.

Check for architectural interactions without presuming an inference bug.

### B. Analyse class logits

Using existing checkpoints and Validation data, investigate:

- Body versus Background logit margins.
- Prediction confidence in correct Background regions.
- Confidence and margins in distant false-positive regions.
- Spatial concentration of errors.
- Differences between Standard and Improved.

Separate small logit-margin mistakes from confidently incorrect predictions.

### C. Analyse spatial geometry

Measure false positives by:

- Distance from true Body.
- Distance from original volume boundaries.
- Position relative to the full 3D image.
- Patch edges or sliding-window boundaries where relevant.
- Connected-component size and spatial extent.

Use physical millimetres derived from the appropriate geometry.

### D. Analyse deep supervision

Inspect the existing Improved model's main and auxiliary outputs, their spatial resolutions, fusion rule, and training gradients.

Identify whether particular heads contribute disproportionately to distant Body predictions.

Do not treat main-head-only inference on a model trained with summed outputs as a valid retrained architecture comparison.

### E. Consider low-cost post-processing

If justified, run a clearly separated Validation-only diagnostic of conservative 3D connected-component filtering or other anatomy-independent filtering.

Do not use ground-truth masks to define inference-time regions.

Do not make post-processing the default model simply because it improves selected examples.

## Methodological freedom

You may write small isolated diagnostic scripts and tests.

You may replace inefficient diagnostic methods with more informative ones.

If you discover a genuine implementation defect, provide the minimal correction and establish its effect through controlled evaluation before attributing differences to model architecture.

Do not overwrite original weights, evaluation JSON files, or historical metrics.

Do not retrain models in this stage.

Do not tune using Test data.

## Deliverable

Provide:

- Root-cause evidence ranked by confidence.
- Quantitative spatial/logit diagnostics.
- Representative Validation visualisations.
- Inference correctness findings.
- Deep-supervision mechanism findings.
- Any safe, reproducible diagnostic code.
- A recommendation on whether edge-focused sampling remains the best next experiment.

Explicitly distinguish correlation from causal evidence.

## P3 — 审查并优化现有 Edge Sampling 方案

Sampler Design · 允许 AI 修改候选实现，不允许修改冻结的数据划分

我们已经建立 G7 Candidate Sampler，因此不需要让 AI 从零写一个完全不同的采样系统。

但是要给予它调整权力，尤其是解决一个实际问题：采到体积边缘不代表采到了有价值的 Hard Negatives。

# P3 — Evaluate and Improve the Candidate 3D Patch Sampling Strategy

Apply the Master Prompt and relevant findings from P1–P2.

## Objective

Determine whether improved training exposure to informative Background and Body boundary regions can reduce distant Body false positives without damaging small-organ segmentation.

Evaluate and, if justified, redesign the existing candidate sampling strategy before spending GPU resources.

## Existing implementation

The project already includes a candidate edge-sampling module, CPU audit script, and configuration, provisionally identified as:

- sampling_edges.py
- audit_g7_edge_sampling.py
- g7_edge_sampling_candidate.json

The candidate replaces one of ten uniform patches per patient per epoch with an edge-focused negative patch.

Local synthetic tests previously passed, but full real-data Rangpur sampling validation has not yet been completed.

Inspect the actual current project tree rather than assuming these paths remain unchanged.

## Tasks

### 1. Review the sampler independently

Evaluate whether the candidate samples genuinely informative negative examples.

Pay particular attention to:

- Completely empty or near-zero patches.
- Background adjacent to anatomical boundaries.
- Distant but nontrivial Background texture.
- Original-volume faces, corners, and padding.
- Patient-level and week-level coverage.
- Spatial diversity across the dataset.
- Small-organ exposure preservation.

Challenge the assumption that increasing corner coverage alone will improve full-volume segmentation.

### 2. Establish the original sampling distribution

Using Train data only, reconstruct or verify:

- Patch counts per patient and epoch.
- Six-class voxel fractions.
- Organ hit rates.
- Original-volume boundary coverage.
- Spatial crop-centre distribution.
- MRI intensity and content characteristics.

Preserve the distinction between raw-volume class imbalance and post-sampling exposure.

### 3. Evaluate alternative candidate strategies

You may consider:

- Uniform plus volume-edge sampling.
- Uniform plus Body-boundary negatives.
- Hard-negative regions defined by existing Train-only evidence.
- Stratified background sampling.
- A balanced mixture of spatial categories.

You are free to recommend keeping the current 9+1 sampler or changing it.

Choose the simplest strategy whose expected advantage is supported by the available evidence.

### 4. Perform CPU preflight and full sampling simulation

Begin with the existing small real-data smoke test.

Then evaluate the complete planned sampling schedule using Train data only.

The earlier engineering proposal used:

- A minimum 5% coverage target for specified outer-corner regions.
- At least 90% preservation of original small-organ hit rates.
- At least 90% preservation of original small-organ voxel exposure.

Treat these as provisional engineering criteria, not scientific laws. You may revise them before training, provided you explain the rationale and do not select thresholds by examining Test results.

### 5. Choose the candidate for training

Compare candidate samplers on:

- Negative-sample informativeness.
- Spatial coverage.
- Organ exposure.
- Reproducibility.
- Runtime and complexity.
- Risk of introducing sampling bias.

Do not optimise only one coverage metric.

## Modification authority

You may modify candidate sampling code, configuration, and dedicated tests.

Do not overwrite the original sampler, historical training protocol, or patient split.

Preserve a reproducible control configuration.

Avoid unnecessary new frameworks or directory restructuring.

## Deliverable

Report:

1. Original versus candidate sampling statistics.
2. Spatial sampling visualisations.
3. The strongest and weakest aspects of the candidate.
4. Any code modifications and their rationale.
5. Tests executed and actual results.
6. A clear recommendation: proceed, revise, or reject.
7. The precise controlled variable for the next training experiment.

Do not initiate GPU training in this phase.

## P4 — 开始真正的受控模型训练

Controlled A100 Experiment · 允许执行 GPU 训练，但必须保留可比较的实验条件

这一阶段才真正消耗 GPU 资源。前提是前面分析支持采样实验，并且新 sampler 已经通过基本检查。

# P4 — Controlled A100 Experiment for the Improved 3D U-Net

Apply the Master Prompt and use the findings from P1–P3.

## Objective

Conduct a controlled training experiment to determine whether the selected new patch-sampling strategy improves the existing Improved 3D U-Net.

The experiment must distinguish improvements due to sampling from changes caused by architecture, loss, evaluation, or training budget.

## Experimental question

Can a modified Train patch distribution reduce distant Background-to-Body false positives while maintaining or improving small-organ segmentation?

## Pre-training decision

First review the diagnostic evidence and sampler preflight results.

If the evidence supports the sampling hypothesis and the candidate is technically valid, proceed with the controlled experiment.

If a confirmed inference defect or an invalid sampler makes the comparison meaningless, correct the problem and establish a valid control first.

Do not spend a full GPU training budget merely because a sampler has already been implemented.

## Controlled design

Use the current Improved 3D model as the reference.

Unless an identified implementation defect requires a documented protocol correction, preserve:

- Patient-level Train/Validation split.
- MRI and mask preprocessing.
- Model architecture.
- Loss function and class weights.
- Optimizer and learning-rate schedule.
- Random seed policy.
- Number of training epochs.
- Number of patches per epoch.
- Validation metric implementation.
- Original checkpoint-selection criterion.
- Full-volume inference procedure.

The existing reference protocol trained for 50 epochs using 260 patches per epoch.

The intended experimental treatment is the selected Train sampling strategy.

Do not silently change several major variables simultaneously.

## Execution

1. Verify training data and output-path separation.
2. Run a short forward/backward training smoke test.
3. Check gradient validity, tensor shapes, and GPU memory.
4. Start the approved controlled experiment on Rangpur A100 using Slurm.
5. Preserve checkpoints and epoch-level training records.
6. Record actual GPU resource consumption.
7. Evaluate using the established Validation protocol.
8. Compare directly with the existing Improved reference.

Use existing infrastructure where possible.

Do not modify or overwrite historical runs.

## Evaluation

Retain the original patient-equal small-organ Dice as the primary model-selection metric.

Also report:

- Six-class Dice and IoU.
- Organ precision and recall.
- Body HD95 in millimetres.
- Distant Background-to-Body false positives.
- False positives by distance and spatial location.
- Per-patient improvements and regressions.
- Representative prostate and rectum boundary cases.
- Training time, peak GPU VRAM, and inference latency.

Keep patient-equal and pooled-voxel metrics explicitly separate.

## Initial engineering targets

The previous proposal used:

- Small-organ Validation Dice at least 0.8420.
- Median Body HD95 below 10 mm.
- At least 90% reduction in distant Background-to-Body false positives.

These are provisional engineering milestones, not proof of superiority over Standard 3D.

Evaluate whether they remain scientifically appropriate. Any revised criteria must be justified independently of the experiment's observed outcome.

Compare against the stronger Standard 3D Body boundary performance as well.

## Scientific interpretation

If Body performance improves but small-organ segmentation deteriorates, report a trade-off rather than an unqualified improvement.

If the treatment fails, investigate whether the result falsifies the sampling hypothesis or whether other mechanisms remain plausible.

Do not attribute improvement to the architecture when only the sampling strategy has changed.

## Deliverable

Produce a controlled experimental comparison containing:

- Exact control and treatment definitions.
- Training configuration.
- Execution and reproducibility evidence.
- Validation metrics.
- Per-patient differences.
- Boundary failure visualisations.
- Computation cost.
- Supported interpretation.
- Recommended next experiment.

Do not use the previously inspected Test data for model selection.

## P5 — 让 AI 自主决定下一次模型优化方向

Adaptive Experiment Design · 不强制依次进行所有改进

这是整套 Prompt 系列中最重要的自主决策环节。

我们已经有三条值得研究的技术路线：

| 候选实验                      | 适合在什么情况下进行                         | 主要研究目的        |
| ------------------------- | ---------------------------------- | ------------- |
| Standard + New Sampler    | 新采样显著改善 Improved                   | 排除采样差异，检验架构贡献 |
| Deep Supervision Ablation | Body 误报仍严重，或诊断支持 head interaction  | 分离辅助输出融合的影响   |
| Loss Rebalancing          | Body 恢复后，小器官 Precision/Recall 仍不平衡 | 控制过度分割与漏检     |

但是，下一轮到底选择哪项，应该由实验结果决定，而不是要求 AI 把三项全部做完。

# P5 — Select the Next Model Improvement Using Experimental Evidence

Apply the Master Prompt.

## Objective

Use the completed diagnostic and controlled-training evidence to select the next scientifically justified intervention.

Do not execute a predetermined sequence of modifications if the results support a better strategy.

## Central research objective

Improve the robustness of 3D anatomical segmentation, particularly:

- Prostate boundary accuracy.
- Rectum and bladder interfaces.
- Base/apex slice behaviour.
- Three-dimensional spatial consistency.
- Body and Background separation.
- Computational efficiency.

Preserve the selected Hard Difficulty research scope.

## Decision framework

Review the latest experiment and decide which scientific question remains most important.

Consider the following candidate branches, but do not regard them as exhaustive.

### Branch A — Architecture contribution under matched sampling

Choose this branch if the new sampling strategy materially improves the Improved model.

Train Standard 3D U-Net with the same new sampler while matching other relevant conditions.

Compare:

- Standard + Original Sampler
- Improved + Original Sampler
- Standard + New Sampler
- Improved + New Sampler

The purpose is to distinguish sampling effects, architecture effects, and possible interactions.

Do not claim that the Improved architecture itself is superior unless comparable training conditions support that conclusion.

### Branch B — Deep-supervision architecture ablation

Choose this branch if evidence implicates output-head interaction or if the sampling intervention fails to resolve major spatial errors.

Evaluate a retrained alternative in which auxiliary heads contribute supervised training losses but the main full-resolution head provides inference logits.

Compare against the original summed-logit architecture.

Retain volumetric residual units and meaningful deep supervision so the Hard Difficulty project remains valid.

Do not substitute main-head-only inference on existing summed-output weights for a proper retrained ablation.

### Branch C — Loss-function intervention

Choose this branch if evidence indicates that false-positive costs are inadequately represented by the current loss.

Potential candidates include:

- Revised class weighting.
- Cross-entropy and Dice weighting adjustments.
- Focal Tversky or Tversky-style objectives.
- A carefully defined false-positive penalty.

Do not assume any candidate will improve results.

Explain the mathematical mechanism, expected gradient effects, and likely precision-recall trade-offs before implementation.

Use Train data to fit the model and Validation to evaluate candidate settings.

Avoid excessive hyperparameter searching given the small independent Validation cohort.

### Branch D — Alternative diagnosis or intervention

If the evidence indicates that normalization, interpolation, padding, spatial resolution, data quality, or another factor is more important than the three options above, propose a different controlled experiment.

You are explicitly permitted to challenge the previous priorities.

## Resource and stopping policy

Prefer one high-information experiment over multiple weak comparisons.

Estimate expected GPU cost before new training.

If additional training is unlikely to improve the scientific conclusion or assessment deliverables, recommend stopping model optimisation.

Do not treat a more complex network as inherently better.

## Output requirements

First produce:

1. The current dominant failure mechanism.
2. Evidence for and against each candidate branch.
3. The preferred next experiment.
4. Its control and treatment.
5. Primary and secondary metrics.
6. The expected computation cost.
7. The condition that would invalidate the hypothesis.

Then implement the selected experiment if it falls within the authorised research scope and resource budget.

Preserve all previous baselines and experiment identities.

After execution, report what actually improved, what regressed, and what remains unresolved.

## P6 — 回归课程核心：临床重要边界与三维连续性

Clinical Boundary Analysis · 直接对应老师 Section 2.2 的 Engineering Dilemma

我认为，这一步在评分上的价值可能比继续提高 0.01 Dice 更高。

你的课程文件并没有要求模型必须全面战胜所有基线才可以获得高分，而是要求通过实验回答：

3D 空间上下文是否真正改善重要边界？额外的 GPU 计算成本是否合理？

目前我们提供的图片主要证明了不同器官的 Over-segmentation 和远处误报。未来还应该特别补强 Prostate base/apex、相邻切片连续性及危险解剖交界。

# P6 — Investigate Clinical Boundary Utility and 3D Spatial Generalisation

Apply the Master Prompt.

## Objective

Conduct the assessment's central scientific investigation:

**Does three-dimensional spatial context and the selected Improved U-Net architecture provide meaningful anatomical boundary and slice-continuity advantages over simpler baselines, and are those gains worth the additional computational cost?**

This stage must go beyond aggregate Dice scores.

## Required comparison

Use available, validly evaluated models:

- 2D U-Net.
- Standard 3D U-Net.
- Selected Improved 3D U-Net configuration.

The 2D and 3D pipelines may differ in sampling, resolution, channel design, and optimisation.

Explicitly document these differences.

Do not attribute all 2D-versus-3D performance differences exclusively to spatial context unless the experimental design supports that causal inference.

Use the matched Standard-versus-Improved comparison when investigating additional architectural contributions.

## Investigation A — Clinical boundary precision

Evaluate Prostate, Rectum, and Bladder boundaries.

Investigate:

- Sørensen-Dice and IoU.
- Precision and recall.
- HD95 and, if reliable, average symmetric surface distance.
- Systematic over-contouring and under-contouring.
- False positives and false negatives near adjacent anatomical structures.

Use appropriate physical spacing and report measurements in millimetres.

Do not infer clinical dose consequences directly from segmentation overlap alone.

## Investigation B — Prostate base and apex

Use the anatomical label extent to establish a reproducible base/apex evaluation protocol.

Measure performance on inferior and superior organ slices separately from easier central slices.

Define the slice-selection rule before comparing outcomes.

Investigate:

- Missing terminal slices.
- Excessive extension beyond the organ.
- Boundary leakage.
- Sensitivity to small target volume.
- Variation across patients.

Avoid choosing only visually favourable examples.

## Investigation C — 3D slice continuity

Investigate whether the models produce:

- Abrupt organ appearance or disappearance.
- Fragmented three-dimensional structures.
- Unstable contour area between adjacent slices.
- Unexplained detached components.
- Unnatural shape transitions.

Use quantitative continuity measurements where meaningful, together with visual evidence.

Distinguish genuine anatomical variation from segmentation instability.

## Investigation D — Representative failures

Prepare 3–5 detailed case autopsies that cover distinct and important mechanisms.

Consider examples involving:

- Distant Body false positives.
- Prostate over-segmentation.
- Rectum precision-recall trade-offs.
- Bladder localisation failure.
- Prostate base/apex boundary errors.

Use Validation cases for continued development and diagnosis.

Previously inspected Test examples may be documented as historical descriptive observations, but must not guide new model selection or threshold tuning.

For each selected case, provide:

- Input MRI.
- Ground-truth segmentation.
- Model prediction.
- False-positive and false-negative overlay.
- Relevant quantitative measurements.
- Likely failure mechanism and its evidential limitations.

## Investigation E — Engineering cost

Benchmark:

- Model parameter count.
- Peak GPU VRAM.
- Per-case inference time.
- Training time.
- Important implementation and hardware differences.

Evaluate whether additional complexity produces measurable benefits.

## Generalisation safeguards

Respect patient-level dependence when reporting uncertainty.

Show per-patient results and clarify how repeated weekly scans are aggregated.

Do not interpret six patients as dozens of independent observations.

Previously inspected Test data are not a fresh blind evaluation resource.

If no new untouched patient cohort is available, state the limit clearly.

## Deliverable

Produce an evidence-based comparison answering:

1. Where does 3D context help?
2. Where does the Improved architecture help or regress?
3. Which anatomical structures and boundary regions remain difficult?
4. Does the improvement justify additional memory and runtime?
5. Which cases should always require expert review?
6. What is the most defensible engineering recommendation?

The objective is a scientifically credible answer, not a predetermined victory for the Improved model.

## P7 — 将研究结果转化为最终高分交付

Assessment Delivery · README、代码质量、Git、Pull Request、AI Disclosure

这一步不再负责无限制优化模型，而是判断我们是否已经拥有足够证据完成课程任务。

# P7 — Finalise the COMP3710 HipMRI Research Prototype and Submission

Apply the Master Prompt.

## Objective

Transform the completed HipMRI research into a professional, reproducible COMP3710 Final Assessment submission.

Optimise for compliance with the confirmed assessment specification, scientific credibility, and high-quality engineering communication.

Do not continue costly model training unless an essential scientific or assessment gap justifies it.

## Task 1 — Verify assessment requirements

Read the supplied COMP3710 Final Assessment Specification and check for any newer confirmed official requirements.

The supplied draft specifies:

- Hard Difficulty 3D Improved U-Net.
- Volumetric residual units and deep supervision.
- An implemented baseline comparison.
- Per-class Dice and IoU.
- Critical boundary analysis and representative failure cases.
- Peak GPU VRAM and inference runtime.
- Reproducible code and a technical README.
- Progressive Git commits.
- GitHub Pull Request and subsequent code-review response.
- Mandatory AI usage disclosure.

The draft also specifies a target Test Dice of at least 0.7 for the Hard 3D Improved U-Net.

Verify the exact interpretation of that target before claiming compliance.

Do not assume the existing patient-equal small-organ Dice is identical to the course's intended scoring definition.

## Task 2 — Final model selection

Use the permitted Validation results and the predefined decision criteria to select the final candidate.

The candidate should:

- Satisfy the selected Hard Difficulty requirements.
- Be functional and reproducible.
- Have a defensible performance profile.
- Include an honest discussion of remaining failures.
- Have a clear engineering justification for its computational cost.

A negative or mixed architectural comparison is acceptable when supported by evidence.

Do not re-use the previously inspected Test cohort for selecting the final configuration.

If an untouched cohort is unavailable, clearly distinguish existing historical Test results from new Validation-only experiments.

Do not label an already inspected cohort as a new independent Test.

## Task 3 — Review the mandatory codebase

Ensure the submission retains the required functional entry points:

- modules.py
- dataset.py
- train.py
- predict.py
- README.md

The assessment also permits additional helper modules.

Verify:

- PyTorch architecture implementation.
- Data loading and spatial preprocessing.
- Patient-level separation.
- Correct loss and gradient calculations.
- Working training and inference.
- Model checkpoint handling.
- Deterministic or reproducible evaluation.
- Readable code, comments, and tests.
- Documented software dependencies.

Respect the specification's restriction on dependencies used within model components.

Avoid large structural refactors unless necessary.

Do not commit datasets or model weights.

## Task 4 — Produce the final technical README

The README should contain:

1. Problem definition and clinical context.
2. Course engineering dilemma.
3. Dataset and patient-level splitting.
4. Feasibility review.
5. Preprocessing and spatial-alignment methods.
6. Baseline and Improved model architectures.
7. Residual learning and deep-supervision mechanisms.
8. Loss functions and training protocol.
9. Controlled experimental design.
10. Quantitative model comparison.
11. Anatomical boundary and 3D continuity analysis.
12. Three to five representative failure autopsies.
13. Computational resource analysis.
14. Engineering trade-offs and recommendation.
15. Reproducibility and usage instructions.
16. Limitations and future work.
17. References.
18. Artificial Intelligence Usage Disclosure.

Integrate existing research evidence rather than producing an unrelated second report.

Clearly differentiate confirmed results, interpretations, hypotheses, and future proposals.

## Task 5 — Prepare the formal engineering recommendation

Answer the course's central dilemma directly.

Explain whether the Improved model should be:

- Preferred over the baseline for further research.
- Retained as an experimental prototype with important limitations.
- Rejected in favour of the simpler model under the measured constraints.

Base the recommendation on actual performance, boundary reliability, computational expense, and generalisation evidence.

Do not claim that the prototype is clinically safe or ready for unsupervised radiotherapy contouring.

## Task 6 — Verify version-control and submission readiness

Check:

- Correct repository and branch.
- Meaningful progressive commit history.
- No accidental dataset or checkpoint commits.
- Working prediction entry point.
- Accurate environment instructions.
- Complete README.
- Clear AI usage disclosure.
- Professional Pull Request description.
- Correct Turnitin PDF submission requirement.
- Outstanding compulsory hurdles and official deadlines.

Do not rewrite Git history or force-push without explicit authorisation.

Do not claim that a Pull Request has been submitted unless it actually has.

Treat later teaching-team code review as a separate assessment phase, and prepare to respond when comments become available.

## Deliverable

Provide a final assessment-readiness report showing:

- Verified requirements.
- Completed deliverables.
- Missing or at-risk components.
- Evidence supporting the model-selection decision.
- Remaining scientific limitations.
- Submission readiness.
- The smallest necessary remaining actions.

Prioritise completing the existing project properly over introducing unnecessary new technology.

## 四、怎样保证另一个 AI 真正拥有自主分析权？

我特别建议你使用下面这种权限分层，而不是简单告诉 Codex“严格按照这些步骤执行”。

| 内容                            | AI 的自主权限      | 原因                      |
| ----------------------------- | ------------- | ----------------------- |
| 研究假设排序                        | 可以完全重新排序      | 我们的原因分析还不是定论            |
| Logits / FP 诊断                | 可以增加、删减或替换诊断  | 要以证据效率为优先               |
| Edge Sampler                  | 可以修改甚至拒绝      | 现有设计尚未由真实训练验证           |
| Loss Function                 | 可以提出并实施受控候选实验 | 不应锁死 Weighted CE + Dice |
| Deep Supervision              | 可以修改融合或训练方式   | 但最终仍需符合 Hard 要求         |
| Validation 指标                 | 可以增加评价视角      | 原始主要选模指标应保留以便比较         |
| 实验顺序                          | 可以改变，有理由即可    | 不浪费有限的 A100 预算          |
| Patient split                 | 不得随意改变        | 保护已建立的实验可比性             |
| Historical logs / checkpoints | 不得覆盖          | 保护实验真实性                 |
| 已观察的 Test                     | 不可用于新模型选择     | 避免评估信息泄漏                |
| 最终研究结论                        | 完全由结果决定       | 允许 Improved 不如 Standard |

也就是说，这组 Prompts 的目的不是代替另一个 AI 思考，而是为它提供一个有研究目标、有必要约束、但允许技术创新的实验环境。

## 五、实际使用方式

我建议你在新的 Codex 会话中提供四类材料：课程 Assessment PDF、最新 `HipMRI_Research_Report.md`、2026 年 10 月 19 日的模型决策备忘录，以及当前项目仓库和可访问的实验日志、JSON、模型文件。

然后按照下面的方式执行：

第一轮：Master + P1

让 Codex 独立审查研究，不直接训练。

第二轮：P2 + P3

在 P1 结果支持的情况下，诊断错误来源并验证采样方案。若发现更有力的原因，可调整路线。

第三轮：P4 + P5

进行有明确假设的模型训练，然后根据 Validation 结果决定下一轮是否值得继续。

第四轮：P6 + P7

回答课程核心研究问题，完成实验报告和提交工作。

## 六、最后给你的研究策略建议

对于这个 COMP3710 项目，我最不希望发生两种情况。

第一种是为了得到比 Standard 更高的 Dice，不断修改网络和 Loss，最终得到了一个分数更高但解释不清、泛化不稳的模型。

第二种是过度关注代码审计、实验记录和复杂的工程控制，反而消耗了真正研究模型的时间。

这组 Prompts 要让另一个 AI 在两者之间保持平衡：

研究问题 → 诊断证据 → 可检验假设 → 最小有效改动 → 受控训练 → Validation 比较 → 独立解释 → 课程交付。

我的最终建议是：将 P1 的独立研究审查作为第一道关口，但不要将 G7 Edge Sampling 或 Deep Supervision 消融视为必须完成的研究路线。 如果另一个 AI 发现比采样不足更有说服力的失效机制，它应该有权改变后续实验方向。

这样既保留了我们已经积累的研究成果，也真正给予另一个 AI 发挥专业判断的空间。
