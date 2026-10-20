# V4 配对实验分析

## 完成记录与选模

A0/A1各50份epoch记录、配置和manifest齐全。从每5轮测量的10个Validation候选中选最高分，同分取较早轮次；两组均选45轮。每名患者先汇总各visit混淆矩阵，再计算指标，最后对6名患者等权平均。复算选模分数与原始记录一致。本地SHA清单记录输入身份，尚未对本次JSON进行远端逐文件核验。

| 最佳轮次指标 | A0 | A1 |
|---|---:|---:|
| 小器官Dice | 0.84500 | 0.82976 |
| Body Dice | 0.97757 | 0.97282 |
| Bones Dice | 0.87323 | 0.86029 |
| Bladder Dice | 0.90822 | 0.89175 |
| Rectum Dice | 0.80341 | 0.78550 |
| Prostate Dice | 0.82338 | 0.81205 |
| Body HD95，volume中位数/mm | 4.7509 | 5.6838 |
| 距真实Body至少20mm的Background→Body体素总数 | 9572 | 1171 |

## 解释与限制

A1相对A0小器官Dice下降0.01524，6/6患者下降，六类Dice均下降。远处Body假阳性减少87.77%，但Body HD95中位数上升，说明远处孤立错误与整体边界准确性不是同一指标。Rectum召回率0.90673→0.91494，同时precision由0.72753下降至0.69460，支持过分割权衡。

历史最佳Validation小器官Dice：Standard 0.85022、v1 0.84200、v2 0.83620、v3 0.85636。两组均未超过v3或Standard。A0相对v3同时改变优化器、权重衰减及学习率计划，不能单独归因于调度；A1/A0主要受控差异是配对空间增强。一个种子、6名被反复使用的Validation患者不足以证明普遍因果效果或泛化优势。不得由此断言空间增强普遍无效。

初始作业679166/679167失败时HOME已耗尽，但未捕获具体Python异常。恢复前latest.pt均可读，分别保存4797/4550次更新；续训679977/679982均COMPLETED 0:0并达到50轮。保留原始与恢复日志；最终完成不等于CUDA重启轨迹逐位等同。

## 已生成与下一步

- [学习曲线](learning_curves.svg)：增强输入不同，两组训练loss不直接等价。
- [六类Dice](six_class_dice.svg)：各自最佳轮次、患者等权。
- [六类指标](six_class_metrics.csv)：Dice、IoU、precision、recall、前景HD95。
- [患者指标](patient_metrics.csv)、[输入SHA清单](record_checksums.json)。

当前为Validation数值复核，未新运行Test或生成真实MRI分割图。下一步固定第45轮权重，复现38例Validation矩阵并生成统一规则的六类与Body三平面对照，检查远处FP减少是否伴随近边界FN增加，定位Rectum precision下降。之后再决定固定模型的Test描述性评估；已看过Test不可充当新盲测，不据其选择超参数。现阶段不启动新训练。

## 七模型统一复核与课程覆盖（2026-10-21）

已重新读取全部七组共350份epoch记录，按同一个患者均衡Validation规则选择各自best并复算分数。七组均非依据Test选模。

| 模型 | 最佳测量轮次 | 患者均衡小器官Validation Dice |
|---|---:|---:|
| 2D补充模型 | 50 | 0.77569 |
| Standard 3D | 50 | 0.85022 |
| Improved v1 | 50 | 0.84200 |
| Improved v2 | 45 | 0.83620 |
| Improved v3 | 50 | **0.85636** |
| v4 A0 | 45 | 0.84500 |
| v4 A1 | 45 | 0.82976 |

因此A1不是主要验证指标最好的模型，当前候选仍为v3；这不是对最终Test或外部泛化能力的排名。[七模型选模图](all_models/seven_model_selection.svg)、[六类热图](all_models/seven_model_six_class_dice.svg)、[六类Dice/IoU/precision/recall数据](all_models/six_class_validation.csv)已生成，全部来自实际记录，无填补Test数值。

重新读取本地课程PDF：`COMP3710_Report_2026_draft_v2.2.pdf`，内部标题为Version 2.0 Draft，项目要求见PDF第5–6页，交付/评分/AI披露见第13–15页。该本地草稿与最新课程公告是否完全一致尚未核验。

| 要求 | 当前证据与尚缺内容 |
|---|---|
| Hard：下采样HipMRI、3D residual units、deep supervision | v1/v2/v3/A0/A1有对应结构代码；结构达成不等于全部评分条件完成。 |
| Test Dice ≥0.7 | 草稿未明确聚合规则；应列每类、Prostate与预声明均值，不能用Validation三器官分数替代。A0/A1尚无固定best Test结果。 |
| 患者分割、标准化、图像mask对齐 | 既有审计与代码证据；Dice内部one-hot及CE整数类别是实现方式，需在最终方法说明中解释。 |
| 同split baseline对比、逐类Test Dice/IoU | Validation七模型已复算；统一七模型Test尚未执行。已有Standard/v1 Test保留，不能把旧结果冒充全部版本评估。 |
| 显存、参数量、每例推理时间 | 训练峰值与原有探针不等于统一推理比较；新程序统一参考volume、3 warm-up/5次计时与allocated/reserved VRAM，另记录全部volume单次同步推理耗时和峰值，区分重复benchmark与队列单次分布。 |
| 2D/3D连续性与base/apex、危险OAR界面 | 新统一审计将输出物理base/apex、Prostate入Bladder/Rectum混淆、连通域/空缺切片/面积变化。它们是描述代理，不是临床风险或剂量结论。 |
| 3–5真实失败案例与FP/FN多面板 | 历史版本已有案例；A0/A1的统一真实图仍待远端生成，固定6个Validation患者并含既有Test的3例描述性案例。需人工读图写出案例解剖分析，不能仅以生成图片算完成。 |
| 2D资料路线 | 当前2D是同一3D队列派生切片的补充对照，并非草稿Easy路线的keras_slices_data；不能声称完整复刻该独立路线。训练配方不同也限制纯空间维度归因。 |
| 可复现README、可行性报告、AI披露、依赖版本、保存/预测入口 | 已有材料但最终交付还需针对选定模型汇总与运行核验，不能仅凭目录存在判定全部达标。 |
| midpoint consultation、正确PR和后续review回应 | 数值程序无法证明；需独立检查实际提交/签到/教师review状态。 |

已准备新的七模型固定评估传输包及CPU几何单元测试；尚未上传或运行，GPU复现、图像与Test结果均pending。输出独立进入v4的`comprehensive_audit_20261021`，不覆盖历史数据。Test已被多次查看，只作固定模型描述性比较，不指导后续调参。最终“是否满足全部要求”需在结果导回和读图后逐项更新，当前不能肯定。

2026-10-21已核验并上传七模型评估包（SHA-256 `cd5c3ff8f5756a485ab00254f822959b3256341ab41404c136ff0afbf0416970`），解压与shell语法检查通过。已提交CPU检查680797及依赖的GPU评估680798；最后观察状态分别为PENDING/Priority与PENDING/Dependency。完成状态、实际指标和图像仍待核验。

## 原始Train增强审计（完整保留）

审计作业679138：COMPLETED，退出码0，用时41秒。下载的audit.json、六张PNG及README均与随附checksums.json匹配；这证明传输完整性，不是独立远端签名。

260对patch覆盖26名Train患者；69个实际增强（26.54%，30%概率的有限样本实现）。增强前后体素占比之比：Bladder 0.9962、Rectum 0.9990、Prostate 0.9936。平均原图外采样0.1412%，单patch最大3.4340%。四项预设机械门槛均通过。

逐张查看六张预览：O025_Week6四张、R039_Week2两张。可见温和位置、形状投影变化及插值平滑，叠加标签与MRI主要结构未见明显整体脱离或图像与标签反向变换。预览仅覆盖两名患者若干二维切片，不构成全体积或临床标注正确性证明；增强前后相同索引也不代表相同物理切面。未使用翻转或强形变。

据此按既定协议提交A0/A1训练。增强是否提高性能，仍需完整Validation按best选模、六类及患者级指标检验；本审计没有训练模型，也没有证明泛化改善。两组训练已提交679166/679167，尚未收到完成证据。
