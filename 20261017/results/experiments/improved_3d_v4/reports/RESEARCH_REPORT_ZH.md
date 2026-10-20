# 七模型最佳权重统一评估与 v4 配对实验结论

更新：2026-10-21。统一评估作业680798，COMPLETED/0:0，20分26秒；CPU检查680797通过。本文是本轮最终分析入口，原始训练、早期方法审查和报告正文保留。

## 1. 结论与模型选择

**A1不是当前最好的模型。按事先约定的Validation患者均衡小器官Dice，仍应保留v3作为主要Improved候选，Standard作为基线。** v3在既有Test的相同均值也最高，但不代表所有患者、类别和边界都胜出。其Test优势受M015明显影响，并存在极端表面距离错误。

**已完成本轮七模型数值评估、统一资源测量及指定真实案例审阅；还不能宣称作业所有交付要求、临床可用性和外部泛化均已满足。** Hard结构和平均Test性能已有证据；课程入口兼容、最终README整理、PR/教师review及midpoint手续需另外核实。Test早已查看，不再属于全新盲测。

## 2. 数据来源、完整性与评估规则

- 固定26/6/6名Train/Validation/Test患者，143/38/30个volume。七模型共有476次模型/volume评估。
- 下载目录37个文件全部通过随附SHA-256；最终JSON `complete=true`。分析脚本另核对运行脚本SHA、锁定配置SHA、全部模型manifest和最佳epoch记录。
- 本地逐矩阵核对266次Validation复现，以及Standard/v1的60次既有Test复现。精确文件SHA未发现跨split MRI重复；不等于排除了不同编码或近重复影像。
- 各模型从50轮内实际测量的10个Validation候选轮次选最高分，同分取较早轮次。不声称逐个验证了50份权重，不用训练loss或Test选轮次。
- 患者均衡重叠指标：先合并同一患者各visit的混淆矩阵，计算Dice/IoU/precision/recall，再平均6名患者。另保留体素pooled结果；38或30个volume不当作38或30名独立患者。
- HD95/ASSD在降采样的评估网格上用物理spacing计算；不是原始分辨率的临床误差保证。2380项前景表面测量全部为`both_present`，无空mask距离被静默删去。基底/尖端按GT阳性切片物理superior位置的首尾20%划分，是几何代理。
- 没有重训练、改阈值、后处理或覆盖原始JSON。远端程序核对selected epoch权重与best.pt一致；本地本轮未重新下载七组权重，不能把记录核验描述成独立本地权重推理。

[完整性证据](../assessment_analysis/integrity_verification.json)；[数值与图表复现脚本](../../../../code/tools/summarize_all_model_audit.py)。

## 3. 七模型总体表现

下表是患者均衡指标。小器官为Bladder、Rectum、Prostate三类平均。

| 模型 | best轮次 | Validation小器官Dice | 既有Test小器官Dice | Test Prostate Dice |
|---|---:|---:|---:|---:|
| 2D | 50 | 0.77569 | 0.70822 | 0.66112 |
| Standard | 50 | 0.85022 | 0.78510 | 0.74329 |
| v1 | 50 | 0.84200 | 0.80778 | **0.75202** |
| v2 | 45 | 0.83620 | 0.79590 | 0.74778 |
| v3 | 50 | **0.85636** | **0.81448** | 0.74850 |
| A0 | 45 | 0.84500 | 0.78365 | 0.73182 |
| A1 | 45 | 0.82976 | 0.78012 | 0.72734 |

v3不是Test前列腺Dice最高者；v1该项略高，但Body边界明显更差。v3相对Standard的Validation均值提高0.00613，5/6名患者提高；Test提高0.02938，但只有3/6名患者提高。

### Test六类Dice

| 模型 | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9896 | 0.9681 | 0.8487 | 0.7508 | 0.7127 | 0.6611 |
| Standard | 0.9904 | 0.9677 | 0.8619 | 0.7993 | 0.8128 | 0.7433 |
| v1 | 0.9683 | 0.9470 | 0.8808 | 0.8428 | 0.8285 | 0.7520 |
| v2 | 0.9783 | 0.9565 | 0.8668 | 0.8484 | 0.7916 | 0.7478 |
| v3 | 0.9885 | **0.9732** | **0.8988** | **0.8539** | **0.8410** | 0.7485 |
| A0 | 0.9892 | 0.9704 | 0.8786 | 0.8002 | 0.8189 | 0.7318 |
| A1 | 0.9890 | 0.9686 | 0.8721 | 0.8236 | 0.7894 | 0.7273 |

[六类Dice/IoU图](../assessment_analysis/assessment_required/figures/six_class_validation_test.svg)；[完整精确指标及pooled口径](../assessment_analysis/assessment_required/tables/six_class_metrics.csv)。不能把三类均值称为全部类别准确率。

## 4. A1改善什么，又退步什么

A1/A0共用v3网络、AdamW及6500次更新的余弦计划；主要受控组间差异是30%概率的温和配对3D旋转/缩放。A0/v3同时改变优化器、权重衰减和学习率计划，不能单独归因于余弦衰减。

- **小器官重叠退步：** A1相对A0的Validation差值−0.01524，6/6患者下降；Test差值−0.00353，5/6患者下降。A1相对v3的Test在6/6患者均下降，不是只由一个患者造成。
- **远处Body FP改善：** Validation 9572→1171（−87.77%）；Test 9636→535（−94.45%）。A1的Test远处Body FP少于Standard的2804，是实际优点。
- **Body局部表面并未同步改善：** Validation Body HD95中位数4.7509→5.6838mm，Test也由4.7509→5.6838mm。远处假阳性减少不等于贴近轮廓的漏分割消失。
- **OAR错类风险增加：** Test中预测Prostate落入真实Bladder/Rectum的体素数A0为937，A1为1639；占预测Prostate的比例2.790%→4.937%。这是分类混淆代理，不是剂量或真实黏膜损伤测量。
- **连续性有局部优点：** A1在38卷Validation中没有多连通Prostate；v3有6卷。但Test中A1仍有2卷多连通。内部空缺切片总数A0/A1均为0，v3为85；远处孤立假阳性会扩大预测z范围并产生“内部空缺”，该数不能直接等同器官主体断裂。

增强的机械预检通过，不能据结果下降直接认定变换实现错误。标签最近邻重采样、额外正则化、训练预算和数据分布交互都可作为假设；现有单种子配对结果不能证明其中哪一个是根因。固定50轮和当前增强强度下，收益不足以替代v3；不能泛化为“空间增强没有用”。

[远处Body误报](../assessment_analysis/supplementary_research/figures/distant_body_false_positives.svg)、[OAR/漏分割权衡](../assessment_analysis/assessment_required/figures/prostate_oar_tradeoff.svg)、[连续性](../assessment_analysis/supplementary_research/figures/prostate_continuity.svg)。

## 5. 3D上下文与临床边界问题

### 5.1 端部依然薄弱

Test按患者等权、患者内visit平均后的前列腺切片区域Dice：

| 模型 | Apex | Central | Base |
|---|---:|---:|---:|
| 2D | 0.5186 | 0.7125 | 0.3473 |
| Standard | 0.7544 | 0.7824 | 0.5097 |
| v1 | 0.7397 | **0.7997** | **0.5993** |
| v2 | 0.7120 | 0.7877 | 0.5836 |
| v3 | 0.7523 | 0.7931 | 0.5069 |
| A0 | 0.7169 | 0.7687 | 0.4839 |
| A1 | 0.7251 | 0.7620 | 0.5472 |

3D相对2D整体端部表现改善，但没有解决基底错误。A1的Base优于A0和v3，却未带来更好的总体小器官Dice；因此不能只盯一个指标。2D/3D的宽度、参数量、采样和loss配方存在差异，这不是仅改变维度的纯因果实验。

### 5.2 中位数不能掩盖极端错误

| 模型 | Test Body HD95中位数/mm | Test Prostate HD95中位数/mm | Test Prostate HD95最大值/mm |
|---|---:|---:|---:|
| Standard | 4.861 | 7.979 | 19.229 |
| v1 | 51.813 | 6.719 | 122.347 |
| v2 | 28.555 | 7.460 | 16.717 |
| v3 | **4.500** | **6.645** | 115.371 |
| A0 | 4.751 | 8.611 | 23.712 |
| A1 | 5.684 | 9.265 | 22.211 |

v3的极端Test病例为L011_Week0；Validation也有S022_Week6的106.12mm。它们没有被本轮固定图片集覆盖：不能从数字臆造该两例的具体形态或根因。后续如需优化，应优先用固定Validation的S022_Week6审阅尾部错误，不能围绕L011 Test调参。v3不是所有安全相关指标的支配解。

[Body全范围分布](../assessment_analysis/assessment_required/figures/body_boundary_distribution.svg)、[Prostate尾部](../assessment_analysis/assessment_required/figures/prostate_boundary_tails.svg)。HD95是双向表面距离合并后的95百分位，ASSD为同一合并表面点集合均值；不是最大Hausdorff距离，也不代表临床容忍阈值。

## 6. 五例真实失败审阅与一个成功对照

以下已实际查看MRI、GT、预测及FP/FN图。数值取整卷混淆矩阵，不能与图中单切片面积混用。青色GT、橙色预测、红色FP、蓝色FN。首行GT最大面积，第二行固定A1最大FP+FN面积，所有模型共用切片；选择偏向诊断A1，不是公平的“所有模型最坏一例”。没有独立临床专家的解剖审核。

### 6.1 K018_Week4：远处Body幻影与近轮廓漏分割并存（Validation）

[Body三平面对照](../comprehensive_audit_20261021/images/validation/K018_Week4_Body_three_planes.png)中v1/v2在身体外有明显红色预测区域；v3/A0/A1显著减少这些区域。v1远处FP为58078、Body HD95为125.632mm；A1为152和5.684mm。A1与GT真实轮廓之间仍可见蓝色漏分割，解释了“远处错误变少但边界并非处处更准”。支持版本间输出形态差异，不单凭本图归因于某个归一化机制。

### 6.2 M015_Week7：Bladder定位与范围失败（既有Test，描述性）

[Bladder对照](../comprehensive_audit_20261021/images/test/M015_Week7_Bladder.png)显示Standard/A0在GT外大片后侧区域预测Bladder，GT内也有漏分割；第二行部分模型几乎遗漏GT区域。整卷Dice为Standard 0.0676、v3 0.5224、A0 0.2265、A1 0.3775。v3改善Standard并不等于成功：其recall仍仅0.4070。图像支持错位置和漏分割并存；影像外观、标注范围或患者差异的根因仍未被因果隔离，不自行判断GT错误。

### 6.3 W029_Week6：3D高召回伴随过分割；2D反而较好（既有Test）

[Prostate对照](../comprehensive_audit_20261021/images/test/W029_Week6_Prostate.png)与[端部图](../comprehensive_audit_20261021/images/test/W029_Week6_Prostate_base_apex.png)显示多个3D预测在较小GT外明显扩张。整卷2D Dice 0.6485，Standard 0.4623，v3 0.4377，A1 0.4057。v3 recall 0.9829而precision仅0.2815；A1 recall同为0.9829、precision 0.2556。它们没有因为找到大多数GT而形成可用轮廓。2D本例更克制，说明3D应学习定位约束与precision控制，而不是盲目提高recall；本Test观察只作解释，不能据此反向选择训练权重。

### 6.4 J026_Week4：Rectum外扩，A1不如v3（Validation）

[Rectum对照](../comprehensive_audit_20261021/images/validation/J026_Week4_Rectum.png)显示中心切片看起来尚可，在另一切片出现GT外预测。整卷v3 Dice 0.8437、precision 0.7574；A1 Dice 0.7851、precision 0.6628，而recall仍达0.9628。支持A1多覆盖了非目标区域，不支持“增强让所有小器官更稳健”。低对比、邻近组织相似是合理机制假设，当前图像没有单独证实。

### 6.5 B040_Week1：Bones覆盖进步仍未消除端部缺失（Validation）

[Bones对照](../comprehensive_audit_20261021/images/validation/B040_Week1_Bones.png)中第二排的GT骨性结构存在边缘漏分割及预测向非GT区域扩张。整卷Dice：2D 0.8270、Standard 0.8607、v1 0.9102、v3 0.9026、A1 0.8743。v1本例优于v3，A1并未继承这项优势。这里是合并Bones标签，不能据此宣称已单独分析每个股骨头或所有细薄骨界面。

### 成功对照：M020_Week1不能代表所有患者

[Prostate base/apex图](../comprehensive_audit_20261021/images/test/M020_Week1_Prostate_base_apex.png)中大部分3D轮廓与GT接近。整卷Standard Dice 0.9046、v3 0.8911、A1 0.8873。它证明流程能在某些体积形成良好轮廓，也证明Standard在具体病例可优于Improved；不能用成功个例覆盖前述失败。

## 7. 泛化和患者依赖

Test仍按6名患者等权统计，不因M015困难而删除它。保留全部患者时v3−Standard为+0.02938；仅作敏感性检查去掉M015后变为−0.00496。逐一留出患者的差值范围−0.00496至+0.04413，说明平均优势对患者构成敏感。此范围不是置信区间，删除M015后的结果不是新的正式Test指标。

A1−v3在Test六名患者均为负，留一患者均值范围也始终为负（−0.03913至−0.02105）。因此A1没有显示本队列中更强的小器官泛化。A1相对A0只在M015患者提高；这不足以抵消其他五名患者退步。

每个配方仅一个种子；Validation被多轮研究复用；Test又已多次查看。没有外部队列、重复种子或预先保留的全新确认集，不能声称统计稳定的普遍优势。当前更诚实的结论是“v3为既定Validation准则下的候选；已有Test表现支持其平均潜力，同时揭示局部失效”。

[患者曲线](../assessment_analysis/supplementary_research/figures/patient_variability.svg)、[配对与留一敏感性表](../assessment_analysis/supplementary_research/tables/paired_patient_sensitivity.csv)。

## 8. 资源与工程价值

同一A100 40GB、同一预处理后的K008_Week0作3次warm-up/5次同步重复；另记录每个volume的单次推理。计时含准备后张量传输、前向和argmax，不含磁盘读取、预处理、表面指标或绘图，不是端到端延迟。

| 模型 | 参数数 | 参考峰值allocated/GiB | Test逐例单次中位数/s |
|---|---:|---:|---:|
| 2D | 1,942,374 | 0.076 | 0.1401 |
| Standard | 1,412,550 | 0.246 | 0.1740 |
| v1 | 2,366,946 | 0.346 | 0.1144 |
| v2 | 2,366,946 | 0.346 | 0.1073 |
| v3 | 2,366,946 | 0.346 | 0.1666 |
| A0 | 2,366,946 | 0.346 | 0.1666 |
| A1 | 2,366,946 | 0.346 | 0.1666 |

本实现v3推理allocated显存约为2D的4.54倍，不是预设一定5–10倍；Test中位推理延迟约1.19倍。网络较小、输入经过降采样、2D逐切片执行均影响这一结果，不能推广为所有3D模型的成本。v1/v2本次更快，单次顺序运行未消除集群负载和实现内核差异，不据此断言其结构必然更快。

训练日志中的时间混有checkpoint IO；两次v4经历存储耗尽与恢复，不能把其Slurm总时长当纯GPU训练速度比较。参数、reserved VRAM、全部单次耗时及P95在[资源表](../assessment_analysis/assessment_required/tables/resources.csv)和[逐例表](../assessment_analysis/assessment_required/tables/per_case_inference.csv)中。

## 9. 作业要求逐项判定

依据本地`COMP3710_Report_2026_draft_v2.2.pdf`（内部标题Version 2.0 Draft）第5–6页和第13–15页。未独立确认它与最新Blackboard通知完全一致。

| 要求 | 本轮判定 | 证据或剩余事项 |
|---|---|---|
| Hard：3D residual units及deep supervision，下采样HipMRI | 已有实现和运行证据 | v3/A0/A1保留残差与多尺度相加输出；结构档次不等于得分承诺。 |
| Test Dice≥0.7 | 按常见平均口径已有数值支持 | 六个3D模型患者均衡Prostate均值都≥0.7，五前景及六类平均也达标；每个患者或每个端部并非都达标。课程聚合定义仍需确认。2D Prostate 0.6611也未达其Easy路线0.75目标，不把它包装成达标主模型。 |
| 同split标准baseline与逐类Test Dice/IoU | 完成 | 七模型同38/30卷，完整六类表及pooled/患者等权口径。 |
| VRAM、参数、推理时间 | 完成所声明测量范围 | 统一重复参考例+全队列单次分布，不声称端到端或跨GPU公平。 |
| 2D/3D端部、连续性及OAR错误 | 完成描述性研究，保留局限 | 有base/apex、HD95/ASSD、连通域、空缺、面积过渡、OAR混淆；细壁界面及剂量没有专门GT或实验，不能称临床验证。 |
| 3–5例真实FP/FN失败剖析 | 完成本报告五例 | 同位置多面板+整卷指标；MRI叠加图在本地附件，仓库不包含这些PNG，最终课程交付需要保证教师可访问或可重生成。 |
| 严格患者split、对齐、标准化及mask编码 | 既有审计+本轮精确重复核查 | 整数CE/内部one-hot Dice需写清；没有独立专家确认所有GT解剖正确。 |
| 模块化代码及指定交付入口 | 职责已分离，仍需最终打包检查 | 包和工具已有；课程指定`modules.py/dataset.py/train.py/predict.py`入口当前未统一，不能宣称完全满足文件格式。 |
| 最终README、依赖与AI披露 | 部分完成，仍需最终交付审阅 | 项目入口本轮更新披露及结果链接；源码适配、环境和实际命令仍需按最终主模型检查。 |
| feasibility midpoint、PR与教师review回应 | 未核实 | 可行性文档/分批Git不等于已参加check-off或已向upstream提交PR，之后review回应也不能提前完成。 |
| 不提交数据或模型文件 | 本轮遵守 | 发布统计SVG/CSV和代码；MRI、权重、ZIP、原始run JSON保留在本地/远端。 |

“所有指标都有研究”应理解为课程列出的性能、资源与边界主题已覆盖，而不是穷尽所有可用指标。校准、surface Dice容忍阈值、外部泛化、临床剂量/专家修订成本尚未测量，也不是本阶段为了凑齐指标而自动追加的项目。

## 10. 下一步与保留策略

1. **不再把A1作为最终最优模型。** 依预设Validation规则保留v3主候选及Standard基线；A0/A1保留为有正反结果的受控实验。
2. **不继续盲目堆叠训练策略。** 当前结果不支持默认采用这组AdamW/cosine/空间增强配方。若继续研究，先审阅Validation的v3极端前列腺案例，隔离位置/远处连通错误；只有明确可证伪机制后才安排一个变量的实验，不能从本Test失败直接调参。
3. **优先完成最终课程交付。** 整理一致的四入口、README、依赖/预测演示、真实图片访问方式和正确PR；确认老师的Dice聚合定义及midpoint记录。
4. **历史权重先核验再归档。** v2已有完整本地备份及105文件SHA一致证据，远端仅删去49份非best逐轮权重。其他版本不因成绩下降立即删除；先保存best、恢复状态、配置、源码SHA、全部epoch记录、原始/恢复日志和本轮校验清单。统计记录不能替代可复现权重。

## 11. 产物导航与复算

- [课程要求指标与图表目录](../assessment_analysis/assessment_required/)：六类、边界、端部、OAR、资源。
- [补充研究目录](../assessment_analysis/supplementary_research/)：每患者/每例结果、连续性、敏感性与远处FP。
- [原始Validation图片](../comprehensive_audit_20261021/images/validation/)、[原始既有Test图片](../comprehensive_audit_20261021/images/test/)：保持下载原件，不据截图估计数值。
- 原始`metrics.json` SHA-256：`74c298b956b684be3c377e495f70d6deb03445a9f5782663b189c33cdc6dbbc2`。
- 从外围项目目录复算：`python -B 20261017/code/tools/summarize_all_model_audit.py`。脚本先校验原始下载和历史记录，再生成独立分析目录，不改训练与远端冻结评估脚本。

本报告使用AI协助整理代码、统计和文字；真实数字经可复算程序核验，指定图像已实际查看。代码测试、矩阵复现及人工读图均不等于独立临床专家审核。
