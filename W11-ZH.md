# COMP3710 HipMRI — 第 11 周技术复盘、未决问题与文件索引

[English version](W11-EN.md)

> 截至 2026-10-15。证据等级：**代码核实**＝当前实现；**原始日志/独立复算**＝本地检查真实运行记录并复算；**学生记录**＝远程运行/哈希回执、原二进制未在本机打开；**推导**＝由已注明分母计算；**假设**＝尚无因果试验。所有结果均属开发集 Validation；Held-out Test 的 30 个体积未作数组解码或评估。文中相对路径从本文件所在目录起算。`research_archive/` 是本地、Git 忽略的历史资料，不等于提交包。

## Part 1 — 已完成工作、模型演进与实验结果

### 1.1 项目与证据边界

任务是 HipMRI 六类 3D semantic segmentation：0 Background、1 Body、2 Bones、3 Bladder、4 Rectum、5 Prostate，目标是改善前景器官，尤其与前列腺放疗勾画相关的分割。架构只有两族：[StandardUNet3D](recognition/hipmri_3d_improved_unet_hard/modules.py) 和 [ImprovedUNet3D](recognition/hipmri_3d_improved_unet_hard/improved_modules.py)。V1–V4 共用后者；V2 改 CE 权重，V3 续训并保护性选模，V4 成对改变 Prostate overlap loss。它们不是四个新网络。V3 E12 是受保护候选但尚未正式冻结；V4 Treatment 按预定联合规则拒绝，Control E20 亦未自动入选。最终 V3 推理兼容性和 Test 尚未验证。主线依据为[综合实验史](recognition/docs/experiments.md)、[最终报告](recognition/FINAL-REPORT.md)、[代码追溯](recognition/docs/evidence/traceability.md)与[P4-05A](research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md)。

### 1.2 数据处理：真实执行链

以下“检查”区分代码测试、学生真实数据运行和独立日志验证。发现文件不等于解码 MRI，header 审计不等于核实临床标注。

| 序 | 实际步骤、目的与参数 | 实现/核验及界限 |
|---:|---|---|
| 1 | 在 `semantic_MRs/*_LFOV.nii.gz` 与 `semantic_labels_only/*_SEMANTIC.nii.gz` 枚举文件，以患者 ID＋Week 配对；避免错配、重复或缺文件。 | [`dataset.py`: `parse_volume_filename`, `discover_dataset`](recognition/hipmri_3d_improved_unet_hard/dataset.py)；生产发现 38 患者、211 对。此时仅目录/名称发现，不是读取图像数组。 |
| 2 | 同一患者的纵向 Weeks 归为一个独立单位。冻结 split 是 Train 26 患者/143 体积，Validation 6/38，Test 6/30；seed 3710 只记来源，不运行时重新抽签。 | `APPROVED_PATIENTS`、`validate_approved_split` 拒绝交叉患者、未知 ID/周、缺失和计数不符；真实 split 核验见[包 README](recognition/hipmri_3d_improved_unet_hard/README.md)。重复扫描增加体积数，却不增加独立患者数；训练/选模不用 Test。 |
| 3 | 允许 label 0–5，单例可仅出现子集；不重编码语义。MRI/mask 3D shape、effective affine、spacing、orientation 与单位相容性必须对齐。 | [`volume_io.py`: `load_volume_pair`](recognition/hipmri_3d_improved_unet_hard/volume_io.py)。211 对 header 的学生只读审计无 grid 不匹配；MRI 标明 mm，mask 标明 unknown，仅在几何相符时条件推断工作单位，保留原 header 含义。header 一致不是解剖准确性证据。 |
| 4 | 实际 `get_fdata(float32)` 解码一对数组；MRI 值有限，mask 先检验整数及 0–5 再转 `uint8`，记录原 dtype、affine、labels。 | `load_volume_pair`、[`test_volume_io.py`](recognition/hipmri_3d_improved_unet_hard/test_volume_io.py)。K019_Week1 真实载入两数组 `(256,256,144)`；随后 Train/Validation 181 对经预处理审计。不能据此声称 Test 数组已读。 |
| 5 | 训练集患者中位 spacing 的中位候选 `(1.6796875,1.6796875,1.55999755859375)` mm；同一物理输出网格/affine 保留方向和 voxel-edge FOV。MRI 线性插值 `order=1, nearest`；mask 最近邻 `order=0, grid-constant, cval=0`，防小类产生小数 ID 或边界标签丢失。 | [`resampling.py`: `make_target_grid`, `resample_volume_pair`](recognition/hipmri_3d_improved_unet_hard/resampling.py)、[`resampling_config.json`](recognition/hipmri_3d_improved_unet_hard/resampling_config.json)及面部边界合成测试。512 MiB 是四倍估算预分配保护，非峰值内存上限；消失的已有前景类触发错误。spacing 统计为学生训练集审计，未重算最优性。 |
| 6 | 在重采样 MRI 的**原始非零**体素取 0.5/99.5 百分位裁剪，再 `float64` 均值/总体标准差 z-score，输出 `float32`；原零维持零，mask 不参与强度统计。 | [`intensity_normalization.py`](recognition/hipmri_3d_improved_unet_hard/intensity_normalization.py)、[`normalization_config.json`](recognition/hipmri_3d_improved_unet_hard/normalization_config.json)与合成测试。逐体积统计避免跨患者信息泄漏，但不同体积的对比度仍可能变化。 |
| 7 | MRI 与 mask 共用 affine/grid/crop；XYZ 同转 `(2,1,0)` 得 ZYX，MRI tensor `[1,D,H,W]`、整数 mask `[D,H,W]`。 | [`patch_dataset.py`: `HipMRIPatchDataset.__getitem__`](recognition/hipmri_3d_improved_unet_hard/patch_dataset.py)、对应测试与真实 patch 记录；mask 不物化 one-hot，课程 one-hot 建议的符合性待问导师。 |
| 8 | 每个 Train volume 每 epoch 抽一块 `64³`；`SeedSequence(seed,epoch,index)` 使裁块与 DataLoader 顺序分离。0.5 是选择 Prostate 分支概率：选 class-5 真值体素并取含它的合法 crop；另有随机分支与缺类 fallback。无生产数据增强/全体积缓存。 | [`patch_dataset_config.json`](recognition/hipmri_3d_improved_unet_hard/patch_dataset_config.json)，[Job 650736 真实记录复算](recognition/docs/analysis/class-prediction-collapse.md#verified-real-patch-exposure-job-650736)：1430 块、754 Prostate 分支＋676 随机；880/1430 实际含 Prostate，概率 0.5 不是正样本率或完整器官覆盖保证。 |
| 9 | `DataLoader(batch_size=1, shuffle=True, num_workers=0)`；每 epoch 143 个 mini-batch/optimizer steps，10 epoch＝1430 steps。模型/优化器/RNG 按 epoch 原子保存；seed 3710，float32、TF32 关闭、确定性仅 warn-only。 | [`train.py`](recognition/hipmri_3d_improved_unet_hard/train.py)、各版本训练脚本及原始 E10 JSONL；恢复核对配置和 source SHA。一个 epoch 是遍历 143 体积，绝非一个 batch；GPU 位级重现不保证。 |
| 10 | Validation 仅选 6 患者/38 全体积；`64³` sliding windows、stride 32、高侧补齐、重叠 softmax **平均**后 argmax，恢复同一重采样 XYZ grid。 | [`validate.py`: `sliding_window_predict`, `select_validation`](recognition/hipmri_3d_improved_unet_hard/validate.py)；E10 两原始 38/38 日志与 V3/V4 审计。patch 中心裁只用于局部诊断，不能替代全体积结果。 |
| 11 | 6×6 confusion matrix 真值行、预测列，记录 TP、FP、FN、pred/GT 数及 DSC/IoU/Precision/Recall；既算 pooled（先合并体素），也算 volume-macro 和 patient-macro（先合并患者 Weeks）。两边都缺的类为 null 并从 macro 排除；单边缺类记零。 | `validate.py`: `confusion_matrix`, `metrics_from_confusion`, `summarize`；[E10 完整日志](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)、[P4-04C 独立复算](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md)。逐病例图仅举例，不能称 apex/base 测量。 |

### 1.3 架构及训练版本对照

| 项目 | Standard E10 | Improved V1 E10 | Improved V2 E10 | Improved V3 E12 | V4 Control E20 | V4 Treatment E20 |
|---|---|---|---|---|---|---|
| 架构/参数 | StandardUNet3D，5,646,470 | ImprovedUNet3D，5,734,114 | Same as V1 | Same as V1 | Same as V1 | Same as V1 |
| 编解码/3D/skip | 四次下采样、双 3D Conv、GroupNorm(1)/ReLU；transpose Conv＋concat skip | 残差单元用于编码/瓶颈/解码；3D Conv 与 skip；主头＋半/四分辨率辅助头 | Unchanged | Unchanged | Unchanged | Unchanged |
| 输入/输出/通道 | 1→6 logits，base 16 | Same as Standard | Unchanged | Unchanged | Unchanged | Unchanged |
| 主/辅输出 | 主输出 | 主＋2 辅，训练用；推理主输出 | Same as V1 | Same as V1 | Same as V1 | Same as V1 |
| loss | 0.5 CE＋0.5 foreground Soft Dice | 三头同一原 loss，`1:.5:.25 /1.75` | 三头 0.5 weighted CE＋0.5 原 Dice | Same as V2 | Same as V3；alpha .5 对应原 Dice | 仅 Prostate overlap 换 soft Tversky alpha .7/beta .3；weighted CE 不变 |
| 优化 | AdamW，LR .001，WD .00001，betas .9/.999，无已记录 schedule | Unchanged | Unchanged | 恢复 V2 model/optimizer/RNG；LR .001 | 从 V3 E12 同源恢复；LR .001 | 与 Control 同源，配对 crop/训练顺序 |
| 数据/训练 | 143 Train，batch 1，1×64³/体积/epoch；Prostate 分支 .5；无增强；seed 3710 | Unchanged | Unchanged | Unchanged | Unchanged | Unchanged |
| split/Validation | 26/6/6 患者；E10 全体积 38，64³/stride32 | Unchanged | Unchanged | E12 split A/B 训练、C 完整 Validation/保护性选择 | E14/16/18/20 完整 Validation；E20 主终点 | 同 Control |
| 父模型/训练量 | 初训 E1–10，1430 steps | 初训 E1–10，1430 | 初训 E1–10，1430 | V2 E10→E11–12，追加 286，累计 1716 | V3 E12→E13–20，追加 1144/arm，累计 2860 | 与 Control 相同 |
| 实际单变量/限制 | 架构参考 | 残差＋deep supervision 联合改变，不能拆开归因 | class-weighted CE，网络不变 | 续训和预定保护性选择 | 原目标续训，是 V4 配对对照，不自动当赢家 | Prostate overlap alpha .5→.7；其余训练目标保持 |

架构计数和实现见[traceability](recognition/docs/evidence/traceability.md#section-2--model-architecture)；V2–V4 协议见[综合史](recognition/docs/experiments.md)。E20 配对不等于孤立“纯 loss”因果证明：两臂共享父点和顺序，但存在 GPU 数值变异、单 seed 与同一 Validation 反复观察。

### 1.4 实际 loss 方程

令 `p_ic=softmax(logits)_ic`，`y_ic=1[y_i=c]`，`N` 为该 head 上 batch×voxel 总数，`ε=10⁻⁵`。代码在 batch＋空间维求和、对前景 `c=1..5` 均值，缺类仍进入 Soft Dice。普通 `CE=−N⁻¹Σ_i log p_i,y_i`；`D_c=(2Σ_i p_ic y_ic+ε)/(Σ_i p_ic+Σ_i y_ic+ε)`；`L_D=1−(1/5)Σ_{c=1}^5 D_c`。Standard 主头 `L=.5CE+.5L_D`。这是一阶可导概率 loss；下表的 Validation Dice 是 argmax 后硬标签 confusion 指标，不能混用。

V1 每个主/辅 head 分别算上述 `L`，两个辅 logits 用 `trilinear, align_corners=False` 上采样到原目标网格，最终 `L_DS=(L_main+.5L_half+.25L_quarter)/1.75`；CE 与 Dice 日志成分也按相同 head 权重合并。V2 令训练 patch 频率 `f_c`，`w_c=f_c^(−1/2)/Σ_j sqrt(f_j)`，满足 `Σ f_c w_c=1`；`WCE=N⁻¹Σ_i w_{y_i}(−log p_i,y_i)`（**不**按权重和再归一）。六权重依类序为 `1.1775485, 0.6250278, 1.6794010, 2.5227875, 4.5640100, 4.2290216`；来源是 Job 650736 的 1430 patch、374,865,920 voxel。V2 每 head `L=.5WCE+.5L_D`，再同 V1 深监督；普通 CE 仅供日志。V3 完全沿用 V2 目标。

V4 **Control** 直接调用 V2 loss。Treatment 仅将 class 5 的 soft Dice overlap score 换成 `T_5=(TP_s+ε/2)/(TP_s+0.7FP_s+0.3FN_s+ε/2)`，其中 `TP_s=Σp_5y_5`、`FP_s=Σp_5(1−y_5)`、`FN_s=Σ(1−p_5)y_5`。其 overlap loss 是 `L_D+(D_5−T_5)/5`，每头 `L=.5WCE+.5[L_D+(D_5−T_5)/5]`，再按三头深监督组合。[冻结 E20 源码](recognition/hipmri_3d_improved_unet_hard/experiment_prostate_fp_e20.py)确认其余四前景 Dice 和六类 weighted CE 都没换；“Tversky 实验”不能误写为全部 loss 都是 Tversky。alpha 0.7 更惩罚 soft FP，beta 0.3 权衡 FN；是否改善最终 hard-label 指标必须由 Validation 判定。

### 1.5 数值实验与证据

**运行单位。** batch size=1 时一个 mini-batch 带来一次 optimizer step；一个 epoch 有 143 steps，Validation run 不更新参数。下表 `Not recorded` 表示本地可核文档没有相应原值；远程成功与本机复算不能混称。E10 原日志分别仅覆盖 Standard E4–10 和 V1 E2–10，但终态记 10/1430；V2 原 per-volume 日志未在本地。来源：[E10 清单](recognition/docs/evidence/failure-cases.md#original-e10-log-authentication)、[实验史](recognition/docs/experiments.md)、[V4 独立审计](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md)。

| run / checkpoint | 父点；epoch / 追加 / 累计 updates | Train / Val；batch；LR | 训练与 Validation 状态/选择 | GPU/时间与证据等级 |
|---|---|---|---|---|
| Standard E10 (`648790`; Val `649669`) | 初训 10 / 10 / 1430 | 143/38；1；.001→.001，无 schedule | 10 完成；38/38，baseline | A100；已见 E4–10 训练 5146.637 s，Val loop 508.611 s；原 JSONL 复算，二进制缺席 |
| V1 E10 (`649668`; Val `650393`) | 初训 10 / 10 / 1430 | 同上 | 10 完成；38/38；`649912` 原 Val 已取消并替换 | A100；已见 E2–10 训练 6906.426 s，Val loop 458.372 s、作业报告 8m04s；原 JSONL 复算 |
| V2 E10 (`652649`; Val `652948`) | 初训 10 / 10 / 1430 | 同上；weighted CE | 学生记录训练及 38/38 Val 完成，成为 V3 父点 | GPU/整程时间 Not independently verified；远程记录/哈希，per-volume 文件和 checkpoint 本地缺席 |
| V3 E11 A | V2 E10 / +1 / 1573 | 同上 | 143 updates；`validation_pending`，不选模 | 学生认证分段日志；具体 runtime Not recorded |
| V3 E12 B；C Validation | V2 E10 / +2 / 1716 | 同上 | B 143 updates；C 零更新/38 Val；C 选 E12 为 protected candidate | B SHA `bb0d03c7…67660d4`，receipt `fffe08a6…95b2e189da`；哈希回执与 C 日志，payload 本地未开 |
| V4 Control E14/16/18/20 | V3 E12 / +2/+4/+6/+8；2002/2288/2574/2860 | 同上；alpha .5 | E13–20 八 epoch、1144 updates/arm；四次 38/38 Val；E20 非自动赢家 | 8 个 epoch journal、4 Val 完成；P4-04C 独立复算，整程 walltime Not recorded |
| V4 Treatment E14/16/18/20 | V3 E12 / 同 Control | 同上；alpha .7 | 同步配对；E20 三门失败，拒绝 | 同上；E20 checkpoint SHA `11be4077…101113` 仅哈希回执，payload 未本机检视 |

**六类 DSC。** 每格类序固定 `Background, Body, Bones, Bladder, Rectum, Prostate`；六位小数仅显示精度。E10/V2 来源分别为[原始 E10 复算](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)与[学生 V2 记录](recognition/docs/experiments.md#improved-v2--weighted-loss-e10)；V3/V4 来源[P4-04C 的 38 体积独立聚合 JSON](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/independent_results.json)。所有列是 Validation，宏平均不能替代 pooled。

| 模型/时点 | pooled DSC，六类依序 | patient-macro DSC，六类依序 | volume-macro DSC，六类依序 |
|---|---|---|---|
| Standard E10 | .941342, .842790, 0, .622004, 0, 0 | .935937, .835655, 0, .583946, 0, 0 | 完整六类值 Not recorded locally |
| V1 E10 | .976012, .906781, **3.9431×10⁻⁷**, .768230, 0, 0 | .973940, .904107, 0（舍入）, .711025, 0, 0 | 完整六类值 Not recorded locally |
| V2 E10 | .953976, .862141, .362280, .826771, .461378, .445695 | .948549, .854755, .361053, .779908, .439592, .453977 | Not independently verified |
| V3 E12 | .969837, .868735, .476785, .821240, .537022, .426721 | .966336, .866039, .476872, .785414, .529170, .415424 | .970080, .867989, .483219, .732820, .468399, .416253 |
| Control E14 | .978476, .907809, .540260, .832454, .503847, .443944 | .976655, .906884, .537156, .792843, .509126, .448163 | .978497, .906996, .540310, .757735, .467099, .439084 |
| Treatment E14 | .977796, .906782, .532547, .830511, .508864, .464330 | .975945, .905765, .530180, .790622, .518236, .461007 | .977838, .905830, .532225, .755654, .472707, .455346 |
| Control E16 | .989507, .942148, .557011, .840684, .583710, .590097 | .989309, .942529, .550730, .798259, .577954, .585625 | .989420, .941413, .552954, .771764, .530700, .581150 |
| Treatment E16 | .989778, .942325, .552775, .842253, .579458, .587816 | .989709, .943034, .551371, .800183, .574215, .587122 | .989672, .941534, .549010, .775710, .525896, .580254 |
| Control E18 | .986332, .919894, .622998, .786453, .553822, .381516 | .985485, .920189, .623358, .738965, .542708, .388404 | .986237, .919572, .626445, .702082, .508290, .380725 |
| Treatment E18 | .985669, .919387, .621311, .713926, .564680, .418833 | .984909, .919635, .620918, .678479, .553599, .425146 | .985565, .918688, .624138, .632502, .519137, .415446 |
| Control E20 | .989198, .946139, .670668, .843456, .593401, .617164 | .988608, .946059, .665497, .803031, .584171, .616010 | .989099, .945737, .669831, .757965, .537333, .621048 |
| Treatment E20 | .984951, .936111, .676095, .841011, .543693, .650778 | .983738, .935057, .671595, .797808, .535731, .643060 | .984891, .935426, .675757, .757564, .503413, .653441 |

V1 Bones 不是严格零：仅 3 个预测 voxel、其中 2 TP；显示为六位小数时会舍入为零。E10 真正零预测的是 Standard 的 Bones/Rectum/Prostate 和 V1 的 Rectum/Prostate。V2 的全量 per-volume 矩阵、本地 IoU/Precision/Recall 全类数据缺席，不能用其 DSC 倒推所有 confusion counts。V3 与 V4 的每类 TP/FP/FN、pred/GT、Precision/Recall、IoU、覆盖率，以及完整每病例与每患者值详见上述独立 JSON；下表呈现关键 class-5 counts，避免把数千记录粘成研究笔记。

**Prostate pooled confusion 与轨迹。** TP、FP、FN 均为跨 38 体积 pooled voxel，GT 固定 251,012；pred=TP+FP。`TP-positive coverage` 指同时有真实正例和预测 TP 的体积数，与“曾预测正类”的覆盖数不同。V3 JSON 未给此单项，V3 选择/审计记录另确认 classes 2–5 为 38/38。来源同上 P4-04C JSON；E10 来自[原始日志复算](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)。Precision/Recall 数值按 JSON 原值显示。

| 时点 | TP | FP | FN | pred | GT | pooled Precision / Recall | TP-positive volumes |
|---|---:|---:|---:|---:|---:|---|---:|
| Standard / V1 E10 | 0 | 0 | 251,012 | 0 | 251,012 | 0 / 0（预测无正类） | 0/38 |
| V3 E12 | 175,508 | 396,069 | 75,504 | 571,577 | 251,012 | .307059 / .699202 | 38/38，据选择规则 |
| Control E14 | 179,248 | 377,266 | 71,764 | 556,514 | 251,012 | .322091 / .714101 | 38/38 |
| Treatment E14 | 172,275 | 318,750 | 78,737 | 491,025 | 251,012 | .350848 / .686322 | 38/38 |
| Control E16 | 205,328 | 239,573 | 45,684 | 444,901 | 251,012 | .461514 / .818001 | 38/38 |
| Treatment E16 | 207,781 | 248,166 | 43,231 | 455,947 | 251,012 | .455713 / .827773 | 38/38 |
| Control E18 | 238,266 | 759,770 | 12,746 | 998,036 | 251,012 | .238735 / .949222 | 38/38 |
| Treatment E18 | 231,042 | 621,211 | 19,970 | 852,253 | 251,012 | .271096 / .920442 | 38/38 |
| Control E20 | 216,950 | 235,093 | 34,062 | 452,043 | 251,012 | .479932 / .864301 | 38/38 |
| Treatment E20 | 197,842 | 159,163 | 53,170 | 357,005 | 251,012 | .554172 / .788177 | 38/38 |

**V3 E12 与 E20 的六类完整 pooled 错误构成。** 各行由[P4-04C 独立 JSON](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/independent_results.json)按 38 体积合并后的 confusion counts 提取。`Precision=TP/pred`、`Recall=TP/GT`、`IoU=TP/(TP+FP+FN)`；`pred=TP+FP`、`GT=TP+FN`。三组共享 GT，依类序为 `195612026,105931724,10144275,1742051,500065,251012` voxels。其它 E14/E16/E18 每类同字段及所有病例矩阵在链接 JSON 中，以上 DSC 轨迹已涵盖这三轮全部六类，不将 per-case 结果充作全量。

| 时点/类 | IoU | Precision | Recall | TP | FP | FN | pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 Background | .941441 | .954112 | .986089 | 192,890,936 | 9,277,090 | 2,721,090 | 202,168,026 |
| V3 Body | .767933 | .925826 | .818276 | 86,681,423 | 6,944,622 | 19,250,301 | 93,626,045 |
| V3 Bones | .313012 | .390753 | .611396 | 6,202,165 | 9,670,162 | 3,942,110 | 15,872,327 |
| V3 Bladder | .696698 | .910514 | .747909 | 1,302,896 | 128,050 | 439,155 | 1,430,946 |
| V3 Rectum | .367075 | .530644 | .543555 | 271,813 | 240,419 | 228,252 | 512,232 |
| V3 Prostate | .271230 | .307059 | .699202 | 175,508 | 396,069 | 75,504 | 571,577 |
| E20 Control Background | .978628 | .988196 | .990203 | 193,695,567 | 2,313,638 | 1,916,459 | 196,009,205 |
| E20 Control Body | .897784 | .944526 | .947758 | 100,397,607 | 5,896,511 | 5,534,117 | 106,294,118 |
| E20 Control Bones | .504515 | .722625 | .625682 | 6,347,090 | 2,436,294 | 3,797,185 | 8,783,384 |
| E20 Control Bladder | .729290 | .798453 | .893835 | 1,557,107 | 393,048 | 184,944 | 1,950,155 |
| E20 Control Rectum | .421870 | .511031 | .707428 | 353,760 | 338,488 | 146,305 | 692,248 |
| E20 Control Prostate | .446303 | .479932 | .864301 | 216,950 | 235,093 | 34,062 | 452,043 |
| E20 Treatment Background | .970349 | .985721 | .984183 | 192,518,020 | 2,788,773 | 3,094,006 | 195,306,793 |
| E20 Treatment Body | .879896 | .937336 | .934890 | 99,034,490 | 6,620,807 | 6,897,234 | 105,655,297 |
| E20 Treatment Bones | .510682 | .685131 | .667294 | 6,769,211 | 3,110,953 | 3,375,064 | 9,880,164 |
| E20 Treatment Bladder | .725642 | .784288 | .906578 | 1,579,305 | 434,374 | 162,746 | 2,013,679 |
| E20 Treatment Rectum | .373337 | .412250 | .798190 | 399,147 | 569,068 | 100,918 | 968,215 |
| E20 Treatment Prostate | .482336 | .554172 | .788177 | 197,842 | 159,163 | 53,170 | 357,005 |

**E10 六类 pooled 体素证据。** `FP=pred−TP`、`FN=GT−TP` 可直接复算；表中 coverage 为**有预测**的体积数，不以 DSC>0 取代。来源[E10 原始矩阵](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)。

| 类 | GT | Standard pred / TP / coverage | V1 pred / TP / coverage |
|---|---:|---|---|
| Background | 195,612,026 | 204,051,044 / 188,109,738 / 38 | 197,074,176 / 191,633,198 / 38 |
| Body | 105,931,724 | 108,979,831 / 90,562,636 / 38 | 115,514,860 / 100,401,777 / 38 |
| Bones | 10,144,275 | 0 / 0 / 0 | 3 / 2 / 2 |
| Bladder | 1,742,051 | 1,150,278 / 899,520 / 38 | 1,592,114 / 1,280,703 / 38 |
| Rectum | 500,065 | 0 / 0 / 0 | 0 / 0 / 0 |
| Prostate | 251,012 | 0 / 0 / 0 | 0 / 0 / 0 |

**E10 IoU 与 V2 四类 precision/recall。** E10 IoU 由完整原日志复算；类序同上。Standard 六类 `0.889184,0.728295,0,0.451383,0,0`；V1 `0.953148,0.829460,1.971555×10⁻⁷,0.623680,0,0`。[E10 原始证据](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)。V2 原始 per-volume 数据本地缺席，但[学生 pooled confusion 汇编](recognition/docs/analysis/class-prediction-collapse.md#current-optimisation-decision---12-october-2026)给：Bones precision/recall `.528495/.275602`，FP/FN `2,494,303/7,348,494`；Bladder `.788611/.868812`，`405,702/228,536`；Rectum `.597848/.375633`，`126,354/312,224`；Prostate `.340148/.646212`，`314,665/88,805`。这四类 TP-positive coverage 都记录为 38/38；其余两类 precision/recall/counts `Not independently verified`。

**训练轨迹（仅真实原始 E10 epoch 事件）。** 每项为 `CE / Soft Dice loss / total`，且各自目标不同的阶段不能直接比较 total 大小；完整秒数和 step 见[证据表](recognition/docs/evidence/failure-cases.md#recorded-training-loss-trajectory)。Standard：E4 `.752481/.830941/.791711`，E5 `.667956/.823227/.745591`，E6 `.721923/.802126/.762024`，E7 `.651950/.781117/.716534`，E8 `.646269/.772173/.709221`，E9 `.687573/.760931/.724252`，E10 `.628552/.769903/.699227`。V1：E2 `.802051/.818914/.810483`，E3 `.655318/.798339/.726828`，E4 `.618251/.775927/.697089`，E5 `.515769/.773786/.644778`，E6 `.622856/.764036/.693446`，E7 `.513024/.735074/.624049`，E8 `.504781/.724838/.614809`，E9 `.516944/.705877/.611411`，E10 `.507582/.714722/.611152`。两模型 E9→10 Soft Dice loss 都上升，不能由 Train total 下降推断收敛。V2/V3/V4 相应完整逐 epoch CE/Dice/total 在本地汇编未独立提取，**Not independently verified**。

**资源与抽样。** [Final Report §5.5](recognition/FINAL-REPORT.md#55-computational-and-operational-considerations)给 E10 resumed-run GPU peak allocated Standard/V1 `487,463,424 / 487,471,616` bytes；reserved `813,694,976 / 836,763,648` bytes。38 体积 Validation loop `508.611 / 458.372` s；其中 sliding-window 函数求和 `315.114 / 277.928` s，不等于端到端单病例 latency。Job 650490 的三例配对延迟/显存是有限样本，不能宣称架构效率优势。Job 650736 的 1430 patch 共 `374,865,920` voxels；Background＋Body 为 `84.20%`，Bones 有正类 patch `86.01%`，Prostate 有正类 patch `61.54%`，说明全无接触并非早期遗漏类的充分解释；不是训练输入文件历史字节不变的证明。[曝光完整分布](recognition/docs/analysis/class-prediction-collapse.md#verified-real-patch-exposure-job-650736)列全部类和患者分布。

**冻结 E20 八门。** 下表用 patient-macro 的 Dice 门、pooled 的 FP/Recall 门，不交叉替代。来源[预定配置](recognition/hipmri_3d_improved_unet_hard/prostate_fp_e20_config.json)与[P4-04C 独立规则复算](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md)。

| 门 | 观测 / 门限 | 结果 |
|---|---|---|
| Prostate patient-macro DSC 增益 | .027050 / ≥.030000 | FAIL |
| Prostate pooled FP 减少比例 | .322979 / ≥.150000 | PASS |
| Prostate pooled Recall 降幅 | .076124 / ≤.050000 | FAIL |
| 其它类 patient-macro DSC 最大降幅 | .048440 Rectum / ≤.020000 | FAIL |
| Prostate 不退步患者 | 4/6 / ≥4/6 | PASS |
| classes 2–5 TP-positive volume 最少覆盖 | 38/38 / 38/38 | PASS |
| Treatment Prostate patient-macro 高于 V3 | +.227635 / >0 | PASS |
| Bones＋Rectum＋Prostate patient-macro 总和高于 V3 | +.428920 / >0 | PASS |

E20 六位患者 Prostate Dice 差（Treatment−Control）：B040 `+.028194`，J026 `+.051247`，K008 `−.022817`，K018 `−.019811`，M030 `+.002517`，S022 `+.122968`。[P4-04C](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md)。E18 两臂 pooled Prostate FP 尤其高（759,770/621,211），E20 反弹式变化提示单次 epoch 优势不稳定，不能据训练 loss 或 E20 单一 Dice 声称普遍提升。

### 1.6 已观察结论与来源冲突

Baseline 与 V1 的 E10 全体积遗漏 Rectum/Prostate；V1 在 Body/Bladder 提升却仍未解决小类，S022 的 Bladder 从 `.6243` 退到 `.2490`。V2 weighted CE 在学生记录的 38-volume Validation 恢复 Bones/Rectum/Prostate overlap，却使若干大类 Dice 下降且 Prostate precision 仅 `.340148`、recall `.646212`。[诊断](recognition/docs/analysis/class-prediction-collapse.md)的 Job 650924 只在两 Train patch、一 Validation crop 测概率/类别竞争，非参数梯度、非完整 Validation，也不能确定因果。V3 E12 保护性选择的 minority patient-macro 综合改善不代表所有类都改善：其 pooled Prostate `.426721` 低于 V2 `.445695`。V4 E20 Treatment 更少 FP 且更高 Prostate precision/DSC，却同时损失 recall 和 Rectum，三门失败，故按原联合规则拒绝。V3 仍只是 protected candidate。

**不能静默调和的证据差异/缺口：** Standard 原 full Validation 使用 `standard_unet_last.pt`，配对 case JSON 用 `standard_unet_epoch10.pt`；旧 full Val 不记 checkpoint binary hash，不能证明两名文件同权重。E10 原训练日志缺 Standard E1–3/V1 E1，不能凭最终计数补造逐 epoch 曲线。V2 原始 per-volume 矩阵和 V3/V4 binary payload 未在本地包中；P4-04D 的 Rangpur 哈希回执建立文件身份，**不**等于本机检查 optimizer/RNG/tensor。P4-04E 补齐缺失 exporter 的源哈希和测试来源，但 checkpoint payload 仍未检视。[P4-04D](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/P4-04D_provenance_20261014/P4-04D_PROVENANCE_ADDENDUM.md)、[P4-04E](research_archive/08_validation_and_independent_audit/COMP3710_P4-04E_20261014/V4_EVIDENCE_CLOSURE.md)、[P4-05A](research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md)。上述限制影响复现强度，不能用 Test 补开发证据。

## Part 2 — 现存疑点、优化方向与助教讨论问题

### 2.1 未决问题总览

**A 训练与优化。** E10 原日志不完整，且 Train loss 曲线非单调；V4 E14→E20 的 Prostate 表现亦不单调。10、12 或 20 epoch 均无“已收敛”证明。继续 V3 E12 的科学理由应来自预声明的患者级 Validation 增益、风险与计算预算，而非 GPU 尚有空时；继续使用已看过的六位患者会进一步增加选择偏差。需要决定早停目标、patience、最低临床/课程相关改善、何时停止看 Validation；单 seed 和 6 患者不能稳估泛化。LR .001 无调度的对照，先查每 epoch 学习率、loss 组成与波动。加 batch 改变每 epoch 更新数、梯度方差、内存、归一化采样；加 epoch 在现 batch 下每次增加 143 steps，两者不能互换。已知 E10 训练与 Val 耗时、A100 显存只是部分资源证据；V3/V4 端到端预算应从原 journals/Slurm 汇总。

**B 数据/预处理。** 26/6/6 独立患者比按 211 volume 随机分更能防纵向泄漏，但小样本的 patient-macro 不确定性高；143 Train 体积不是 143 个独立人。全部 Train volume 含六类，真实 patch Prostate 阳性率 61.54%、Bones 86.01%，所以“完全没见到”不成立；但 patch 内器官比例、边界和患者覆盖不均仍可能限制学习。约 1.68×1.68×1.56 mm 重采样、`64³` patch 的细节和上下文取舍未做消融。增强、patch 大小、前景分支概率、患者均衡抽样都未有控制结果；先用 Train/既有 Validation 只读分析 class counts、合法几何、切片覆盖、标注定义与低质病例，并保护 Test 标签。mask header unknown 的条件推断须保留来源说明；不应凭 image/mask header 对齐推断标注正确。

**C 架构。** Standard↔V1 同时改变残差单元和 deep supervision，故两者边际作用无法从现实验拆分；5.65M/5.73M 参数在 26 患者上是否合适未知。V2 同架构的小类恢复提示容量不足并非唯一解释，但不能排除局部上下文不足。attention、多尺度和更大 patch 有额外计算/过拟合风险；先检查训练/验证错例、soft probability 与边界距离，再决定架构消融。

**D 目标函数。** 早期遗漏类伴随很低/零 hard prediction；Job 650736 显示确有 patch exposure，Job 650924 显示选定 patch 的目标概率被竞争类压过，但不能量出 CE、Dice 对参数的梯度、也不能指认 loss 是唯一根因。V2 的 `f^-1/2` weighted CE 改变小类相对惩罚并出现恢复，同时 Prostate pred 476,872 > GT 251,012（V2 学生记录），precision/recall 出现 trade-off。V4 将 class-5 overlap 中 FP 系数 .5→.7，E20 减 FP 32.30%，又掉 Recall .076124 与 Rectum；系数调节须同时保护多类和患者。下一步若研究 loss，先在不更新参数的 Train-only 固定 patch 上测各 loss component 的共享参数梯度 norm/方向及 class-conditioned logits；再只改变一个系数，预声明评价。概率高低本身不是梯度强弱。

**E 评估与科学有效性。** Prostate DSC 平衡 FP/FN，但 radiotherapy 的 under/over-contouring 可能不对称；同时看 Recall、Precision、体积误差、每患者及其它器官。Pooled 按 voxel/多周权重，patient-macro 每患者一票，因此不同数值是不同问题，不是错误。背景和 Body 不应吞没关注小器官的主选模分数；但权重应和导师/课程标准一致。现有 4 组刻意选取 case、少量 Z 切片无法证明 apex/base、边界或 population failure rate；surface Dice、HD95/ASSD、连续切片/专家评审需定义 spacing、异常值规则与病例数后才能作为新证据。反复用六患者调权重、续训和选模使 Validation 渐失独立性；freeze 前必须锁定患者级标准、错误审查范围、V3 checkpoint/推理路径/前处理/版本/输出目录，之后才考虑一次最终 Test。

**F 工程/复现。** V3 C 原 journal 和学生哈希回执定位 E12，但 `selection.json` 字段与原 checkpoint model/optimizer/RNG 在本机未独立打开；原 [`predict.py`](recognition/hipmri_3d_improved_unet_hard/predict.py) Improved 分支只接受 E10 加载语义。写作时工作区另有未入 Git 的 [`predict_v3.py`](recognition/hipmri_3d_improved_unet_hard/predict_v3.py) 和[合成测试](recognition/hipmri_3d_improved_unet_hard/test_prediction_v3.py)：源码显示其提供受保护 V3 E12 adapter 和 preflight，但尚无原 checkpoint/真实 MRI 的远程接受证据，故不能称最终推理兼容性已验证。需读原 payload/receipt、严格 source hash，核验 strict state、配置、六标签、affine/spacing、64³/stride32/mean-softmax parity、重复性、故障时拒绝。E10 原日志及 V4 journals/ZIP 可以复算，V2 原 per-volume 证据和部分远程资源回执仍缺。GPU 内存记录仅涵盖特定运行，环境/NiBabel/SciPy/PyTorch/CUDA 版本与路径需要冻结。P4-05A 明确 **NOT READY TO FREEZE**；课程是否要求 one-hot、官方指标/报告附件/导师 check-off 应问助教，不把文档中的说法当成已确认评分细则。

### 2.2 假设—证据—辨别实验

以下均是**建议调查**，未执行；优先级按先修复可信测量、再改模型。所有实数据调查只限 Train/既有 Validation，不能读 Test。

| 类别/问题 | 支持证据 | 可能原因 | 反证/不确定性 | 区分性诊断 | 可得到的信息 | 风险/成本 | 优先 |
|---|---|---|---|---|---|---|---|
| 工程：V3 无最终推理路径 | P4-05A；E10 `predict.py` loader | checkpoint schema/source guard 不兼容 | V3 C 阶段内推理已成功 | CPU 只读 payload/receipt strict load；合成＋批准的旧 Val parity | 分离工程错误与模型质量 | 低算力、需远程原件 | Critical |
| 评估：患者差异被 pooled 掩盖 | S022 Bladder 退步；E20 K008/K018 退步 | longitudinal volume 权重、解剖异质 | 6 人 CI 很宽 | 既有 38 例 patient table、leave-one-patient-out 敏感性 | 主指标是否稳健 | 低；重复看 Val 风险 | Critical |
| loss：小类目标信号弱或竞争 | E10 漏检；V2 weighted CE 恢复 | CE/Dice 梯度方向或幅度 | 650736 有曝光；650924 无梯度 | Train-only 固定 patch 分别 backward，不 step，比较各 component 梯度 | 区分 objective 机制与单纯曝光 | 中；不可误作历史梯度 | High |
| sampling：有效 organ context 不足 | 64³；Rectum 阳性 777/1430；小碎片 | crop 边界截断/患者偏倚 | Prostate 分支保证至少 1 voxel，Bones 曝光高 | Train patch per-patient 器官占比、截断/空间覆盖审计 | 是否值得调 sampler/patch | 低–中；新 crop 实验贵 | High |
| 优化：E12/20 未稳 | E18 FP 激增；E10 loss 非单调 | LR/随机性/过拟合 | 无多 seed、无完整 V2 epoch 曲线 | 先汇总既有 epoch loss/Val、LR/step，预定停止标准后受控续训 | 区分不稳与系统趋势 | A100 高；Val 过拟合 | High |
| 数据：spacing/标注细节 | resampling＋mask unit unknown | 小结构被插值/标注不一致 | 211 header 配对无错，前景未消失 | Train/Val 原图与重采样边界配对审查、标签协议核验 | 判定是否数据治理先行 | 专家时间；不能改 frozen 真值 | High |
| 架构：容量/上下文不足 | Prostate 仍低；64³ | 感受野/容量 | 同架构 V2 恢复，残差/DS 未消融 | 完成以上诊断后单变量 architecture/patch 对照 | 架构净效应 | GPU 与多重比较高 | Later |
| loss：FP↔Recall | E20 FP −32.30% 但 Recall −.076124、Rectum −.048440 | alpha 促保守预测、类别竞争 | E14/E16 效应不一致 | 固定模型的 Validation 概率校准/阈值诊断；后续单 alpha 实验须预注册 | 分离阈值与训练目标影响 | 中；调 Val 风险 | High |
| 边界：apex/base 病例 | 仅稀疏 Z panel | 局部位置或锯齿误差 | 未标识真实解剖 apex/base | 专家命名切片、HD95/ASSD 和表面可视化预定义审查 | 临床相关错误性质 | 中；专家与 spacing 规则 | Later |

### 2.3 下一步选择框架

| 方向 | 已知/缺证；只改一个变量 | 支持或否定标准 | 成本/时机 |
|---|---|---|---|
| 数据/标注审计 | header/label 范围正确；临床轮廓未审。固定模型，仅查 Train/既有 Val 标注与低质病例 | 发现可重复错标/异常 spacing 才考虑数据纠错；若一致则减弱该假设 | 专家时间；**现在** |
| 改 patch sampling | 已有 Prostate 61.54%、Bones 86.01% 阳性；缺解剖完整性。变分支概率或患者权重一次 | 预定 patient-macro 小类与覆盖率提升且无损失；否则否定 | 重训；**诊断后** |
| 数据增强 | 当前无增强，泛化未知。仅引入一组物理合理变换 | 独立患者指标稳定增益、无几何/标签破坏；否则退回 | 中高 GPU；**稍后** |
| LR/调度 | .001 固定，E18 波动；缺 LR ablation。只变 LR 或 schedule | 多 checkpoint/患者稳定、预定指标增益；否则否定 | GPU 高；**先审轨迹** |
| 延长 epoch | V3 已 +2、V4 两臂 +8 且不单调。只变预算并预定停机/选择 | 持续多患者改善且无其它类损害；否则停 | GPU/Val 选择偏差高；**不立即做** |
| batch size | 1 受 3D patch/内存制约；无较大 batch 实测。改变时要保持或显式调整 updates | 稳定性/吞吐/患者结果改善；若仅省时无质量改善不当新科学结果 | 显存高，混杂更新数；**稍后** |
| CE/Dice/Tversky 系数 | V2 恢复，V4 trade-off；先测 Train-only component gradients，再单系数预声明 | FP/Recall/其它类联合门通过；否则否定 | 中高、反复 Val 风险；**诊断后** |
| 架构 | V1 联合改变，无法隔离；同架构 V2 进步。单组件消融优先于 attention | 单变量在患者级且资源受控的增益；否则无需复杂化 | 高；**后置** |
| 选模规则 | V3 protected、V4 联合门合理但未正式冻结。明确课程/临床权重 | 固定后具有可复算、对病例/类别稳健的选择 | 低算力；**现在** |
| 诊断/评价 | 已有 confusion、四例 panel；缺 surface、CI/患者敏感性、apex/base | 指出可重复失败模式并分辨 FP/FN 边界；否则不扩大主张 | 低–中；**现在** |

**调查顺序：** ① P4-05A 工程缺口与 receipt/payload 只读核验；② 固定现有 38 例的患者级、class-wise 错例/不确定性和课程选模要求；③ Train patch/标注/spacing 审计；④ 固定 Train patch 的 CE/Dice 梯度诊断；⑤ 若仍有明确假设，经导师讨论后单变量、预声明、预算受限的实验。此处不启动训练、不选新 checkpoint、不冻结模型。

### 2.4 助教会议：先问的 10 个问题

每项给中文疑点、可直接使用的专业英文问法、关键证据和将影响的决定。若会议短，先问 1–5。

| 优先 | 主题与中文疑点 | English question（括号内为口语短版） | 为什么问／手头证据／影响决定 |
|---|---|---|---|
| Critical 1 | E12 是否已有足够续训证据？ | “Given the non-monotonic E10 loss and the E14–E20 validation trajectory, what evidence would justify any further training from the protected V3 E12 checkpoint?” (“Would more epochs be justified?”) | E18 FP 激增；决定是否停止训练或预设额外 budget。 |
| Critical 2 | 小类问题先查数据/采样还是 loss？ | “With positive prostate patches in 61.54% of reconstructed training crops and V2 recovery after weighted CE, which diagnostic would best separate sampling from objective effects?” | 650736 与 V2；决定 Train-only audit/gradient 顺序。 |
| Critical 3 | FP–Recall 平衡如何评价？ | “Would you accept the E20 treatment’s 32.30% pooled prostate FP reduction when pooled recall falls by 0.076 and rectum patient-macro Dice falls by 0.048?” | V4 三门失败；确认联合标准/临床意义。 |
| Critical 4 | V3 E12 保护性选择是否足够？ | “Is the V3 E12 patient-macro minority-class selection rule adequate for this assessment, given six reused validation patients and a lower pooled prostate Dice than V2?” | V3 E12 vs V2；决定是否仅保留候选或需更多分析。 |
| Critical 5 | Freeze/Test 边界 | “What evidence and approval do you expect before we freeze the exact V3 checkpoint and run the held-out Test once?” | P4-05A NOT READY，E10 driver 不兼容；决定冻结收据与 Test 时点。 |
| High 6 | 官方聚合和评分 | “Should the report’s primary metric be pooled, volume-macro or patient-macro Dice, and are there course-specific class or patient weighting requirements?” | 三种值不同；决定主表/选模规则。 |
| High 7 | 错例的解剖评价 | “For the prostate and rectum errors, would per-patient precision/recall plus surface distance or expert-reviewed apex/base slices be the most useful evidence?” | 当前图只有稀疏 Z levels；决定追加诊断。 |
| High 8 | 早停规则 | “How would you predefine early stopping here without repeatedly tuning to the same six validation patients?” | E10/V4 非单调；决定 patience、最小改善、看 Val 次数。 |
| High 9 | 作业科研深度 | “Do the Baseline–V4 controls, independent E20 audit and explicit rejected treatment meet the expected experimental depth, or is a specific additional comparison required?” | 已有对照和失败决策；询问**要求**，不假定评分结论。 |
| High 10 | one-hot 与提交 | “Is using integer masks with cross-entropy and soft-Dice acceptable under the assignment’s one-hot guidance, and which exact figures or evidence must accompany the final report?” | 当前实现 long mask，报告图有外部资产；决定提交准备。 |

**补充可选技术问题**（同样是未决，不视作课程要求已确认）：

| 优先/主题 | 中文问题与 English question | 证据及将影响的决定 |
|---|---|---|
| Optional／梯度 | 650924 只能测概率；“Would a no-update, fixed-training-patch CE-versus-Dice gradient audit be a useful next diagnostic?” | 判定是否先测目标函数信号。 |
| Optional／纵向患者 | 体积数高估独立样本；“Should longitudinal weeks be weighted equally within each patient or should every volume count equally during training?” | 26 Train 患者/143 扫描；抽样设计。 |
| Optional／spacing/patch | “Is the training-only median spacing and a 64-cube patch defensible for prostate boundaries, or should we audit context before changing either?” | 固定 spacing 和 patch；确定审计范围。 |
| Optional／小样本不确定性 | “Would patient-level bootstrap or leave-one-patient-out sensitivity be appropriate with only six validation patients?” | E20 两患者退步；不确定性报告。 |
| Optional／增强 | “Which physically plausible MRI augmentations would you permit without corrupting label geometry?” | 目前无增强；下一单变量方案。 |
| Optional／架构消融 | “Would you value a residual-only or deep-supervision-only ablation before considering attention?” | V1 两组件同时改；决定是否值得 GPU。 |
| Optional／Test 一次性 | “After protocol freeze, should the held-out Test be evaluated exactly once, and how should we report an unexpected failure without retuning?” | 保护独立评估；最终程序。 |

## Part 3 — 完整项目文件树与文件功能索引

以下文件树由写作时的实际文件系统生成；对应的英文注释版本见 [W11-EN.md](W11-EN.md)。`research_archive` 为本地历史证据，不能与 Git-tracked active source 混淆；`W11-ZH.md` 与 `W11-EN.md` 是本次的两份成品。排除 `.git` 内部、虚拟环境、`__pycache__`、`.pytest_cache` 与生成的依赖缓存。未查实单文件作用时明确写 Purpose not verified / 用途待核实。

### 3.1 中文注释文件树

```text
Final project/
├── .vscode/
│   └── settings.json  # VS Code 工作区设置
├── recognition/
│   ├── docs/
│   │   ├── analysis/
│   │   │   └── class-prediction-collapse.md  # E10 漏检、patch 与概率分析
│   │   ├── evidence/
│   │   │   ├── failure-cases.md  # E10 原始日志与配对错例证据
│   │   │   └── traceability.md  # 报告至代码证据索引
│   │   ├── experiments.md  # Baseline–V4 综合实验史
│   │   └── FEASIBILITY_REPORT.md  # 项目可行性与剩余要求
│   ├── figures/
│   │   └── e10_recorded_training_losses.png  # 真实 E10 epoch 损失图
│   ├── hipmri_3d_improved_unet_hard/
│   │   ├── diagnostics/
│   │   │   ├── __init__.py  # 诊断包标记
│   │   │   ├── checkpoint_probabilities.py  # 固定 patch 的 checkpoint 概率探针
│   │   │   ├── class_collapse.py  # Train patch 曝光重建
│   │   │   ├── test_checkpoint_probabilities.py  # 合成测试：概率 probe
│   │   │   └── test_class_collapse.py  # 合成测试：patch exposure
│   │   ├── __init__.py  # Python 包标记
│   │   ├── ai.md  # AI 使用与历史执行来源
│   │   ├── benchmark_k019_preprocessing_cache.py  # K019 缓存基准，非生产缓存
│   │   ├── dataset.py  # MRI/mask 发现与冻结患者划分
│   │   ├── experiment_prostate_fp.py  # E20 前 V4 历史草稿程序
│   │   ├── experiment_prostate_fp_e20.py  # 冻结 V4 配对实验与决策
│   │   ├── improved_modules.py  # 共用残差与深监督 ImprovedUNet3D
│   │   ├── improved_v2_config.json  # V2 曝光计数与 CE 类别权重
│   │   ├── improved_v3_config.json  # V3 续训与选模阈值
│   │   ├── intensity_normalization.py  # MRI 非零百分位与 z-score 规则
│   │   ├── modules.py  # StandardUNet3D 网络
│   │   ├── normalization_config.json  # MRI 百分位及标准化设置
│   │   ├── patch_dataset.py  # 对齐的确定性 patch 抽取
│   │   ├── patch_dataset_config.json  # 64 立方裁块与前景抽样规则
│   │   ├── predict.py  # E10 checkpoint 预测、指标及图像
│   │   ├── predict_v3.py  # 新增 V3 E12 受保护推理适配器；远程验收待完成
│   │   ├── profile_pipeline.py  # 预处理吞吐与内存分析器
│   │   ├── prostate_fp_config.json  # 历史 V4 草稿设置
│   │   ├── prostate_fp_e20_config.json  # 冻结 V4 处理与验收门限
│   │   ├── README.md  # 五版本实现与证据指南
│   │   ├── resampling.py  # 物理网格 MRI/mask 重采样
│   │   ├── resampling_config.json  # 目标 spacing／插值／分配规则
│   │   ├── run_improved_v3_split.py  # V3 A/B 训练与 C 选模调度
│   │   ├── smoke_standard_unet.py  # Standard GPU 冒烟检查
│   │   ├── submit_v3_split.sh  # V3 Slurm 分段提交脚本
│   │   ├── test_dataset.py  # 合成测试：dataset discovery/分段
│   │   ├── test_improved_integration.py  # 合成测试：Improved integration
│   │   ├── test_improved_unet.py  # 合成测试：Improved network
│   │   ├── test_improved_v2.py  # 合成测试：V2 objective/训练
│   │   ├── test_improved_v3.py  # 合成测试：V3 continuation
│   │   ├── test_improved_v3_split.py  # 合成测试：V3 stage orchestration
│   │   ├── test_intensity_normalization.py  # 合成测试：MRI normalisation
│   │   ├── test_patch_dataset.py  # 合成测试：crop sampling
│   │   ├── test_prediction.py  # 合成测试：E10 预测
│   │   ├── test_prediction_v3.py  # 合成测试：V3 推理 admission
│   │   ├── test_prostate_fp_e20.py  # 合成测试：V4 E20 controls
│   │   ├── test_resampling.py  # 合成测试：grid interpolation
│   │   ├── test_standard_unet.py  # 合成测试：Standard network
│   │   ├── test_training.py  # 合成测试：Standard 训练/损失
│   │   ├── test_validation.py  # 合成测试：full-volume metrics
│   │   ├── test_volume_io.py  # 合成测试：NIfTI loading/geometry
│   │   ├── train.py  # Standard E10 训练与 checkpoint
│   │   ├── train_improved.py  # Improved V1 训练与辅助损失
│   │   ├── train_improved_v2.py  # V2 加权 CE 训练与状态校验
│   │   ├── train_improved_v3.py  # V3 状态恢复续训
│   │   ├── validate.py  # 滑窗推理与混淆矩阵指标
│   │   ├── validate_improved.py  # Improved V1 验证适配器
│   │   ├── validate_improved_v2.py  # V2 验证适配器与严格校验
│   │   └── volume_io.py  # 配对 NIfTI 数组及几何校验
│   ├── .gitattributes  # recognition 的 Git 属性规则
│   ├── FINAL-REPORT.md  # 技术报告与 验证 解释
│   ├── README.md  # 分割项目导航与当前决策
│   └── requirements.txt  # Python 运行依赖版本范围
├── research_archive/
│   ├── 07_model_development_snapshots/
│   │   ├── P5-08_superseded_active_docs_2026-10-15/
│   │   │   ├── recognition/
│   │   │   │   ├── CLASS_COLLAPSE_DIAGNOSIS.md  # 历史开发快照：类别 漏检 DIAGNOSIS
│   │   │   │   ├── Failiure-analysis.md  # 历史开发快照：Failiure 分析
│   │   │   │   ├── FAILURE_CASE_EVIDENCE.md  # 历史开发快照：失败 病例 证据
│   │   │   │   ├── IMPROVED_V3_EXPERIMENT.md  # 历史开发快照：IMPROVED V3 实验
│   │   │   │   ├── IMPROVED_V3_SPLIT_EXECUTION.md  # 历史开发快照：IMPROVED V3 分段 执行
│   │   │   │   └── REPORT_CODE_TRACEABILITY.md  # 历史开发快照：报告 CODE TRACEABILITY
│   │   │   ├── PROVENANCE.md  # 历史开发快照：来源
│   │   │   └── summary.md  # 历史开发快照：摘要
│   │   └── v4_e14_pre_e20_draft_2026-10-13/
│   │       ├── experiment_prostate_fp.py  # 历史开发快照脚本：实验 prostate fp
│   │       ├── prostate_fp_config.json  # 历史开发快照结构化记录／清单：prostate fp config
│   │       └── PROVENANCE.md  # 历史开发快照：来源
│   ├── 08_validation_and_independent_audit/
│   │   ├── COMP3710_P4-04A_readiness_20261014/
│   │   │   └── READINESS_AUDIT.md  # 历史独立 Validation 审计：准备度 审计
│   │   ├── COMP3710_P4-04C_audit_20261014/
│   │   │   ├── P4-04D_provenance_20261014/
│   │   │   │   └── P4-04D_PROVENANCE_ADDENDUM.md  # 历史独立 Validation 审计：P4 04D 来源 补遗
│   │   │   ├── reconstructed_tests/
│   │   │   │   ├── analyze_e20.py  # 历史独立 Validation 审计脚本／测试：分析 e20
│   │   │   │   ├── analyze_validation.py  # 历史独立 Validation 审计脚本／测试：分析 验证
│   │   │   │   ├── export_results.py  # 历史独立 Validation 审计脚本：导出 结果
│   │   │   │   ├── inventory.json  # 历史独立 Validation 审计数据／清单：清单
│   │   │   │   ├── package_evidence.py  # 历史独立 Validation 审计脚本／测试：打包 证据
│   │   │   │   ├── run_twice.py  # 历史独立 Validation 审计脚本：运行 两次
│   │   │   │   ├── test_analysis.py  # 合成测试：分析
│   │   │   │   ├── test_e20.py  # 合成测试：e20
│   │   │   │   └── test_export_results.py  # 合成测试：导出 结果
│   │   │   ├── audit_e20_independent.py  # 历史独立 Validation 审计脚本：审计 e20 独立
│   │   │   ├── audit_exports.py  # 历史独立 Validation 审计脚本：审计 exports
│   │   │   ├── audit_provenance.py  # 历史独立 Validation 审计脚本：审计 来源
│   │   │   ├── export_audit.json  # 历史独立 Validation 审计数据／清单：导出 审计
│   │   │   ├── fix_exports.py  # 历史独立 Validation 审计脚本：修复 exports
│   │   │   ├── fix_provenance.py  # 历史独立 Validation 审计脚本：修复 来源
│   │   │   ├── INDEPENDENT_AUDIT.md  # 历史独立 Validation 审计：独立 审计
│   │   │   ├── independent_results.json  # 历史独立 Validation 审计数据／清单：独立 结果
│   │   │   ├── provenance_audit.json  # 历史独立 Validation 审计数据／清单：来源 审计
│   │   │   ├── render_reports.py  # 历史独立 Validation 审计脚本：渲染 reports
│   │   │   ├── SCIENTIFIC_EN.md  # 历史独立 Validation 审计：科研 英文
│   │   │   └── SCIENTIFIC_ZH.md  # 历史独立 Validation 审计：科研 中文
│   │   └── COMP3710_P4-04E_20261014/
│   │       ├── reproducibility_source/
│   │       │   ├── analyze_e20.py  # 历史独立 Validation 审计脚本／测试：分析 e20
│   │       │   ├── analyze_validation.py  # 历史独立 Validation 审计脚本／测试：分析 验证
│   │       │   ├── export_results.py  # 历史独立 Validation 审计脚本：导出 结果
│   │       │   ├── inventory.json  # 历史独立 Validation 审计数据／清单：清单
│   │       │   ├── README.md  # 历史独立 Validation 审计：说明
│   │       │   ├── run_twice.py  # 历史独立 Validation 审计脚本：运行 两次
│   │       │   ├── SHA256SUMS_LOCAL.txt  # 历史独立 Validation 审计日志／校验值：SHA256 校验和 本地
│   │       │   ├── test_analysis.py  # 合成测试：分析
│   │       │   ├── test_e20.py  # 合成测试：e20
│   │       │   └── test_export_results.py  # 合成测试：导出 结果
│   │       ├── audit_e20_independent.py  # 历史独立 Validation 审计脚本：审计 e20 独立
│   │       ├── audit_exports.py  # 历史独立 Validation 审计脚本：审计 exports
│   │       ├── audit_provenance.py  # 历史独立 Validation 审计脚本：审计 来源
│   │       ├── build_closure.py  # 历史独立 Validation 审计脚本：构建 结案
│   │       ├── cpu_test_stdout.txt  # 历史独立 Validation 审计日志／校验值：CPU 测试 stdout
│   │       ├── export_audit.json  # 历史独立 Validation 审计数据／清单：导出 审计
│   │       ├── independent_results.json  # 历史独立 Validation 审计数据／清单：独立 结果
│   │       ├── provenance_audit.json  # 历史独立 Validation 审计数据／清单：来源 审计
│   │       ├── SHA256SUMS.txt  # 历史独立 Validation 审计日志／校验值：SHA256 校验和
│   │       ├── SUPPLEMENT_SOURCE_MANIFEST.json  # 历史独立 Validation 审计数据／清单：补充 源码 清单
│   │       ├── V4_DEPENDENCY_AUDIT.md  # 历史独立 Validation 审计：V4 依赖 审计
│   │       ├── V4_EVIDENCE_CLOSURE.md  # 历史独立 Validation 审计：V4 证据 结案
│   │       ├── V4_P4-04E_SUPPLEMENT_source_only.zip  # 历史独立 Validation 审计证据 ZIP：V4 P4 04E 补充 仅源码
│   │       ├── V4_PROVENANCE_MATRIX.md  # 历史独立 Validation 审计：V4 来源 矩阵
│   │       ├── V4_REPRODUCIBILITY_MANIFEST.json  # 历史独立 Validation 审计结构化记录／清单：V4 复现性 清单
│   │       └── V4_TEST_RESULTS.md  # 历史独立 Validation 审计：V4 测试 结果
│   ├── 09_rangpur_original_evidence/
│   │   ├── P4-03B-evidence-20261013/
│   │   │   ├── P4-03B_INDEPENDENT_AUDIT.md  # 历史Rangpur 原始证据：P4 03B 独立 审计
│   │   │   └── p4_03b_evidence_20261013T134448164139Z.zip  # 历史Rangpur 原始证据证据 ZIP：p4 03b 证据 20261013T134448164139Z
│   │   └── P4-04B-evidence-20261014/
│   │       └── p4_04b_e20_evidence_retry1.zip  # 历史Rangpur 原始证据证据 ZIP：p4 04b e20 证据 retry1
│   ├── 10_scientific_documentation/
│   │   ├── COMP3710_P4-03C_docs_20261014/
│   │   │   ├── historical_snapshots/
│   │   │   │   ├── README.md  # 历史科研文档：说明
│   │   │   │   ├── recognition__CLASS_COLLAPSE_DIAGNOSIS.md  # 历史科研文档：recognition  类别 漏检 DIAGNOSIS
│   │   │   │   ├── recognition__FINAL-REPORT.md  # 历史科研文档：recognition  最终 报告
│   │   │   │   ├── recognition__hipmri_3d_improved_unet_hard__ai.md  # 历史科研文档：recognition  hipmri 3d improved unet hard  ai
│   │   │   │   ├── recognition__hipmri_3d_improved_unet_hard__README.md  # 历史科研文档：recognition  hipmri 3d improved unet hard  说明
│   │   │   │   ├── recognition__IMPROVED_V3_EXPERIMENT.md  # 历史科研文档：recognition  IMPROVED V3 实验
│   │   │   │   ├── recognition__IMPROVED_V3_SPLIT_EXECUTION.md  # 历史科研文档：recognition  IMPROVED V3 分段 执行
│   │   │   │   ├── recognition__README.md  # 历史科研文档：recognition  说明
│   │   │   │   └── recognition__REPORT_CODE_TRACEABILITY.md  # 历史科研文档：recognition  报告 CODE TRACEABILITY
│   │   │   ├── AI_USAGE_DISCLOSURE_ADDENDUM_20261014.md  # 历史科研文档：AI USAGE DISCLOSURE 补遗 20261014
│   │   │   ├── BASELINE_V4_COMPARISON.md  # 历史科研文档：BASELINE V4 对比
│   │   │   ├── CONSISTENCY_AUDIT.md  # 历史科研文档：CONSISTENCY 审计
│   │   │   ├── DOCUMENT_SYNC_MANIFEST.md  # 历史科研文档：文档 SYNC 清单
│   │   │   ├── EVIDENCE_DATA.json  # 历史科研文档数据／清单：证据 数据
│   │   │   ├── FEASIBILITY_ADDENDUM_20261014.md  # 历史科研文档：FEASIBILITY 补遗 20261014
│   │   │   ├── finalize_p403c.py  # 历史科研文档脚本／测试：finalize p403c
│   │   │   ├── GPU_RESOURCE_LIMITATIONS_ADDENDUM_20261014.md  # 历史科研文档：GPU RESOURCE LIMITATIONS 补遗 20261014
│   │   │   ├── P4-03B_AUDIT_SOURCE.md  # 历史科研文档：P4 03B 审计 源码
│   │   │   ├── README.md  # 历史科研文档：说明
│   │   │   ├── README_ADDENDUM_20261014.md  # 历史科研文档：说明 补遗 20261014
│   │   │   ├── render_addenda.py  # 历史科研文档脚本：渲染 addenda
│   │   │   ├── render_p403c.py  # 历史科研文档脚本：渲染 p403c
│   │   │   ├── render_reports.py  # 历史科研文档脚本：渲染 reports
│   │   │   ├── RESEARCH_EXPERIMENT_LOG.md  # 历史科研文档：研究 实验 日志
│   │   │   ├── RESEARCH_PROGRESS_EN.md  # 历史科研文档：研究 进展 英文
│   │   │   ├── RESEARCH_PROGRESS_ZH.md  # 历史科研文档：研究 进展 中文
│   │   │   ├── STATE_PACKET_20261014.md  # 历史科研文档：状态包 20261014
│   │   │   └── V4_INTERMEDIATE_VALIDATION.md  # 历史科研文档：V4 INTERMEDIATE 验证
│   │   ├── COMP3710_V4_CLOSURE_20261014/
│   │   │   ├── audit_e20_independent.py  # 历史科研文档脚本：审计 e20 独立
│   │   │   ├── audit_exports.py  # 历史科研文档脚本：审计 exports
│   │   │   ├── audit_provenance.py  # 历史科研文档脚本：审计 来源
│   │   │   ├── build_v4_closure.py  # 历史科研文档脚本：构建 v4 结案
│   │   │   ├── COMP3710_STATE_PACKET_AFTER_V4.md  # 历史科研文档：COMP3710 状态包 AFTER V4
│   │   │   ├── export_audit.json  # 历史科研文档数据／清单：导出 审计
│   │   │   ├── FINAL_DOCUMENT_QA.txt  # 历史科研文档日志／校验值：最终 文档 QA
│   │   │   ├── FINAL_QA_EXECUTION.txt  # 历史科研文档日志／校验值：最终 QA 执行
│   │   │   ├── INDEPENDENT_AUDIT_RUN.txt  # 历史科研文档日志／校验值：独立 审计 运行
│   │   │   ├── independent_results.json  # 历史科研文档数据／清单：独立 结果
│   │   │   ├── P4-03B_CPU_TESTS.txt  # 历史科研文档日志／校验值：P4 03B CPU 测试
│   │   │   ├── P4-04B_CPU_TESTS.txt  # 历史科研文档日志／校验值：P4 04B CPU 测试
│   │   │   ├── provenance_audit.json  # 历史科研文档数据／清单：来源 审计
│   │   │   ├── run_final_qa.py  # 历史科研文档脚本：运行 最终 qa
│   │   │   ├── SHA256SUMS.txt  # 历史科研文档日志／校验值：SHA256 校验和
│   │   │   ├── V4_AI_USAGE_AND_REPRODUCIBILITY.md  # 历史科研文档：V4 AI USAGE 与 复现性
│   │   │   ├── V4_CLOSURE_MANIFEST.json  # 历史科研文档结构化记录／清单：V4 结案 清单
│   │   │   ├── V4_COMPLETION_CHECKLIST.md  # 历史科研文档：V4 完成 核对清单
│   │   │   ├── V4_EVIDENCE_INDEX.md  # 历史科研文档：V4 证据 INDEX
│   │   │   ├── V4_FINAL_QA_REPORT.md  # 历史科研文档：V4 最终 QA 报告
│   │   │   ├── V4_FINAL_SCIENTIFIC_CONCLUSION_EN.md  # 历史科研文档：V4 最终 科研 结论 英文
│   │   │   ├── V4_FINAL_SCIENTIFIC_CONCLUSION_ZH.md  # 历史科研文档：V4 最终 科研 结论 中文
│   │   │   ├── V4_FINAL_STATUS.md  # 历史科研文档：V4 最终 状态
│   │   │   └── V4_OPEN_ISSUES_AND_LIMITATIONS.md  # 历史科研文档：V4 OPEN ISSUES 与 LIMITATIONS
│   │   └── COMP3710_V4_FINAL_DOCS_20261014/
│   │       ├── build_v4_final_docs.py  # 历史科研文档脚本：构建 v4 最终 docs
│   │       ├── V4_ACCEPTANCE_DECISION.md  # 历史科研文档：V4 验收 决策
│   │       ├── V4_AI_USAGE_DISCLOSURE.md  # 历史科研文档：V4 AI USAGE DISCLOSURE
│   │       ├── V4_CONTROL_TREATMENT_COMPARISON.md  # 历史科研文档：V4 对照 处理 对比
│   │       ├── V4_DOCUMENTATION_VALIDATION.txt  # 历史科研文档日志／校验值：V4 文档 验证
│   │       ├── V4_EXPERIMENT_TIMELINE.md  # 历史科研文档：V4 实验 时间线
│   │       ├── V4_FAILURE_ANALYSIS.md  # 历史科研文档：V4 失败 分析
│   │       ├── V4_FINAL_REPORT_EN.md  # 历史科研文档：V4 最终 报告 英文
│   │       ├── V4_FINAL_REPORT_ZH.md  # 历史科研文档：V4 最终 报告 中文
│   │       ├── V4_REPORT_INTEGRATION_GUIDE.md  # 历史科研文档：V4 报告 INTEGRATION GUIDE
│   │       ├── V4_REPRODUCIBILITY_AND_LIMITATIONS.md  # 历史科研文档：V4 复现性 与 LIMITATIONS
│   │       ├── V4_RESEARCH_LOG_FINAL.md  # 历史科研文档：V4 研究 日志 最终
│   │       └── validate_v4_docs.py  # 历史科研文档脚本／测试：validate v4 docs
│   ├── 11_tests_debug_and_analysis/
│   │   ├── COMP3710_P4-04B_analysis_20261014/
│   │   │   ├── analyze_e20.py  # 历史分析／测试脚本／测试：分析 e20
│   │   │   ├── analyze_validation.py  # 历史分析／测试脚本／测试：分析 验证
│   │   │   ├── export_results.py  # 历史分析／测试脚本：导出 结果
│   │   │   ├── extend_summary.py  # 历史分析／测试脚本／测试：扩展 摘要
│   │   │   ├── fix_optional.py  # 历史分析／测试脚本：修复 可选项
│   │   │   ├── IMPLEMENTATION_LOG.md  # 历史分析／测试：实现 日志
│   │   │   ├── inventory.json  # 历史分析／测试数据／清单：清单
│   │   │   ├── package_evidence.py  # 历史分析／测试脚本／测试：打包 证据
│   │   │   ├── READINESS_STATUS.md  # 历史分析／测试：准备度 状态
│   │   │   ├── README.md  # 历史分析／测试：说明
│   │   │   ├── run_twice.py  # 历史分析／测试脚本：运行 两次
│   │   │   ├── SHA256SUMS_LOCAL.txt  # 历史分析／测试日志／校验值：SHA256 校验和 本地
│   │   │   ├── test_analysis.py  # 合成测试：分析
│   │   │   ├── test_e20.py  # 合成测试：e20
│   │   │   ├── test_export_results.py  # 合成测试：导出 结果
│   │   │   └── update_readme.py  # 历史分析／测试脚本：更新 说明
│   │   ├── hipmri_p4_03a_readonly_analysis_20261013/
│   │   │   ├── analyze_validation.py  # 历史分析／测试脚本／测试：分析 验证
│   │   │   ├── IMPLEMENTATION_LOG.md  # 历史分析／测试：实现 日志
│   │   │   ├── inventory.json  # 历史分析／测试数据／清单：清单
│   │   │   ├── README.md  # 历史分析／测试：说明
│   │   │   └── test_analysis.py  # 合成测试：分析
│   │   └── hipmri_p4_03b_readonly_analysis_20261013/
│   │       ├── analyze_validation.py  # 历史分析／测试脚本／测试：分析 验证
│   │       ├── export_results.py  # 历史分析／测试脚本：导出 结果
│   │       ├── IMPLEMENTATION_LOG.md  # 历史分析／测试：实现 日志
│   │       ├── inventory.json  # 历史分析／测试数据／清单：清单
│   │       ├── README.md  # 历史分析／测试：说明
│   │       ├── run_real_e14_e16.py  # 历史分析／测试脚本：运行 真实 e14 e16
│   │       ├── test_analysis.py  # 合成测试：分析
│   │       └── test_export_results.py  # 合成测试：导出 结果
│   ├── 12_handover_and_pretest_readiness/
│   │   └── COMP3710_P4-05A_20261014/
│   │       ├── build_p4_05a.py  # 历史 测试 前准备脚本：构建 p4 05a
│   │       ├── INFERENCE_PROTOCOL_AUDIT.md  # 历史 测试 前准备：推理 协议 审计
│   │       ├── P4-05A_READINESS_AUDIT.md  # 历史 测试 前准备：P4 05A 准备度 审计
│   │       ├── PRETEST_FREEZE_CHECKLIST.md  # 历史 测试 前准备：测试 前 冻结 核对清单
│   │       ├── PROPOSED_MODEL_FREEZE_RECEIPT.json  # 历史 测试 前准备结构化记录／清单：拟议模型 冻结 收据
│   │       ├── STATE_PACKET_P4_05A.md  # 历史 测试 前准备：状态包 P4 05A
│   │       └── V3_CANDIDATE_IDENTITY.md  # 历史 测试 前准备：V3 候选 身份
│   ├── 13_inventories/
│   │   ├── COMP3710_P5_01_AUDIT_20261014_211851_6c7e5f1f/
│   │   │   ├── build_inventory.py  # 历史文件清单脚本：构建 清单
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # 历史文件清单：COMP3710 目录树 C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # 历史文件清单：COMP3710 目录树 D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # 历史文件清单：COMP3710 GIT REPOSITORY 审计
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # 历史文件清单表格／导出：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # 历史文件清单结构化记录／清单：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # 历史文件清单：COMP3710 P5 01 审计 摘要
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # 历史文件清单：COMP3710 项目 资产 图
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # 历史文件清单：COMP3710 Rangpur 证据 清单
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # 历史文件清单：COMP3710 终端 与 实验 历史
│   │   │   └── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # 历史文件清单：COMP3710 未分类 与 复核 需要
│   │   ├── COMP3710_P5_01_AUDIT_REFINED_20261014_212128_2fc0b2db/
│   │   │   ├── build_inventory.py  # 历史文件清单脚本：构建 清单
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # 历史文件清单：COMP3710 目录树 C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # 历史文件清单：COMP3710 目录树 D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # 历史文件清单：COMP3710 GIT REPOSITORY 审计
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # 历史文件清单表格／导出：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # 历史文件清单结构化记录／清单：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # 历史文件清单：COMP3710 P5 01 审计 摘要
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # 历史文件清单：COMP3710 项目 资产 图
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # 历史文件清单：COMP3710 Rangpur 证据 清单
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # 历史文件清单：COMP3710 终端 与 实验 历史
│   │   │   ├── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # 历史文件清单：COMP3710 未分类 与 复核 需要
│   │   │   └── patch_builder.py  # 历史文件清单脚本／测试：patch builder
│   │   ├── COMP3710_P5_01_DELTA_20261014_212842_f56b8873/
│   │   │   ├── AUDIT_SUMMARY_ZH.md  # 历史文件清单：审计 摘要 中文
│   │   │   ├── build_delta.py  # 历史文件清单脚本：构建 delta
│   │   │   ├── DIRECTORY_TREE_C_PROJECT_ONLY.md  # 历史文件清单：目录树 C 项目 ONLY
│   │   │   ├── DIRECTORY_TREE_D.md  # 历史文件清单：目录树 D
│   │   │   ├── DUPLICATE_VERSION_ANALYSIS.md  # 历史文件清单：重复 版本 分析
│   │   │   ├── FILE_DEPENDENCY_MAP.md  # 历史文件清单：文件 依赖 图
│   │   │   ├── GIT_READONLY_AUDIT.md  # 历史文件清单：GIT READONLY 审计
│   │   │   ├── MASTER_FILE_INVENTORY.csv  # 历史文件清单表格／导出：总 文件 清单
│   │   │   ├── MASTER_FILE_INVENTORY.json  # 历史文件清单数据／清单：总 文件 清单
│   │   │   ├── P5_01_STATE_PACKET.md  # 历史文件清单：P5 01 状态包
│   │   │   ├── PROJECT_STAGE_MAP.md  # 历史文件清单：项目 STAGE 图
│   │   │   ├── RANGPUR_LOCAL_EVIDENCE_MAP.md  # 历史文件清单：Rangpur 本地 证据 图
│   │   │   ├── TERMINAL_EXPERIMENT_HISTORY.md  # 历史文件清单：终端 实验 历史
│   │   │   └── UNKNOWN_AND_UNCLASSIFIED.md  # 历史文件清单：未知 与 未分类
│   │   ├── COMP3710_P5_01_FINAL_20261014_212222_d215a281/
│   │   │   ├── build_inventory.py  # 历史文件清单脚本：构建 清单
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # 历史文件清单：COMP3710 目录树 C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # 历史文件清单：COMP3710 目录树 D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # 历史文件清单：COMP3710 GIT REPOSITORY 审计
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # 历史文件清单表格／导出：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # 历史文件清单结构化记录／清单：COMP3710 总 文件 清单
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # 历史文件清单：COMP3710 P5 01 审计 摘要
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # 历史文件清单：COMP3710 项目 资产 图
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # 历史文件清单：COMP3710 Rangpur 证据 清单
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # 历史文件清单：COMP3710 终端 与 实验 历史
│   │   │   ├── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # 历史文件清单：COMP3710 未分类 与 复核 需要
│   │   │   ├── verify_and_enrich.py  # 历史文件清单脚本：验证 与 enrich
│   │   │   └── verify_final_inventory.py  # 历史文件清单脚本：验证 最终 清单
│   │   └── COMP3710_P5_02_PLAN_20261014_213035_1323b32a/
│   │       ├── P5_02_APPROVAL_READY_SUMMARY_ZH.md  # 历史文件清单：P5 02 批准 就绪 摘要 中文
│   │       ├── P5_02_CONFLICT_AND_REVIEW_LIST.md  # 历史文件清单：P5 02 CONFLICT 与 复核 LIST
│   │       ├── P5_02_DUPLICATE_PROVENANCE_MATRIX.md  # 历史文件清单：P5 02 重复 来源 矩阵
│   │       ├── P5_02_GIT_TRACKING_POLICY.md  # 历史文件清单：P5 02 GIT 跟踪 规则
│   │       ├── P5_02_PER_FILE_MIGRATION_MANIFEST.csv  # 历史文件清单表格／导出：P5 02 PER 文件 迁移 清单
│   │       ├── P5_02_PER_FILE_MIGRATION_MANIFEST.json  # 历史文件清单结构化记录／清单：P5 02 PER 文件 迁移 清单
│   │       ├── P5_02_RANGPUR_ASSET_TRACEABILITY.md  # 历史文件清单：P5 02 Rangpur 资产 TRACEABILITY
│   │       ├── P5_02_REFERENCE_BREAKAGE_RISKS.md  # 历史文件清单：P5 02 引用 断链 风险
│   │       ├── P5_02_RESEARCH_DIRECTORY_DESIGN.md  # 历史文件清单：P5 02 研究 目录 设计
│   │       └── P5_02_TARGET_DIRECTORY_TREE.md  # 历史文件清单：P5 02 TARGET 目录树
│   ├── ARCHIVE_README.md  # 历史研究档案：ARCHIVE 说明
│   ├── MASTER_ARCHIVE_INDEX.md  # 历史研究档案：总 ARCHIVE INDEX
│   ├── MIGRATION_VERIFICATION_REPORT.md  # 历史研究档案：迁移 VERIFICATION 报告
│   ├── ORIGINAL_PATH_MAPPING.csv  # 历史研究档案表格／导出：原路径映射
│   ├── RANGPUR_EVIDENCE_INDEX.md  # 历史研究档案：Rangpur 证据 INDEX
│   ├── RESEARCH_STAGE_NAVIGATION.md  # 历史研究档案：研究 STAGE 导航
│   ├── TEST_AND_DEBUG_INDEX.md  # 历史研究档案：测试 与 调试 INDEX
│   └── UNRESOLVED_DEPENDENCIES.md  # 历史研究档案：UNRESOLVED 依赖
├── .gitignore  # Git 忽略规则：本地证据与生成文件
├── LICENSE.txt  # 仓库许可条款
├── README.md  # 项目总入口与状态
├── W11-EN.md  # 第11周完整英文复盘及英文文件索引
└── W11-ZH.md  # 第11周完整中文复盘及中文文件索引
```
