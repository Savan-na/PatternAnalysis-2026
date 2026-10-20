# V4 A0/A1 配对训练协议

冻结日期：2026-10-20。Train/Validation设计，不读取Test调参。未执行远端训练。

## 两个实验

| 项目 | A0 | A1 |
|---|---|---|
| 网络 | v3：GN(1)、残差、原相加深监督 | 完全相同 |
| 优化器 | AdamW，卷积权重decay=1e-5，bias/norm=0 | 相同 |
| 学习率 | 0.001→0.00001，6500更新cosine | 相同 |
| loss | 原0.5 weighted CE+0.5前景Dice | 相同，不压缩类别权重 |
| 空间增强 | 无 | 每patch独立30%概率；世界XYZ轴±10°、各向同性scale 0.9–1.1 |
| 强度增强、翻转、弹性形变 | 无 | 无 |
| 数据/预算 | 原患者划分、患者均衡uniform、seed371017、batch2、50epoch×130更新 | 相同 |

A1−A0检查空间增强在AdamW/cosine配方中的效果；相对历史v3的变化包含优化器和调度等组合，不能单独归因。两组不是新的网络结构比较。两组从头初始化，同一患者顺序、visit、crop中心、模型随机数流；增强使用descriptor独立随机流。相同初始化digest应在运行后核对。

## 几何与曝光

世界毫米空间旋转，经完整NIfTI affine逆映射回体素坐标，兼容不等体素间距。以原patch中心为中心，从完整预处理Train体积采样目标64×64×32 XYZ，随后转CDHW。MRI线性、mask最近邻；超出原图部分填0，逐patch记录填充比例、角度、scale、前后六类计数。无基于GT的拒绝重采样；增强后小器官可能移出patch，不假设每patch保留全部器官。

CPU审计原Train第一轮26患者260对patch，保存六组真实MRI/标签叠加。预设机械门槛：小器官总体体素占比A1/A0≥0.85、平均原图外采样≤10%、完整患者数与六预览；门槛用于拒绝明显坏的增强，不是临床阈值或有效性证明。先查看真实预览，再提交GPU；不因门槛通过就声称对齐及泛化已得到完整临床验证。

## 评价与完成标准

与历史模型相同，每5epoch验证全部38例，以六患者等权小器官Dice选best，平分取早。保存每个epoch权重、记录、best及每batch最新恢复状态。主分析使用best，不强取50。六类Dice/IoU、precision/recall、所有患者、Body HD95尾部/远处误报和资源成本一并报告。

预设实用候选标准：A1较A0患者等权小器官Dice增加≥0.005，至少4/6患者改善；任一前景类Dice下降超过0.01或Body远处FP率/HD95尾部明显恶化则不称全面胜出。阈值是项目决策规则，不是统计显著性或临床标准。历史Standard和v3也必须纳入比较；若只改善A0而仍不及v3，应如实报告。单seed不能证明稳定泛化；胜出后再复核seed，不进行Test导向搜索。

## 恢复与日志

学习率由完成更新数唯一确定，checkpoint保存计数、调度协议、AdamW状态及模型/CPU/CUDA/NumPy RNG；增强无隐含随机状态。每batch记录使用LR、下一LR及两patch变换；恢复核对代码/config/数据根路径/运行时版本和LR。每epoch保存immutable快照。6500步余弦终点固定，不用旧延长epoch开关；未来延训需要独立声明新调度协议并保留历史。

本地测试涵盖不等间距配对几何、identity、合法标签、stateless重放、拒绝非Train，以及CPU AdamW中断恢复与连续训练逐参数一致。真实MRI审计及CUDA完整恢复尚待远端执行，不能以CPU测试替代。

## 操作

1. 上传并校验传输包，安全解压新文件。
2. `bash tools/submit_v4_paired.sh audit`：CPU测试及真实Train预览，不训练。
3. 查看审计、下载预览核查，然后 `bash tools/submit_v4_paired.sh train`：A0/A1各自排队50epoch。
4. 中断后分别用 `resume-a0` / `resume-a1`，避免覆盖/重复提交。

远端权重：`logs/training/improved_3d_v4_a{0,1}_20261020/`。
本地分析隔离在 `results/experiments/improved_3d_v4/a0_control/` 与 `a1_spatial/`；跨两组分析放 `paired_analysis/`。历史v1/v2/v3不改动、不归档删除。
