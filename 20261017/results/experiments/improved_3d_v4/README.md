# V4 A0/A1配对训练

## 执行身份与最近状态

两组均申请50epoch、6500次更新。按用户最近`sacct`输出：A0作业679166为RUNNING，已用02:13:53；A1作业679167为RUNNING，已用02:10:45。尚未收到完成日志。这是最后确认状态，不是实时查询。

| 组别 | 唯一组间差异 | 远端run | 完成后本地位置 |
|---|---|---|---|
| A0 | 不加空间增强 | `logs/training/improved_3d_v4_a0_20261020/` | `a0_control/` |
| A1 | 加温和配对3D空间增强 | `logs/training/improved_3d_v4_a1_20261020/` | `a1_spatial/` |

共同设置及恢复检查以[冻结协议](../../../code/docs/V4_Paired_Training_ZH.md)为准：v3网络、AdamW、cosine；本次无强度增强、无CE权重压缩、无G7角落采样。不要重新提交正在运行的作业。

## 本区内容

- [Train增强审计复核](paired_analysis/README.md)：机械门槛、六张真实预览、曝光及局限。
- `augmentation_audit_20261020/`：原始审计JSON、checksums及六张图，保持下载内容。
- [优化方法审查](reports/OPTIMIZATION_REVIEW_ZH.md)：候选列表与决策理由，不等于全部已实施。
- `paired_analysis/`：后续统一best比较及对Standard/v3的收益、退步和计算成本。

A0/A1本地目录在实际结果导回时创建，不以两个三行README占位。旧单loss预案只保留[审计包所需的完整冻结副本](reports/OPTIMIZATION_REVIEW_ZH.md#historical-loss-plan)，不再复制到本区。尚无v4质量结果，不宣称增强有效或模型已经优于基线。
