# V4 A0/A1配对训练

## 执行身份与最近状态

两组均完成50epoch、6500次更新，最佳轮次均为45。初始679166/679167在HOME空间耗尽时失败；保留checkpoint后恢复作业679977/679982均COMPLETED/0:0。统一七模型评估680798也COMPLETED/0:0（20分26秒），37个下载文件校验通过。这是最近确认状态，不是实时查询。

| 组别 | 唯一组间差异 | 远端run | 完成后本地位置 |
|---|---|---|---|
| A0 | 不加空间增强 | `logs/training/improved_3d_v4_a0_20261020/` | `a0/training_records/` |
| A1 | 加温和配对3D空间增强 | `logs/training/improved_3d_v4_a1_20261020/` | `a1/training_records/` |

共同设置及恢复检查以[冻结协议](../../../code/docs/V4_Paired_Training_ZH.md)为准：v3网络、AdamW、cosine；本次无强度增强、无CE权重压缩、无G7角落采样。不要重新提交正在运行的作业。

## 本区内容

- [最终七模型与v4系统报告](reports/RESEARCH_REPORT_ZH.md)：性能、边界、真实失败、泛化、成本与课程要求判定。
- [配对记录与Train增强审计复核](paired_analysis/README.md)：机械门槛、六张真实预览、训练记录初步分析及局限。
- `assessment_analysis/`：当前七模型已核验的课程指标与补充研究图表/CSV，两个分区独立。
- `comprehensive_audit_20261021/`：原始七模型评估、37文件校验清单和真实分割PNG，不改写。
- `augmentation_audit_20261020/`：原始审计JSON、checksums及六张图，保持下载内容。
- [优化方法审查](reports/OPTIMIZATION_REVIEW_ZH.md)：候选列表与决策理由，不等于全部已实施。
- `paired_analysis/`：后续统一best比较及对Standard/v3的收益、退步和计算成本。

A0/A1本地目录已有真实下载记录。旧单loss预案完整保留在[方法历史](reports/OPTIMIZATION_REVIEW_ZH.md#historical-loss-plan)，不冒充本次执行条件。结论以系统报告为准：A1远处Body误报减少，但没有取得主要验证指标优势；不立即删除历史权重或开始下一次训练。
