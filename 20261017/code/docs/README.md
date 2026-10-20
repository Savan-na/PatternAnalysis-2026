# 方法与执行文档

本目录只保存研究规则、可复现操作及仍有身份约束的协议；当前运行状态见[项目入口](../../README.md)，不在每份协议中复制。

| 文档 | 独立用途 |
|---|---|
| [ALL_MODEL_ASSESSMENT_AUDIT](ALL_MODEL_ASSESSMENT_AUDIT.md) | 已完成的七模型固定best评估方法；当前数值、失败图及课程结论见v4系统报告 |
| [STUDY_PROTOCOL](STUDY_PROTOCOL.md) | 患者隔离、Train/Validation/Test信息边界及研究规则 |
| [METHODS_EXECUTION_HISTORY](METHODS_EXECUTION_HISTORY.md) | 完整保留三篇论文实现决策、G1–G7证据门槛和历史命令；按sources/evidence/commands分节，不自动重跑 |
| [P2_INFERENCE_CONTEXT](INFERENCE_CONTEXT_DIAGNOSTIC_ZH.md) | 固定权重上下文诊断设计，已有结果在results/diagnostics |
| [P2B_PREDICTION_MECHANISMS](PREDICTION_MECHANISM_DIAGNOSTIC_ZH.md) | 机制诊断方案；数值正控失败限制因果解释 |
| [V2_TRAINING](V2_TRAINING.md) | v2独立监督训练与恢复协议，冻结字节保留 |
| [V3_TRAINING](V3_TRAINING.md) | v3 GN实验训练与恢复协议，冻结字节保留 |
| [BEST_CHECKPOINT_AUDIT](BEST_CHECKPOINT_AUDIT.md) | 唯一保留的五模型best统一审计协议，尚未执行完成 |
| [V4_Paired_Training](V4_Paired_Training_ZH.md) | 当前A0/A1空间增强配对实验的冻结操作协议 |

未执行的旧四模型审计、对应测试/配置/提交器/包和重复预案副本已删除。旧单loss预案完整并入[v4方法报告](../../results/experiments/improved_3d_v4/reports/OPTIMIZATION_REVIEW_ZH.md#historical-loss-plan)。最新五模型审计包`best_checkpoint_audit_current_20261020.zip`锁定实际A0/A1协议；之前未执行的旧包已替换。仍保留的冻结文件即使写着当时的“尚未运行”，也不改写原始字节；当前状态由实验入口解释。
