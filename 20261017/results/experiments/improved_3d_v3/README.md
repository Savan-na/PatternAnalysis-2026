# Improved 3D v3独立实验区

[完整系统研究报告](reports/RESEARCH_REPORT_ZH.md)按设计、Validation结果、泛化、方法审查四节保留原始分析；不再维护四份互相交叉的报告。

- [最新五模型best图表与指标](../../model_evaluation/best_checkpoint_comparison/README.md)：课程要求与补充研究分区。
- `training_records/`：远端下载的50轮JSON、manifest和冻结配置。
- `tables/`与`provenance/`：v3专属预设门槛、患者增益、原始表与来源校验。
- [generalization_review](../../model_evaluation/generalization_diagnostics/README.md)：患者不确定性、历史Test影响等独立诊断。
- [冻结训练协议](../../../code/docs/V3_TRAINING.md)：源码身份、训练和恢复方式。

旧四模型概览图已在逐项核对数据后删除；五模型best图完整包含其指标，当前学习曲线也保留v3逐轮Body HD95。真实MRI图不能由这些统计图替代，完整固定权重审计仍待执行。

本版本训练完成（作业678535，02:02:48）；具体分数、收益与退步只在系统报告和数值表维护。后续[v4](../improved_3d_v4/README.md)独立保存，不覆盖本版本证据。
