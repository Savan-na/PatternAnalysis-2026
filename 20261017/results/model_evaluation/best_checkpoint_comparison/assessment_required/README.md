# 课程要求相关指标与图表

本组以五模型的Validation best比较，完整覆盖Background、Body、Bones、Bladder、Rectum、Prostate。课程明确要求Test逐类Dice/IoU、资源及失败/关键边界分析；当前新Test数据尚未生成，不能把此处Validation图当作Test。

| 图表/表格 | 用途 | 状态 |
|---|---|---|
| figures/six_class_dice_iou | 患者等权六类Dice/IoU；CSV同时给pooled | 已生成 |
| figures/learning_curves_and_lr | Validation曲线、可比3D loss、五模型真实lr及v2/v3逐轮Body HD95 | 已生成；覆盖原v3轨迹图，v2单独监督loss不作同量比较 |
| figures/body_boundary_by_case | 已有四个3D best的Body HD95与患者分布 | 已生成；2D边界待统一评估 |
| tables/resources_measured_and_pending | 静态真实参数量、实际Train峰值及IO混合时间 | 已生成；统一推理VRAM/latency仍pending |
| tables/selected_checkpoints.json | 最佳轮次、分数、来源SHA、候选轮次 | 已核对 |
| base/apex、危险界面、连续性及3–5例autopsy | 回答课程工程问题 | 历史图/部分Test指标已有；五模型同口径的新结果待远端 |

HD95为本项目支持边界分析的指标选择；不是课程要求尝试所有可用surface指标。正式Test比较、统一成本和新v3实际图仍是缺口。本轮没有把缺失值当0，也没有用训练峰值伪装成推理显存。

远端审计导回后，`summarize_best_checkpoint_audit.py`会生成`audited_`前缀表格、Test/Validation图和上级`audited_coverage.json`。该文件存在且校验通过时，以其中实际覆盖为准；当前没有该文件。
