# 文件存储规则

这里定义路径，不复制模型分数或实时队列状态。最新进度见[项目入口](../README.md)，按问题查报告见[文档导航](../DOCUMENTATION_INDEX.md)。

| 内容 | 位置与约束 |
|---|---|
| 训练源码、测试、冻结配置 | `code/`；历史训练源码仍用于复现与恢复，不随结果优劣删除 |
| 运行方法与冻结协议 | `code/docs/`；当前v4协议保持原始SHA |
| 原始数据审计/采样记录 | `logs/audits/` |
| 原始GPU预检与Slurm记录 | `logs/profiles/` |
| 初始三模型下载记录 | `logs/training/{baseline_2d,baseline_3d,improved_3d}/` |
| v2/v3下载记录 | 各自`results/experiments/<version>/training_records/`，不再搬移来源 |
| Train问题与解释图 | `results/data_analysis/<topic>/` |
| 机制诊断 | `logs/validation/inference_context/<run>/`和`results/diagnostics/inference_context/<run>/` |
| 五模型best历史图与表 | `results/model_evaluation/best_checkpoint_comparison/`，作为旧阶段快照保留 |
| v4分组及共同分析 | `results/experiments/improved_3d_v4/{a0,a1,paired_analysis}/`；分组内training_records是下载原始记录 |
| 当前七模型评估与分析 | v4专区内`comprehensive_audit_20261021/`保留原始输出，`assessment_analysis/{assessment_required,supplementary_research}/`分表格与图表 |
| 本地完整权重备份 | `archives/remote_training_backups/`；Git忽略，清理远端前逐文件SHA核验，不随统计结果优劣删除 |
| 真实案例图片 | `case_studies/<split>/<topic>/`或版本专属案例目录，README与图同组 |
| 完整分析报告 | 综合报告`results/reports/`，版本报告`experiments/<version>/reports/` |
| 文件清理台账 | `results/documentation_audit/`，保留删除SHA、替代来源和合并行数 |
| 唯一旧项目压缩包 | 外围`experiment_store.zip`，保留，不混入新实验比较 |

七模型基底/尖端、连续性及统一资源测量已完成，结果及局限见v4系统报告。机器记录JSON、MRI、权重、来源SHA不能被简短报告取代。只删除已确认重复或被完整替代的派生产物。

从旧远端工具下载的平铺comparisons结果，可运行`python code/tools/organize_result_artifacts.py`分类；同名冲突会停止。新v4直接按版本输出，不混入旧comparisons。
