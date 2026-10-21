# HipMRI 文档导航

更新：2026-10-21。本文区分当前执行、历史方案、实验结果与固定证据；全部文件清单见[PROJECT_MAP](PROJECT_MAP.md)，逐Markdown审查见[整理记录](results/documentation_audit/README.md)。

## 1. 建议阅读顺序

以研究主题组织项目，读者无需了解内部阶段编号。模型版本保留，是可复现的实验身份。

| 阶段 | 已完成的作用 | 当前如何使用 |
|---|---|---|
| 数据质量与分布 | 原始Train完整性、标签、患者和强度分布 | data_analysis：按具体问题命名 |
| 预处理与采样验证 | 空间保持、患者均衡、边界曝光 | 采样可行性不等于模型质量改善 |
| 方法与预测机制诊断 | 课程、论文、推理范围和输出行为审查 | diagnostics及方法审查报告 |
| 训练与性能评估 | 初始2D/Standard/v1训练和既有Test评估 | 原始run证据与当前best比较分开 |
| v2/v3 | 独立监督、GN等实际受控训练 | 保留完整版本报告和真实结果 |
| v4 A0/A1 | AdamW/cosine下空间增强配对实验及七模型统一评估 | 两组50轮及评估完成；最新结论见v4系统报告 |

| 问题 | 文档 |
|---|---|
| 向导师或助教汇报当前研究 | [中文版阶段沟通报告](results/reports/SUPERVISOR_PROGRESS_BRIEF_ZH.md)、[English progress brief](results/reports/SUPERVISOR_PROGRESS_BRIEF_EN.md) |
| 项目现在进展到哪里 | [项目入口](README.md)、[v4执行状态](results/experiments/improved_3d_v4/README.md) |
| 各模型最好的结果如何比较 | [七模型最新系统报告](results/experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)、[选模/课程/论文历史复核](results/experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md#methods) |
| 全部10模型的最新图表在哪里 | [模型性能对比图库](results/reports/model_performance_comparison/README.md)：最佳Validation轮次、六类和边界；v5 Test尚未评估 |
| v2为什么没有达到目标 | [v2结果](results/experiments/improved_3d_v2/reports/RESEARCH_REPORT_ZH.md#results)、[真实图像审阅](results/experiments/improved_3d_v2/case_studies/validation/review_45_50/README.md) |
| v3改善和代价是什么 | [v3结果](results/experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md#validation)、[泛化复核](results/experiments/improved_3d_v3/reports/RESEARCH_REPORT_ZH.md#generalization) |
| 当前v4到底改什么 | [冻结配对协议](code/docs/V4_Paired_Training_ZH.md)、[Train增强复核](results/experiments/improved_3d_v4/paired_analysis/README.md) |
| 还有哪些优化值得考虑 | [v4优化方法综述](results/experiments/improved_3d_v4/reports/OPTIMIZATION_REVIEW_ZH.md)，候选不等于已实施 |
| 下一次实验具体如何选择 | [v5研究依据](results/experiments/improved_3d_v5/reports/RESEARCH_PLAN_ZH.md)、[C/L/S 三分支执行协议](code/docs/V5_PARALLEL_SCREEN_ZH.md)：已准备，未提交训练 |
| 查看课程要求、原始研究治理 | [综合报告索引](results/reports/README.md)、[方法文档索引](code/docs/README.md) |
| 查原始数据问题和推理机制 | [Train审计](results/data_analysis/README.md)、[推理与预测机制诊断](results/diagnostics/inference_context/README.md) |
| 文件应该放在哪里 | [存储指南](results/STORAGE_GUIDE.md) |

## 2. 分类与命名规则

- `README.md`：所在目录的入口、数据来源、用途、状态和下一层链接。图片组的简短解释与图片放在一起。
- `results/reports/`：跨数据来源的综合研究、课程对齐、研究治理；不再堆入单一版本的设计报告。
- `results/experiments/improved_3d_vN/reports/`：版本专属方案、文献设计及结果报告；标题应区分“方案”“结果”“复核”。
- `case_studies/<split>/<group>/`、`diagnostics/<topic>/<run>/`：案例/诊断解读与其图片就近保存。长期综合结论引用这些证据，不复制图像。
- `code/docs/`：运行方法、恢复说明、冻结协议；运行日志/指标不是方法文档。
- `logs/`、`training_records/`：原始机器记录和下载副本，保持run身份；不把报告覆盖到原始JSON。
- 目录使用小写英文`snake_case`，表达研究问题，例如`training_distribution`、`preprocessing_validation`、`best_checkpoint_comparison`。
- Markdown使用`UPPER_SNAKE_CASE[_ZH].md`，目录入口为`README.md`。版本报告统一名为`RESEARCH_REPORT_ZH.md`，版本由父目录表达；日期仅用于区分实际诊断run。
- Python/配置/脚本使用描述性小写名称；现有冻结执行文件、原始run ID和下载文件名是明确例外，不为改名破坏恢复SHA或实验来源。
- 根目录`feasibility review.md`按用户指定名称保留；用户治理原文改用`RESEARCH_GOVERNANCE_ZH.md`保存，正文不删减。
- [路径迁移登记](code/config/artifact_paths.json)记录旧名到主题名的对应。旧编号仅在原始指令、历史命令、run证据和冻结源码中出现，当前索引和新产物不再依靠它们命名。

## 3. 当前、历史与固定快照

当前训练最佳比较：2D50、Standard50、v1-50、v2-45、v3-50、A0-45、A1-45，均按相同已记录Validation候选规则。七模型原始评估与最新分析保存在v4专区；旧五模型表作为历史快照保留，不回写原始来源。

早期数据审查、机制诊断及旧v4单loss预案属于研究过程记录。其“下一步”可能已被后续方案替代，应从索引确认状态，不能直接照抄历史命令重复提交。

仍在使用或保存唯一身份的`code/docs/`协议及下载图片组README受到传输manifest/checksums保护，保留其原始字节；未执行且已被完整替代的旧四模型协议及重复副本已按台账删除。状态更新记录最近已确认的事件，不冒充实时队列查询。

十二份长文已完整合并为四份主题文档；旧v4单loss预案另完整并入v4方法报告，同主题只保留一个报告入口。图片说明仍就近，必要目录索引保留，不为减少文件数把所有内容塞进一个巨型Markdown。历史分析可能引用相同事实，但不再单独维护整份重复报告。

## 4. 更新方式

新增结果先更新对应实验README，再更新必要的综合结论；运行`python code/tools/audit_markdown.py`检查Markdown文件与相对链接，然后运行`python code/tools/update_project_map.py`更新完整树。不要手工改自动生成清单。Markdown自动检查不能判断临床正确性，也不联网验证外部链接。

GitHub/source-only检出使用`python code/tools/audit_markdown.py --repository-only`：真正断链仍报错，按Git规则排除的临床图及原始运行附件单独列为本地材料，不伪装成已上传。
