# HipMRI Segmentation Research

本项目是COMP3710医学图像分割研究。当前方法、实验和证据按明确研究主题组织，从本入口查找。

## 当前进度

- 数据：26 Train、6 Validation、6 Test患者；患者划分固定。
- 已完成：2D、Standard 3D、Improved v1/v2/v3、v4 A0/A1各50轮。主比较使用各自Validation最佳checkpoint，不自动取末轮。
- 2026-10-21：[七模型统一评估与真实案例分析](results/experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)完成；476次评估、266次Validation矩阵及60次历史Test矩阵核对通过。v3仍为既定Validation准则下的主候选，A1未超过它。
- 已补齐：七模型六类Test Dice/IoU、表面/端部/连续性与统一资源测量。待核实：最终交付入口、图片访问、midpoint和upstream PR/review；不宣称已获得全部课程标准或临床安全认证。
- 已查看的Test不参与方案、超参数或checkpoint选择；后续复测不称为新的盲测。

## 阅读入口

| 目的 | 入口 |
|---|---|
| 找报告、图片和方法 | [文档导航](DOCUMENTATION_INDEX.md) |
| 当前已完成模型的完整比较 | [七模型综合报告](results/experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)、[统一六类图](results/experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/six_class_validation_test.svg) |
| 完整研究论证及原始阶段分析 | [研究报告区](results/reports/README.md) |
| 源码、协议和恢复 | [code](code/README.md) |
| 原始机器记录 | [logs](logs/README.md) |
| 存储位置 | [存储指南](results/STORAGE_GUIDE.md) |
| 全部文件 | [PROJECT_MAP](PROJECT_MAP.md) |
| 本轮删除、合并和保留依据 | [清理记录](results/documentation_audit/README.md) |

原始MRI和权重主要在Rangpur；本地保存下载记录、分析和案例。外围的[英文可行性审查](<../feasibility review.md>)保留其独立课程交付用途。`experiment_store.zip`含旧项目唯一历史材料，按用户决定保留，不作为本次新实验结果。

## GitHub中的内容

发布源码、冻结配置、完整研究文档和统计摘要SVG/CSV。原始MRI、权重、训练JSON和临床分割PNG留在本地或Rangpur；报告中的临床图链接是本地分析附件，不表示仓库包含这些图像。主题路径及旧来源对应见[命名规则](DOCUMENTATION_INDEX.md#2-分类与命名规则)。

## Artificial Intelligence Usage Disclosure

本项目使用ChatGPT/Codex协助代码实现与重构、诊断脚本、测试、图表和报告整理。Rangpur命令由用户在其账户执行；AI根据回传日志和下载记录开展分析。验证包括单元测试、源码/产物SHA、患者划分与混淆矩阵复现、指标重算和指定真实分割图审阅；失败诊断和负结果保留。未完成的指标、因果机制及临床专家确认没有写成已验证结论。用户仍需理解、审阅并对最终提交负责；AI辅助与自动核验不能代替独立专业审查。
