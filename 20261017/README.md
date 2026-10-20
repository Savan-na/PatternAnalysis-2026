# HipMRI Segmentation Research

本项目是COMP3710医学图像分割研究。当前方法、实验和证据按明确研究主题组织，从本入口查找。

## 当前进度

- 数据：26 Train、6 Validation、6 Test患者；患者划分固定。
- 已完成：2D、Standard 3D、Improved v1/v2/v3各50轮。主比较使用各自Validation最佳checkpoint，不自动取末轮。
- 正在执行：[v4 A0/A1配对训练](results/experiments/improved_3d_v4/README.md)。两组最近均为RUNNING；尚未收到完成记录。
- 待补：固定权重的五模型统一边界、资源及描述性Test审计。保留的提交入口是`code/tools/submit_best_checkpoint_audit.sh`，未执行的旧四模型入口已删除。
- 已查看的Test不参与方案、超参数或checkpoint选择；后续复测不称为新的盲测。

## 阅读入口

| 目的 | 入口 |
|---|---|
| 找报告、图片和方法 | [文档导航](DOCUMENTATION_INDEX.md) |
| 当前已完成模型的完整比较 | [五模型best分析](results/model_evaluation/best_checkpoint_comparison/README.md) |
| 完整研究论证及原始阶段分析 | [研究报告区](results/reports/README.md) |
| 源码、协议和恢复 | [code](code/README.md) |
| 原始机器记录 | [logs](logs/README.md) |
| 存储位置 | [存储指南](results/STORAGE_GUIDE.md) |
| 全部文件 | [PROJECT_MAP](PROJECT_MAP.md) |
| 本轮删除、合并和保留依据 | [清理记录](results/documentation_audit/README.md) |

原始MRI和权重主要在Rangpur；本地保存下载记录、分析和案例。外围的[英文可行性审查](<../feasibility review.md>)保留其独立课程交付用途。`experiment_store.zip`含旧项目唯一历史材料，按用户决定保留，不作为本次新实验结果。

## GitHub中的内容

发布源码、冻结配置、完整研究文档和统计摘要SVG/CSV。原始MRI、权重、训练JSON和临床分割PNG留在本地或Rangpur；报告中的临床图链接是本地分析附件，不表示仓库包含这些图像。主题路径及旧来源对应见[命名规则](DOCUMENTATION_INDEX.md#2-分类与命名规则)。
