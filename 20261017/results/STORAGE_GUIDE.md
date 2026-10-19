# 文件归档与后续分析位置

| 内容 | 位置 |
|---|---|
| 源码、冻结配置、测试、方法 | code/，保持调用路径 |
| 原始审计和采样明细 | logs/audits/ |
| 三模型 50 轮记录 | logs/training/<arm>/ |
| GPU 及学习能力预检 | logs/profiles/ |
| Train 数据及采样分析 | results/data_analysis/<stage>/ |
| 学习曲线 | results/figures/comparisons/training/ |
| 六类 Validation 指标 | results/figures/comparisons/validation/ |
| 固定 Test 描述图 | results/figures/comparisons/test/ |
| 分割案例与配套 JSON | results/case_studies/comparisons/<split>/<topic>/ |
| 已引用的综合测量 | results/tables/comparisons/，保持原路径 |
| 跨来源分析报告 | results/reports/ |

## 后续预留：不是已有结果

| 分析 | 原始记录 | 图表位置 | 状态 |
|---|---|---|---|
| 推理上下文与 logit | logs/validation/inference_context/<run_id>/ | results/diagnostics/inference_context/ | 已预留说明，未执行 |
| 基底/尖端及连续性 | logs/validation/boundary_continuity/<run_id>/ | results/case_studies/comparisons/validation/boundary_continuity/ | 仅规划 |
| 公平资源比较 | logs/profiles/<benchmark_id>/ | results/figures/comparisons/resources/ | 仅规划 |
| 新受控训练 | logs/training/<experiment_id>/<arm>/ | results/figures/comparisons/<experiment_id>/ | 等诊断决定 |

只为正在准备的诊断建立说明目录，其他路径产生结果时再建，避免空目录堆积。图片组 README 与图同目录；跨来源结论放 reports。MRI 和权重仍留在 Rangpur。

旧远程脚本仍可输出 comparisons 根目录；下载或生成后，从项目根运行 python -B code/tools/organize_result_artifacts.py 分类。遇到同名文件会停止，不自动覆盖。本轮不重新计算指标。
