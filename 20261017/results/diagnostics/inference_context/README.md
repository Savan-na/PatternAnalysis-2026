# 推理范围与预测机制诊断

本区研究固定权重模型为何产生远处Body误检，区分可观测结果、实现问题与未证明的机制。

| 诊断 | 证据与状态 |
|---|---|
| [输入范围敏感性](context_sensitivity_20261020/CONTEXT_SENSITIVITY_REVIEW_ZH.md) | 四例全体积/滑窗对照已完成；远处误检下降，但小器官受损，未替换正式推理 |
| [空间与logit证据](prediction_mechanisms_20261020/README.md) | 原始预测记录衍生的描述性定位，不是单独因果证明 |
| [数值回放审查](prediction_mechanisms_20261020/NUMERICAL_REPLAY_REVIEW_ZH.md) | 正控失败后停止机制归因，保留失败及数值审计证据 |

方法见[推理范围协议](../../../code/docs/INFERENCE_CONTEXT_DIAGNOSTIC_ZH.md)和[预测机制协议](../../../code/docs/PREDICTION_MECHANISM_DIAGNOSTIC_ZH.md)。原始logs中的历史run ID及checksum身份不更名；图和分析目录采用明确主题＋日期。
