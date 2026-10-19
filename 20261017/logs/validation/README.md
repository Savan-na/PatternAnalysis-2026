# Validation 诊断原始记录

既有选模记录在 logs/training/<arm>/epoch_*.json，Body 与辅助输出审计保留在 results/tables/comparisons/。

下一步预留 inference_context/<run_id>/ 保存逐病例测量、配置及源码和权重哈希，目前未运行。其他诊断在产生结果时建立独立目录。Test 记录不放这里。
