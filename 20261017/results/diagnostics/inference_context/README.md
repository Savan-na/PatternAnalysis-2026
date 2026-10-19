# 推理上下文诊断（预留，未执行）

将存放固定权重下全体积与训练尺寸滑窗推理的 Validation 对比、logit 和辅助输出贡献图。用于检验输入范围与归一化是否参与 Body 误检，当前不能认定根因。

原始记录预留 logs/validation/inference_context/<run_id>/，必须记录权重哈希、窗口、重叠率和融合方式。
