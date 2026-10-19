# 推理上下文诊断

将存放固定权重下全体积与训练尺寸滑窗推理的 Validation 对比、logit 和辅助输出贡献图。用于检验输入范围与归一化是否参与 Body 误检，当前不能认定根因。

原始记录预留 logs/validation/inference_context/<run_id>/，必须记录权重哈希、窗口、重叠率和融合方式。

四例固定权重 P2 的设计、测试与远程操作见 [执行说明](../../../code/docs/P2_INFERENCE_CONTEXT.md)。完成后，每次实验的图与简报放在独立 run_id 子目录。

P2 已完成：[四例结果与中文分析](p2_four_cases_20261020/P2_review_ZH.md)。上下文敏感性得到支持，但有小器官损伤，未采用滑窗作为替代协议。
