# v2分析图

本组图由50轮完整JSON及历史Standard/v1的Validation记录复算，用于检查六类取舍、六患者差异和训练后期退步。`all_six_classes`统一采用患者等权，`patient_comparison`先合并患者周次，`training_and_boundary_trajectory`保留所有验证轮次及三个head训练loss。每图提供PNG和SVG。病例MRI另存case_studies，不将这里的统计图冒充空间错误证据。来源哈希在provenance，本次复算脚本为`code/tools/summarize_v2_results.py`。
