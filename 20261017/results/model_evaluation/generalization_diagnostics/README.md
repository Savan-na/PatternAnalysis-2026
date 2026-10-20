# 泛化与失败机制复核

本区保存患者级统计、历史Test影响分析及总体审查；原始训练记录不修改。`tables/audit.json`复核患者ID互斥、六患者配对bootstrap与leave-one-out；`figures/patient_influence`显示平均值可能掩盖的患者影响。bootstrap仅基于六个既定患者，不校正反复Validation选模，不是独立显著性证明。

v3完整Validation边界与真实图像、新固定checkpoint描述性Test评估尚待远端任务。Test已被查看，不能指导v4；v4预案已另行在Test前固定。没有新增训练或清理历史数据。
