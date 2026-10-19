# G7：原均匀采样与 9+1 边缘采样

这组图表来自 Rangpur CPU 作业 `676249`，覆盖 143 个 Train volume 的
50 轮固定患者日程，每个方案 13,000 个 patch。它用于检查候选采样是否增加
体积边缘的有内容背景，同时保住六类标签曝光；它没有训练模型，也不表示分割已改善。

[类别与空间覆盖图](g7_exposure_and_geometry.png)同时显示六类体素份额、类别命中率
和外侧 2/8 体素区域覆盖。[位置图](g7_crop_origins.png)比较原方案的全部 crop
与候选方案中替换的 1,300 个边缘 crop；位置按可用裁剪范围归一化。
精确数值在 [class_exposure.csv](class_exposure.csv) 和
[patient_class_exposure.csv](patient_class_exposure.csv)。所有预设门槛通过，
但小器官体素曝光仍下降约 4%–5%，需在后续训练中观察其代价。

原始来源为 `logs/audits/g7_edge_sampling_full_20261019/` 的 `manifest.json`
和 `patches.csv`，两者哈希见 [verification.json](verification.json)。本机程序
核对了 26,000 行、11,700 对保留位置、50 轮原训练曝光和候选内容门槛；
物理距离的逐体素定义由远端审计计算，本机未读取 MRI 重新计算距离。
复现命令为项目根目录下 `python -B code/tools/summarize_g7_edge_sampling.py`。
