# 最佳checkpoint泛化审计

七个模型各自按已记录Validation选择best；38卷Validation、30卷已被查看的Test。所有Validation矩阵复现，Standard/v1 Test矩阵复现。记录六类矩阵、五类表面距离、基底/尖端、连通分量及面积过渡、远处误报和隔离推理成本。Test不是新盲测，不能用于后续调参。one-empty表面状态需单列，不能静默排除。指标不代表临床安全。
