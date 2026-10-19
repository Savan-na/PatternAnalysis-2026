# P2：固定权重推理上下文诊断

## 当前状态

程序、固定配置和本地验证已准备好，**Rangpur 真实病例尚未运行**。这一步落实 P1 的最高优先级诊断，不是新训练，不生成新的可选模型权重。

英文可行性报告位于仓库最外层：[feasibility review.md](<../../../feasibility review.md>)。

## 固定设计

- 四例 Validation：K018_Week4、B040_Week4、J026_Week4、K008_Week0。不读取 Test 数组或 Test 评估结果。
- 两组不变的 G5 epoch-50 最佳权重，复用严格加载器核对配置、划分、源码及 checkpoint 身份。
- 先逐例精确复现原整卷混淆矩阵，再执行固定 64×64×32 XYZ、50% overlap、均匀 raw-logit 平均的滑窗路径。末端窗口锚定边界，记录实际覆盖。
- 整卷非零 MRI Z-score 不变，不重新按窗口标准化，不做 TTA、阈值调整、GT ROI 或后处理。
- 测量六类 Dice/IoU/Precision/Recall、Body HD95、远处误检、到体积边缘的物理距离和六邻域连通域。
- 分解 Improved 三个 head 的 Body−Background margin，并记录 Body−最大其他类别 margin。
- 两模型都在相同的 Improved 整卷误检体素区域额外比较 margin，图片采用同一事后评估切面。GT 与误检定位不影响推理。

## 运行与输出

传输包：`code/transfer/p2_inference_context_20261020.zip`，只包含六个新增源文件：诊断模块、测试、配置、测量脚本、提交脚本及汇总脚本。

从本机 PowerShell 上传（一次密码）：

```powershell
scp '.\20261017\code\transfer\p2_inference_context_20261020.zip' 's4906923@login0.compute.eait.uq.edu.au:~/p2_inference_context_20261020.zip'
```

进入 Rangpur 后先核对 ZIP SHA-256，检查六个允许的相对文件名并拒绝覆盖，再解压至项目根目录。不要覆盖已有冻结源码。

分别进行语法检查和提交：

```bash
bash -n "$HOME/20261018_g1_stage/20261017/code/tools/submit_p2_inference_context.sh"
```

```bash
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_p2_inference_context.sh"
```

这是十分钟 A100 测试分区作业，运行前再执行五项测试。日志为 `logs/profiles/p2_inference_context_<jobid>.out`。若失败或超时，先检查日志和部分输出，不重复提交或改变窗口。

| 内容 | 路径 |
|---|---|
| 原始逐例记录、manifest 和校验和 | `logs/validation/inference_context/p2_four_cases_20261020/` |
| 8 张 Body 三平面对比、4 张辅助头 margin 图 | `results/diagnostics/inference_context/p2_four_cases_20261020/` |
| 下载后派生六类 CSV、PNG/SVG、中文简报 | 与上行图片同目录 |

只有 `manifest.json` 的 `complete=true` 且八组模型/病例记录完整才算完成。文件存在检查与队列检查可防止误覆盖和重复提交。检查原始结果后，将两个目录分别一次性下载到本机对应目录，每条 SCP 只输入一次密码。

本机在有 NumPy/Matplotlib 的环境、`20261017` 根目录运行：

```bash
python -B code/tools/summarize_p2_inference_context.py
```

汇总脚本检查原始校验和、源码配置及八组完整记录后生成 `P2_review_ZH.md`。本地 Windows 绘图预检通过临时 `MKL_THREADING_LAYER=SEQUENTIAL` 避免运行库冲突，没有关闭安全检查。

## 结果解释关口

四例只探索机制。若 Improved 远处误检减少且小器官保留、Standard 变化较小，再考虑独立标识两臂的新推理协议并扩至 38 例；不能从四例选最终模型，不能把变化单独归因于 InstanceNorm。

若改善很弱或小器官受损，转向已有审计的 G7 单因素采样训练，不扫描滑窗参数。后续结构实验继续保留残差和有意义的 Deep Supervision；已有 Test 只作历史描述。
